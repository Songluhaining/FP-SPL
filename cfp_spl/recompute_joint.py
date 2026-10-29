from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .configurations import load_configurations
from .evaluation import _metrics
from .factor_graph import load_factor_graph
from .feature_sources import load_feature_map
from .joint_graph import build_joint_counterfactual_graph
from .spectrum import load_spectrum_priors
from .tensor_inference import infer_factor_graph


def _test_graph(version: Path, config: str, identity: str) -> Path:
    class_name, _, method = identity.rpartition(".")
    return version / "configurations" / config / "runs" / class_name.replace(".", "_") / method / "factor-graph.tsv"


def _labels(labels_root: Path) -> dict[tuple[str, str], int]:
    result = {}
    for path in labels_root.glob("*/variant_labels.csv"):
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("FP transformed from F", "") != "1" and row["LABEL"] in {"FP", "TP"}:
                result[(path.parent.name, row["VARIANT"])] = int(row["LABEL"] == "FP")
    return result


def _evaluate_group(keys: list[tuple[str, str]], truth: dict, scores: dict,
                    threshold: float, budget: int | None = None) -> dict:
    ordered = sorted(keys)
    labels = [truth[key] for key in ordered]
    values = [scores[key] for key in ordered]
    if budget is None:
        predicted = [int(value >= threshold) for value in values]
    else:
        ranked = sorted(ordered, key=lambda key: (-scores[key], key[0], key[1]))
        selected = set(ranked[:budget])
        predicted = [int(key in selected) for key in ordered]
    return {
        "configurations": len(ordered), "ground_truth_false_passing": sum(labels),
        "predicted_positive": sum(predicted), **_metrics(labels, predicted, values),
    }


def run(cache: Path, dataset: Path, labels_root: Path, output: Path,
        base_versions_file: Path, threshold: float = 0.55,
        pulp_predictions: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    truth = _labels(labels_root)
    scores = {key: 0.0 for key in truth}
    usable = {key: 0 for key in truth}
    base_versions = set(json.loads(base_versions_file.read_text(encoding="utf-8")))
    diagnostics = Counter()
    pair_rows = []
    errors = []

    @lru_cache(maxsize=256)
    def graph(path_text: str):
        value = load_factor_graph(Path(path_text))
        for node in value.values.values():
            if node.get("observed") and min(abs(node["correctness"]), abs(node["correctness"] - 1.0)) > 1e-8:
                raise ValueError("旧缓存中的观测节点缺少可恢复的真假值")
        return value

    versions = sorted(path for path in cache.iterdir() if path.is_dir() and path.name.startswith("_"))
    for position, cached_version in enumerate(versions, 1):
        data_version = dataset / cached_version.name
        configs = {item.name: item for item in load_configurations(data_version)}
        plans = json.loads((cached_version / "pair-plan.json").read_text(encoding="utf-8"))
        feature_maps = {}
        spectra = {}

        def feature_map(name: str):
            if name not in feature_maps:
                feature_maps[name] = load_feature_map(configs[name].variant_dir)
            return feature_maps[name]

        def spectrum(name: str):
            if name not in spectra:
                spectra[name] = load_spectrum_priors(configs[name].variant_dir)
            return spectra[name]

        for plan in plans:
            a, b = plan["passing_configuration"], plan["failing_configuration"]
            key = (cached_version.name, a)
            if key not in truth or not plan.get("passing_test"):
                diagnostics["missing_counterfactual_test"] += 1
                continue
            path_b = _test_graph(cached_version, b, plan["failing_test"])
            path_a = _test_graph(cached_version, a, plan["passing_test"])
            if not path_a.is_file() or not path_b.is_file():
                diagnostics["missing_cached_graph"] += 1
                continue
            try:
                joint = build_joint_counterfactual_graph(
                    graph(str(path_b)), graph(str(path_a)), feature_map(b), feature_map(a),
                )
                priors = dict(spectrum(a))
                priors.update(spectrum(b))
                inferred = infer_factor_graph(joint.graph, priors, {})
                candidates = [
                    joint.graph.statements[node].suspiciousness
                    for node in joint.shared_statements
                    if joint.statement_source[node] in joint.failing_causal_sources
                ]
                score = max(candidates, default=0.0)
                scores[key] = max(scores[key], score)
                usable[key] += 1
                diagnostics["usable_joint_pairs"] += 1
                diagnostics["exact_joint_pairs" if inferred.exact else "bounded_joint_pairs"] += 1
                diagnostics["converged_joint_pairs" if inferred.converged else "unconverged_joint_pairs"] += 1
                diagnostics["shared_statement_variables"] += len(joint.shared_statements)
                pair_rows.append({
                    "version": cached_version.name, "passing_configuration": a,
                    "failing_configuration": b, "failing_test": plan["failing_test"],
                    "passing_test": plan["passing_test"], "joint_score": score,
                    "shared_statements": len(joint.shared_statements),
                    "induced_width": inferred.induced_width, "exact": inferred.exact,
                    "converged": inferred.converged,
                })
            except Exception as exc:
                diagnostics["joint_errors"] += 1
                errors.append({**plan, "error": str(exc)})

        version_keys = sorted(key for key in truth if key[0] == cached_version.name)
        version_out = output / cached_version.name
        version_out.mkdir(exist_ok=True)
        with (version_out / "ranking.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("configuration", "score", "usable_joint_pairs"))
            writer.writeheader()
            for key in sorted(version_keys, key=lambda item: (-scores[item], item[1])):
                writer.writerow({"configuration": key[1], "score": scores[key],
                                 "usable_joint_pairs": usable[key]})
        print(f"[{position}/{len(versions)}] {cached_version.name}: {len(version_keys)} configurations")

    all_keys = sorted(truth)
    groups = {
        "all": all_keys,
        "base_feature": [key for key in all_keys if key[0] in base_versions],
        "optional_feature": [key for key in all_keys if key[0] not in base_versions],
    }
    fixed = {name: _evaluate_group(keys, truth, scores, threshold) for name, keys in groups.items()}
    report = {
        "scope": {
            "versions": len(versions), "configurations": len(all_keys),
            "false_passing": sum(truth.values()), "true_passing": len(truth) - sum(truth.values()),
            "configurations_with_usable_joint_pair": sum(value > 0 for value in usable.values()),
            "configurations_without_usable_joint_pair": sum(value == 0 for value in usable.values()),
        },
        "cache_audit": dict(diagnostics),
        "fixed_joint_posterior_threshold": threshold,
        "fixed_threshold_results": fixed,
    }
    if pulp_predictions and pulp_predictions.is_file():
        rows = list(csv.DictReader(pulp_predictions.open(encoding="utf-8-sig")))
        pulp = {(row["version"], row["variant"]): int(row["pulp_prediction"] == "FP") for row in rows}
        pulp_scores = {(row["version"], row["variant"]): float(row["pulp_score"]) for row in rows}
        pulp_report = {}
        equal_budget = {}
        per_version_budget = {}
        for name, keys in groups.items():
            selected = [key for key in keys if key in pulp]
            labels = [truth[key] for key in selected]
            predictions = [pulp[key] for key in selected]
            pulp_report[name] = {
                "configurations": len(selected), "ground_truth_false_passing": sum(labels),
                "predicted_positive": sum(predictions),
                **_metrics(labels, predictions, [pulp_scores[key] for key in selected]),
            }
            equal_budget[name] = _evaluate_group(
                selected, truth, scores, threshold, budget=sum(predictions),
            )
            chosen = set()
            for version in sorted({key[0] for key in selected}):
                version_keys = [key for key in selected if key[0] == version]
                budget = sum(pulp[key] for key in version_keys)
                chosen.update(sorted(version_keys, key=lambda key: (-scores[key], key[1]))[:budget])
            ordered = sorted(selected)
            per_version_budget[name] = {
                "configurations": len(ordered),
                "ground_truth_false_passing": sum(truth[key] for key in ordered),
                "predicted_positive": len(chosen),
                **_metrics(
                    [truth[key] for key in ordered],
                    [int(key in chosen) for key in ordered],
                    [scores[key] for key in ordered],
                ),
            }
        report["pulp_binary_results"] = pulp_report
        report["joint_ranking_at_same_positive_budget_as_pulp"] = equal_budget
        report["joint_ranking_at_same_per_version_budget_as_pulp"] = per_version_budget
        (output / "fair-comparison.json").write_text(
            json.dumps({
                "pulp_continuous_score_and_binary_decision": pulp_report,
                "joint_same_global_budget": equal_budget,
                "joint_same_per_version_budget": per_version_budget,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with (output / "joint-pair-results.jsonl").open("w", encoding="utf-8") as stream:
        for row in pair_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "joint-errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从已有原始执行子图重建联合反事实图并重新评价")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--base-versions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--pulp-predictions", type=Path)
    args = parser.parse_args(argv)
    report = run(args.cache, args.dataset, args.labels, args.output,
                 args.base_versions, args.threshold, args.pulp_predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

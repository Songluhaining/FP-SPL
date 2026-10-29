from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

from .configurations import load_configurations
from .factor_graph import load_factor_graph
from .feature_sources import load_feature_map
from .hierarchical_graph import build_hierarchical_counterfactual_graph
from .spectrum import load_spectrum_priors
from .tensor_inference import infer_factor_graph


def _test_graph(version: Path, config: str, identity: str) -> Path:
    class_name, _, method = identity.rpartition(".")
    return (version / "configurations" / config / "runs" /
            class_name.replace(".", "_") / method / "factor-graph.tsv")


def _labels(labels_root: Path, versions: set[str]) -> dict[tuple[str, str], int]:
    result = {}
    for version in versions:
        path = labels_root / version / "variant_labels.csv"
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("FP transformed from F", "") != "1" and row["LABEL"] in {"FP", "TP"}:
                result[(version, row["VARIANT"])] = int(row["LABEL"] == "FP")
    return result


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return 0.0
    wins = sum((left > right) + 0.5 * (left == right)
               for left in positives for right in negatives)
    return wins / (len(positives) * len(negatives))


def run(cache: Path, dataset: Path, labels_root: Path, hard_output: Path,
        output: Path, versions: tuple[str, ...]) -> dict:
    version_set = set(versions)
    truth = _labels(labels_root, version_set)
    grid = [(b, a) for b in (0.80, 0.95) for a in (0.20, 0.50, 0.80)]
    rows: list[dict] = []

    @lru_cache(maxsize=128)
    def graph(path: str):
        return load_factor_graph(Path(path))

    for failing_manifestation, passing_manifestation in grid:
        scores = {key: 0.0 for key in truth}
        calibrated_scores = {key: 0.0 for key in truth}
        usable = {key: 0 for key in truth}
        top_evidence: dict[tuple[str, str], dict] = {}
        for version in versions:
            cached_version = cache / version
            data_version = dataset / version
            configs = {item.name: item for item in load_configurations(data_version)}
            plans = json.loads((cached_version / "pair-plan.json").read_text(encoding="utf-8"))
            feature_maps = {name: load_feature_map(item.variant_dir) for name, item in configs.items()}
            spectra = {name: load_spectrum_priors(item.variant_dir) for name, item in configs.items()}
            for plan in plans:
                a, b = plan["passing_configuration"], plan["failing_configuration"]
                key = (version, a)
                if key not in truth or not plan.get("passing_test"):
                    continue
                path_a = _test_graph(cached_version, a, plan["passing_test"])
                path_b = _test_graph(cached_version, b, plan["failing_test"])
                if not path_a.is_file() or not path_b.is_file():
                    continue
                hierarchy = build_hierarchical_counterfactual_graph(
                    graph(str(path_b)), graph(str(path_a)), feature_maps[b], feature_maps[a],
                    failing_manifestation, passing_manifestation,
                )
                source_priors = dict(spectra[a])
                source_priors.update(spectra[b])
                explicit = {node: 0.5 for node in hierarchy.local_statement_nodes}
                explicit.update({
                    node: source_priors[source].error_prior if source in source_priors else 0.01
                    for node, source in hierarchy.global_source.items()
                })
                inferred = infer_factor_graph(
                    hierarchy.graph, {}, {}, statement_error_priors=explicit,
                )
                if not inferred.converged:
                    continue
                candidates = [
                    (hierarchy.graph.statements[node].suspiciousness, node)
                    for node in hierarchy.shared_defects
                    if hierarchy.global_source[node] in hierarchy.failing_causal_sources
                ]
                score, top_node = max(candidates, default=(0.0, None))
                if score >= scores[key] and top_node is not None:
                    local = hierarchy.global_to_local[top_node]
                    top_evidence[key] = {
                        "source": hierarchy.global_source[top_node],
                        "source_prior": explicit[top_node],
                        "global_defect_posterior": score,
                        "passing_local_error_posterior": (
                            hierarchy.graph.statements[local["A"]].suspiciousness
                            if "A" in local else None
                        ),
                        "failing_local_error_posterior": (
                            hierarchy.graph.statements[local["B"]].suspiciousness
                            if "B" in local else None
                        ),
                        "failing_test": plan["failing_test"],
                        "passing_test": plan["passing_test"],
                        "test_similarity": plan.get("test_similarity"),
                    }
                scores[key] = max(scores[key], score)
                similarity = float(plan.get("test_similarity") or 0.0)
                calibrated_scores[key] = max(calibrated_scores[key], score * similarity)
                usable[key] += 1

        ordered = sorted(truth)
        labels = [truth[key] for key in ordered]
        values = [scores[key] for key in ordered]
        usable_keys = [key for key in ordered if usable[key]]
        rows.append({
            "failing_manifestation": failing_manifestation,
            "passing_manifestation": passing_manifestation,
            "auc_all": _auc(labels, values),
            "auc_usable": _auc([truth[key] for key in usable_keys],
                                [scores[key] for key in usable_keys]),
            "auc_similarity_calibrated": _auc(
                labels, [calibrated_scores[key] for key in ordered],
            ),
            "positive_scores": [scores[key] for key in ordered if truth[key]],
            "highest_negative": max((scores[key] for key in ordered if not truth[key]), default=0.0),
            "usable": len(usable_keys),
            "scores": {f"{key[0]}/{key[1]}": scores[key] for key in ordered},
            "similarity_calibrated_scores": {
                f"{key[0]}/{key[1]}": calibrated_scores[key] for key in ordered
            },
            "top_evidence": {
                f"{key[0]}/{key[1]}": value for key, value in top_evidence.items()
            },
        })

    hard_scores = {}
    for version in versions:
        path = hard_output / version / "ranking.csv"
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            hard_scores[(version, row["configuration"])] = float(row["score"])
    ordered = sorted(truth)
    hard_auc = _auc([truth[key] for key in ordered], [hard_scores.get(key, 0.0) for key in ordered])
    improved = sum(row["auc_all"] > hard_auc for row in rows)
    gate_passed = improved >= 4 and min(row["auc_all"] for row in rows) >= hard_auc - 0.02
    report = {
        "stage": "真实小样本机制门控",
        "versions": list(versions),
        "configurations": len(truth),
        "false_passing": sum(truth.values()),
        "hard_sharing_auc": hard_auc,
        "parameter_grid": rows,
        "gate_rule": "六组参数中至少四组优于硬共享，且最差组下降不超过0.02",
        "gate_passed": gate_passed,
        "next_action": "进入混合缺陷子集" if gate_passed else "停止扩大实验并分析结构缺失",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分层反事实图的低成本阶段门控实验")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--hard-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--versions", nargs="+", required=True)
    args = parser.parse_args(argv)
    report = run(args.cache, args.dataset, args.labels, args.hard_output,
                 args.output, tuple(args.versions))
    print(json.dumps({key: value for key, value in report.items()
                      if key != "parameter_grid"}, ensure_ascii=False, indent=2))
    for row in report["parameter_grid"]:
        print({key: value for key, value in row.items() if key not in {
            "scores", "similarity_calibrated_scores", "top_evidence", "positive_scores",
        }})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

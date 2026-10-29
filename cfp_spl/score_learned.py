from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .configurations import load_configurations
from .evaluation import _metrics
from .factor_graph import load_factor_graph
from .feature_sources import load_feature_map
from .java_tests import discover_tests
from .learned_factor_graph import build_learned_counterfactual_graph
from .spectrum import load_spectrum_priors
from .tensor_inference import infer_factor_graph
from .test_equivalence import test_pair_vector
from .train_masking import _load_graph


def _test_graph(version: Path, config: str, identity: str) -> Path:
    class_name, _, method = identity.rpartition(".")
    return (version / "configurations" / config / "runs" /
            class_name.replace(".", "_") / method / "factor-graph.tsv")


def _labels(labels_root: Path, versions: set[str]) -> dict[tuple[str, str], int]:
    result = {}
    for version in versions:
        path = labels_root / version / "variant_labels.csv"
        if not path.is_file():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("FP transformed from F", "") != "1" and row["LABEL"] in {"FP", "TP"}:
                result[(version, row["VARIANT"])] = int(row["LABEL"] == "FP")
    return result


def _mean_probability(values: list[float]) -> float:
    """同一配置多条反事实证据的等权概率汇聚，重复证据不会抬高分数。"""
    return float(sum(values) / len(values)) if values else 0.0


def run(graph_root: Path, cache: Path, dataset: Path, equivalence_model: Path,
        masking_model: Path, output: Path, versions: list[str] | None = None,
        labels_root: Path | None = None, allow_seen_versions: bool = False) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("请先安装显卡版PyTorch再执行学习模型评分") from exc
    from .learning import CounterfactualGraphNetwork, TestEquivalenceNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_checkpoint = torch.load(equivalence_model, map_location=device, weights_only=True)
    q_model = TestEquivalenceNetwork(
        q_checkpoint.get("input_size", 10), q_checkpoint.get("hidden_size", 32),
    ).to(device)
    q_model.load_state_dict(q_checkpoint["state_dict"])
    q_model.eval()
    m_checkpoint = torch.load(masking_model, map_location=device, weights_only=True)
    m_model = CounterfactualGraphNetwork(
        hidden_size=m_checkpoint.get("hidden_size", 64),
        relations=m_checkpoint.get("relations", 5),
    ).to(device)
    m_model.load_state_dict(m_checkpoint["state_dict"])
    m_model.eval()

    selected = set(versions or ())
    if selected and not allow_seen_versions:
        missing_split = [name for name, checkpoint in (
            ("测试等价模型", q_checkpoint), ("故障屏蔽模型", m_checkpoint),
        ) if "training_versions" not in checkpoint]
        if missing_split:
            raise ValueError(
                "模型缺少版本切分信息，无法证明无泄漏：" + "、".join(missing_split)
            )
        seen = selected & (set(q_checkpoint["training_versions"]) |
                           set(m_checkpoint["training_versions"]))
        if seen:
            raise ValueError(
                "评分版本曾用于弱监督训练；请改用两个模型共同留出的版本：" +
                "、".join(sorted(seen))
            )
    metadata = [json.loads(line) for line in
                (graph_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = [row for row in metadata if row["weak_outcome_flip"] == 1 and
                (not selected or row["version"] in selected)]
    pair_rows = []
    errors = []
    scores: dict[tuple[str, str], list[float]] = {}

    @lru_cache(maxsize=256)
    def raw_graph(path: str):
        return load_factor_graph(Path(path))

    by_version: dict[str, list[dict]] = {}
    for row in metadata:
        by_version.setdefault(row["version"], []).append(row)
    for version_name, rows in sorted(by_version.items()):
        cached_version = cache / version_name
        configs = {item.name: item for item in load_configurations(dataset / version_name)}
        feature_maps = {name: load_feature_map(config.variant_dir)
                        for name, config in configs.items()}
        spectra = {name: load_spectrum_priors(config.variant_dir)
                   for name, config in configs.items()}
        tests = {
            name: {test.identity: test for test in discover_tests(config.variant_dir)}
            for name, config in configs.items()
        }
        for row in rows:
            a, b = row["passing_configuration"], row["failing_configuration"]
            try:
                test_a = tests[a][row["passing_test"]]
                test_b = tests[b][row["failing_test"]]
                vector = torch.as_tensor(test_pair_vector(test_b, test_a), device=device).reshape(1, -1)
                paired = _load_graph(graph_root, row)
                with torch.no_grad():
                    q_probability = float(torch.sigmoid(q_model(vector)).item())
                    m_probability = float(torch.sigmoid(
                        m_model(paired, device).masking_logit
                    ).item())
                path_a = _test_graph(cached_version, a, row["passing_test"])
                path_b = _test_graph(cached_version, b, row["failing_test"])
                spectrum = dict(spectra[a])
                spectrum.update(spectra[b])
                learned = build_learned_counterfactual_graph(
                    raw_graph(str(path_b)), raw_graph(str(path_a)),
                    feature_maps[b], feature_maps[a], spectrum,
                    q_probability, m_probability,
                )
                diagnostics = infer_factor_graph(
                    learned.graph, {}, {},
                    statement_error_priors=learned.statement_error_priors,
                )
                score = learned.false_passing_probability
                scores.setdefault((version_name, a), []).append(score)
                pair_rows.append({
                    **row, "test_equivalence_probability": q_probability,
                    "masking_probability": m_probability,
                    "false_passing_probability": score,
                    "inference_exact": diagnostics.exact,
                    "inference_width": diagnostics.induced_width,
                    "inference_converged": diagnostics.converged,
                })
            except Exception as exc:
                errors.append({**row, "error": str(exc)})

    configuration_rows = []
    keys = set(scores)
    if labels_root:
        truth = _labels(labels_root, set(by_version))
        keys |= set(truth)
    else:
        truth = {}
    for key in sorted(keys):
        values = scores.get(key, [])
        configuration_rows.append({
            "version": key[0], "configuration": key[1],
            "score": _mean_probability(values), "usable_pairs": len(values),
            "label": truth.get(key),
        })
    report = {
        "device": str(device), "versions": len(by_version),
        "pair_scores": len(pair_rows), "errors": len(errors),
        "configurations_with_evidence": sum(bool(row["usable_pairs"]) for row in configuration_rows),
        "configurations_without_evidence": sum(not row["usable_pairs"] for row in configuration_rows),
    }
    if truth:
        evaluated = [row for row in configuration_rows if row["label"] is not None]
        labels = [int(row["label"]) for row in evaluated]
        values = [float(row["score"]) for row in evaluated]
        report["all_configurations"] = _metrics(
            labels, [int(value >= 0.5) for value in values], values,
        )
        usable = [row for row in evaluated if row["usable_pairs"]]
        usable_labels = [int(row["label"]) for row in usable]
        usable_values = [float(row["score"]) for row in usable]
        report["configurations_with_evidence_only"] = _metrics(
            usable_labels, [int(value >= 0.5) for value in usable_values], usable_values,
        ) if len(set(usable_labels)) == 2 else None

    output.mkdir(parents=True, exist_ok=True)
    (output / "pair-scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    with (output / "configuration-scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "version", "configuration", "score", "usable_pairs", "label",
        ))
        writer.writeheader()
        writer.writerows(configuration_rows)
    (output / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用学习势函数和统一因子图计算假通过概率")
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--equivalence-model", type=Path, required=True)
    parser.add_argument("--masking-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--versions", nargs="*")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--allow-seen-versions", action="store_true",
                        help="仅用于机制排错，允许评分训练中见过的版本")
    args = parser.parse_args(argv)
    report = run(
        args.graphs, args.cache, args.dataset, args.equivalence_model,
        args.masking_model, args.output, args.versions, args.labels,
        args.allow_seen_versions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

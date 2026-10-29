from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .configurations import feature_delta as configuration_delta, load_configurations
from .factor_graph import load_factor_graph
from .feature_sources import load_feature_map
from .paired_graph import build_paired_execution_graph


def _test_graph(version: Path, config: str, identity: str) -> Path:
    class_name, _, method = identity.rpartition(".")
    return (version / "configurations" / config / "runs" /
            class_name.replace(".", "_") / method / "factor-graph.tsv")


def run(cache: Path, dataset: Path, output: Path,
        versions: list[str] | None = None) -> dict:
    selected = versions or sorted(path.name for path in cache.iterdir() if path.is_dir())
    output.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    alignments = []

    @lru_cache(maxsize=256)
    def graph(path: str):
        return load_factor_graph(Path(path))

    def save(paired, identity_text: str, record: dict) -> None:
        identity = hashlib.sha1(identity_text.encode()).hexdigest()[:16]
        relative = Path(record["version"]) / f"{identity}.npz"
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            path, node_features=paired.node_features, edges=paired.edges,
            edge_types=paired.edge_types, node_ids=np.asarray(paired.node_ids),
            aligned=np.asarray(paired.aligned_statements),
        )
        statement_nodes = sum(name.startswith(("A:s:", "B:s:")) for name in paired.node_ids)
        alignment = 2 * len(paired.aligned_statements) / max(1, statement_nodes)
        alignments.append(alignment)
        records.append({
            **record, "nodes": len(paired.node_ids), "edges": paired.edges.shape[1],
            "aligned_statements": len(paired.aligned_statements),
            "statement_alignment_ratio": alignment, "path": str(relative),
        })

    for version_name in selected:
        cached_version = cache / version_name
        if not (cached_version / "pair-plan.json").is_file():
            continue
        configs = {item.name: item for item in load_configurations(dataset / version_name)}
        feature_maps = {name: load_feature_map(config.variant_dir)
                        for name, config in configs.items()}
        plans = json.loads((cached_version / "pair-plan.json").read_text(encoding="utf-8"))
        executions: dict[str, dict[str, dict[str, Path]]] = {"passed": {}, "failed": {}}
        for plan in plans:
            if not plan.get("passing_test"):
                continue
            a, b = plan["passing_configuration"], plan["failing_configuration"]
            path_a = _test_graph(cached_version, a, plan["passing_test"])
            path_b = _test_graph(cached_version, b, plan["failing_test"])
            if not path_a.is_file() or not path_b.is_file():
                continue
            executions["passed"].setdefault(plan["passing_test"], {})[a] = path_a
            executions["failed"].setdefault(plan["failing_test"], {})[b] = path_b
            try:
                paired = build_paired_execution_graph(
                    graph(str(path_b)), graph(str(path_a)), feature_maps[b], feature_maps[a],
                    plan.get("feature_delta", []), float(plan.get("test_similarity", 0.0)),
                    float(plan.get("oracle_similarity", 0.0)),
                    plan.get("test_provenance", "similar"),
                )
                save(paired, f"flip|{version_name}|{a}|{b}|{plan['failing_test']}|{plan['passing_test']}", {
                    "version": version_name,
                    "passing_configuration": a,
                    "failing_configuration": b,
                    "failing_test": plan["failing_test"],
                    "passing_test": plan["passing_test"],
                    "test_provenance": plan.get("test_provenance"),
                    "test_similarity": plan.get("test_similarity"),
                    "oracle_similarity": plan.get("oracle_similarity"),
                    "pair_kind": "failed_passed",
                    "weak_outcome_flip": 1,
                })
            except Exception as exc:
                errors.append({"version": version_name, **plan, "error": str(exc)})

        for outcome, groups in executions.items():
            for test_identity, by_config in groups.items():
                items = sorted(by_config.items())
                for offset in range(0, len(items) - 1, 2):
                    (left_name, left_path), (right_name, right_path) = items[offset:offset + 2]
                    try:
                        delta = configuration_delta(configs[left_name], configs[right_name])
                        paired = build_paired_execution_graph(
                            graph(str(right_path)), graph(str(left_path)),
                            feature_maps[right_name], feature_maps[left_name], delta,
                            1.0, 1.0, "exact",
                        )
                        save(paired,
                             f"control|{version_name}|{outcome}|{left_name}|{right_name}|{test_identity}", {
                            "version": version_name,
                            "passing_configuration": left_name,
                            "failing_configuration": right_name,
                            "failing_test": test_identity,
                            "passing_test": test_identity,
                            "test_provenance": "exact",
                            "test_similarity": 1.0,
                            "oracle_similarity": 1.0,
                            "pair_kind": f"{outcome}_{outcome}",
                            "weak_outcome_flip": 0,
                        })
                    except Exception as exc:
                        errors.append({
                            "version": version_name, "pair_kind": f"{outcome}_{outcome}",
                            "left": left_name, "right": right_name,
                            "test": test_identity, "error": str(exc),
                        })

    with (output / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    median_alignment = float(np.median(alignments)) if alignments else 0.0
    report = {
        "versions": len(set(item["version"] for item in records)),
        "exported_pairs": len(records),
        "errors": len(errors),
        "exact_pairs": sum(item["test_provenance"] == "exact" for item in records),
        "similar_pairs": sum(item["test_provenance"] == "similar" for item in records),
        "outcome_flip_pairs": sum(item["weak_outcome_flip"] == 1 for item in records),
        "same_outcome_controls": sum(item["weak_outcome_flip"] == 0 for item in records),
        "median_statement_alignment_ratio": median_alignment,
        "gate_minimum_pairs": 10,
        "gate_minimum_median_alignment": 0.20,
        "gate_passed": len(records) >= 10 and median_alignment >= 0.20 and not errors,
    }
    (output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从已有A/B执行缓存导出配对动态差异图")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--versions", nargs="*")
    args = parser.parse_args(argv)
    report = run(args.cache, args.dataset, args.output, args.versions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .configurations import load_configurations
from .evaluation import _metrics
from .java_tests import discover_tests
from .test_equivalence import test_pair_vector
from .train_masking import _load_graph
from .version_split import fold_of


def _key(row: dict) -> tuple[str, str, str, str, str]:
    return (row["version"], row["passing_configuration"],
            row["failing_configuration"], row["failing_test"],
            row["passing_test"])


def _evaluate(keys: list[tuple[str, str]], truth: dict, scores: dict,
              threshold: float = 0.5, budget: int | None = None) -> dict:
    ordered = sorted(keys)
    labels = [truth[key] for key in ordered]
    values = [scores.get(key, 0.0) for key in ordered]
    if budget is None:
        predictions = [int(value >= threshold) for value in values]
    else:
        ranked = sorted(ordered, key=lambda key: (-scores.get(key, 0.0), key))
        selected = set(ranked[:budget])
        predictions = [int(key in selected) for key in ordered]
    return {
        "configurations": len(ordered), "false_passing": sum(labels),
        "predicted_positive": sum(predictions),
        **_metrics(labels, predictions, values),
    }


def run(graph_root: Path, joint_pairs: Path, dataset: Path, models_root: Path,
        predictions: Path, output: Path, fold_count: int = 5) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("请先安装显卡版PyTorch再执行五折学习重排") from exc
    from .learning import CounterfactualGraphNetwork, TestEquivalenceNetwork

    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = [json.loads(line) for line in
                (graph_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = { _key(row): row for row in metadata if row["weak_outcome_flip"] == 1 }
    causal = [json.loads(line) for line in joint_pairs.read_text(encoding="utf-8").splitlines()]
    if set(metadata) != {_key(row) for row in causal}:
        raise ValueError("成对图与联合因果结果不能逐条对齐")

    reference = list(csv.DictReader(predictions.open(encoding="utf-8-sig")))
    truth = {(row["version"], row["variant"]): int(row["label"] == "FP")
             for row in reference}
    pulp_prediction = {
        (row["version"], row["variant"]): int(row["pulp_prediction"] == "FP")
        for row in reference
    }
    groups = {(row["version"], row["variant"]): row["group"] for row in reference}
    models = {}

    def load_models(fold: int):
        if fold in models:
            return models[fold]
        q_data = torch.load(
            models_root / f"learned-equivalence-fold{fold}" / "test-equivalence.pt",
            map_location=device, weights_only=True,
        )
        m_data = torch.load(
            models_root / f"learned-masking-fold{fold}" / "masking-proxy.pt",
            map_location=device, weights_only=True,
        )
        q_model = TestEquivalenceNetwork(
            q_data.get("input_size", 10), q_data.get("hidden_size", 32),
        ).to(device)
        q_model.load_state_dict(q_data["state_dict"])
        q_model.eval()
        m_model = CounterfactualGraphNetwork(
            hidden_size=m_data.get("hidden_size", 64),
            relations=m_data.get("relations", 5),
        ).to(device)
        m_model.load_state_dict(m_data["state_dict"])
        m_model.eval()
        models[fold] = (q_model, m_model, q_data, m_data)
        return models[fold]

    tests = {}
    pair_rows = []
    scores = {
        name: defaultdict(float) for name in
        ("causal", "causal_equivalence", "causal_masking", "full")
    }
    for position, causal_row in enumerate(causal, 1):
        key = _key(causal_row)
        row = metadata[key]
        version = row["version"]
        fold = fold_of(version, fold_count)
        q_model, m_model, q_data, m_data = load_models(fold)
        if version in set(q_data.get("training_versions", ())) or \
                version in set(m_data.get("training_versions", ())):
            raise ValueError(f"版本泄漏到训练折：{version}")
        if version not in tests:
            tests[version] = {
                config.name: {test.identity: test for test in discover_tests(config.variant_dir)}
                for config in load_configurations(dataset / version)
            }
        test_a = tests[version][row["passing_configuration"]][row["passing_test"]]
        test_b = tests[version][row["failing_configuration"]][row["failing_test"]]
        vector = torch.as_tensor(
            test_pair_vector(test_b, test_a), device=device,
        ).reshape(1, -1)
        graph = _load_graph(graph_root, row)
        with torch.no_grad():
            q = float(torch.sigmoid(q_model(vector)).item())
            m = float(torch.sigmoid(m_model(graph, device).masking_logit).item())
        causal_score = float(causal_row["joint_score"])
        values = {
            "causal": causal_score,
            "causal_equivalence": causal_score * q,
            "causal_masking": causal_score * m,
            "full": causal_score * q * m,
        }
        config_key = (version, row["passing_configuration"])
        for name, value in values.items():
            scores[name][config_key] = max(scores[name][config_key], value)
        pair_rows.append({
            **causal_row, "fold": fold, "test_equivalence_probability": q,
            "masking_probability": m, **{f"score_{name}": value
                                          for name, value in values.items()},
        })
        if position % 250 == 0:
            print(f"[{position}/{len(causal)}] pairs")

    all_keys = sorted(truth)
    key_groups = {
        "all": all_keys,
        "base_feature": [key for key in all_keys if groups[key] == "base_feature"],
        "optional_feature": [key for key in all_keys if groups[key] != "base_feature"],
        "with_evidence": [key for key in all_keys if key in scores["full"]],
    }
    pulp_budget = {name: sum(pulp_prediction[key] for key in keys)
                   for name, keys in key_groups.items()}
    fixed = {variant: {name: _evaluate(keys, truth, variant_scores)
                       for name, keys in key_groups.items()}
             for variant, variant_scores in scores.items()}
    equal_budget = {variant: {
        name: _evaluate(keys, truth, variant_scores, budget=pulp_budget[name])
        for name, keys in key_groups.items()
    } for variant, variant_scores in scores.items()}
    per_version = {}
    for variant, variant_scores in scores.items():
        chosen = set()
        for version in sorted({key[0] for key in all_keys}):
            version_keys = [key for key in all_keys if key[0] == version]
            budget = sum(pulp_prediction[key] for key in version_keys)
            chosen.update(sorted(
                version_keys, key=lambda key: (-variant_scores.get(key, 0.0), key)
            )[:budget])
        labels = [truth[key] for key in all_keys]
        values = [variant_scores.get(key, 0.0) for key in all_keys]
        per_version[variant] = {
            "configurations": len(all_keys), "false_passing": sum(labels),
            "predicted_positive": len(chosen),
            **_metrics(labels, [int(key in chosen) for key in all_keys], values),
        }

    report = {
        "scope": {"versions": len({key[0] for key in all_keys}),
                  "configurations": len(all_keys), "false_passing": sum(truth.values()),
                  "true_passing": len(all_keys) - sum(truth.values()),
                  "pairs": len(pair_rows),
                  "configurations_with_evidence": len(scores["full"])},
        "device": str(device), "fold_count": fold_count,
        "fixed_threshold_0_5": fixed,
        "same_global_budget_as_pulp": equal_budget,
        "same_per_version_budget_as_pulp": per_version,
        "pulp_positive_budget": pulp_budget,
    }
    (output / "pair-scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    with (output / "configuration-scores.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as stream:
        fields = ("version", "configuration", "label", "group", "pulp_prediction",
                  "causal", "causal_equivalence", "causal_masking", "full")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in all_keys:
            writer.writerow({
                "version": key[0], "configuration": key[1], "label": truth[key],
                "group": groups[key], "pulp_prediction": pulp_prediction[key],
                **{name: values.get(key, 0.0) for name, values in scores.items()},
            })
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="五折弱监督模型对联合因果证据进行无泄漏重排")
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--joint-pairs", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args(argv)
    report = run(args.graphs, args.joint_pairs, args.dataset, args.models_root,
                 args.predictions, args.output, args.fold_count)
    print(json.dumps(report["scope"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

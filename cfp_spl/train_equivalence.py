from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .configurations import load_configurations
from .java_tests import discover_tests
from .test_equivalence import EquivalenceExample, make_hard_examples, make_weak_examples
from .version_split import split_versions


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if not positive_count or not negative_count:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return ((rank_sum - positive_count * (positive_count + 1) / 2.0)
            / (positive_count * negative_count))


def _examples(dataset: Path, max_versions: int | None) -> tuple[
    list[EquivalenceExample], list[EquivalenceExample]
]:
    versions = [path for path in sorted(dataset.iterdir())
                if (path / "config.report.csv.done").is_file()]
    if max_versions:
        versions = versions[:max_versions]
    reliable: list[EquivalenceExample] = []
    hard: list[EquivalenceExample] = []
    for version in versions:
        tests = []
        for config in load_configurations(version):
            tests.extend(discover_tests(config.variant_dir))
        reliable.extend(make_weak_examples(version.name, tests))
        hard.extend(make_hard_examples(version.name, tests))
    return reliable, hard


def run(dataset: Path, output: Path, epochs: int = 100, seed: int = 7,
        max_versions: int | None = None, validation_fold: int = 0,
        fold_count: int = 5) -> dict:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("请先安装显卡版PyTorch再训练测试等价模型") from exc
    from .learning import TestEquivalenceNetwork

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    values, hard_values = _examples(dataset, max_versions)
    versions = sorted({item.version for item in values})
    if len(versions) < 5:
        raise ValueError("至少需要五个缺陷版本才能按版本划分训练集和验证集")
    training_versions, validation_versions = split_versions(
        versions, validation_fold, fold_count,
    )
    train = [item for item in values if item.version not in validation_versions]
    validation = [item for item in hard_values if item.version in validation_versions]
    if not train or not validation:
        raise ValueError("训练集或验证集为空")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestEquivalenceNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    features_train = torch.as_tensor(np.stack([x.features for x in train]), device=device)
    labels_train = torch.as_tensor([x.label for x in train], dtype=torch.float32, device=device)
    positives = max(1, sum(x.label for x in train))
    negatives = max(1, len(train) - positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    best = None
    patience = 15
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_train)
        loss = loss_fn(logits, labels_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            features = torch.as_tensor(np.stack([x.features for x in validation]), device=device)
            labels = np.asarray([x.label for x in validation], dtype=np.int64)
            scores = torch.sigmoid(model(features)).cpu().numpy()
            auc = _auc(labels, scores)
            brier = float(np.mean((scores - labels) ** 2))
        history.append({"epoch": epoch, "loss": float(loss.item()),
                        "validation_auc": auc, "validation_brier": brier})
        key = (auc, -brier)
        if best is None or key > best[0]:
            best = (key, epoch, {name: value.detach().cpu().clone()
                                 for name, value in model.state_dict().items()})
            patience = 15
        else:
            patience -= 1
            if patience == 0:
                break
    assert best is not None
    model.load_state_dict(best[2])
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_size": 10,
                "hidden_size": 32,
                "training_versions": sorted(training_versions),
                "validation_versions": sorted(validation_versions),
                "validation_fold": validation_fold,
                "fold_count": fold_count}, output / "test-equivalence.pt")
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_versions": len(set(item.version for item in train)),
        "validation_versions": len(validation_versions),
        "training_version_names": sorted(training_versions),
        "validation_version_names": sorted(validation_versions),
        "validation_fold": validation_fold,
        "fold_count": fold_count,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "positive_train_examples": sum(item.label for item in train),
        "positive_validation_examples": sum(item.label for item in validation),
        "validation_kind": "非完全相同的近等价测试与词法相近困难负例",
        "best_epoch": best[1],
        "validation_auc": best[0][0],
        "validation_brier": -best[0][1],
        "gate_threshold_auc": 0.85,
        "gate_passed": best[0][0] >= 0.85,
        "history": history,
    }
    (output / "test-equivalence-training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练测试行为等价性的弱监督模型")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-versions", type=int)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.output, args.epochs, args.seed, args.max_versions,
                 args.validation_fold, args.fold_count)
    print(json.dumps({key: value for key, value in report.items() if key != "history"},
                     ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

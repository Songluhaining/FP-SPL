from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .paired_graph import PairedExecutionGraph, RELATIONS
from .train_equivalence import _auc
from .version_split import split_versions


def _load_graph(root: Path, row: dict) -> PairedExecutionGraph:
    data = np.load(root / row["path"])
    aligned_values = data["aligned"]
    aligned = tuple(tuple(str(value) for value in item) for item in aligned_values)
    graph = PairedExecutionGraph(
        tuple(str(value) for value in data["node_ids"]),
        data["node_features"], data["edges"], data["edge_types"], RELATIONS,
        aligned, float(row["test_similarity"]), float(row["oracle_similarity"]),
    )
    graph.validate()
    return graph


def run(graph_root: Path, output: Path, epochs: int = 60, seed: int = 11,
        validation_fold: int = 0, fold_count: int = 5) -> dict:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("请先安装显卡版PyTorch再训练屏蔽机制模型") from exc
    from .learning import CounterfactualGraphNetwork

    rows = [json.loads(line) for line in
            (graph_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    # 结果翻转只保留同一测试标识，避免模型把测试内容变化误学为屏蔽机制。
    selected = [row for row in rows if row["weak_outcome_flip"] == 0 or
                row["passing_test"] == row["failing_test"]]
    versions = sorted({row["version"] for row in selected})
    if len(versions) < 5:
        raise ValueError("屏蔽机制门控至少需要五个缺陷版本")
    training_versions, validation_versions = split_versions(
        versions, validation_fold, fold_count,
    )
    train_rows = [row for row in selected if row["version"] not in validation_versions]
    validation_rows = [row for row in selected if row["version"] in validation_versions]
    if {row["weak_outcome_flip"] for row in validation_rows} != {0, 1}:
        raise ValueError("留出版本必须同时含结果翻转执行对和同结果控制对")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train = [(row, _load_graph(graph_root, row)) for row in train_rows]
    validation = [(row, _load_graph(graph_root, row)) for row in validation_rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CounterfactualGraphNetwork(relations=len(RELATIONS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    positives = max(1, sum(row["weak_outcome_flip"] for row, _graph in train))
    negatives = max(1, len(train) - positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    best = None
    patience = 12
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for position, (row, graph) in enumerate(train, 1):
            label = torch.tensor(float(row["weak_outcome_flip"]), device=device)
            logit = model(graph, device).masking_logit
            loss = loss_fn(logit.reshape(1), label.reshape(1)) / 8.0
            loss.backward()
            total_loss += float(loss.item()) * 8.0
            if position % 8 == 0 or position == len(train):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            labels = np.asarray([row["weak_outcome_flip"] for row, _graph in validation])
            scores = np.asarray([
                torch.sigmoid(model(graph, device).masking_logit).item()
                for _row, graph in validation
            ])
        auc = _auc(labels, scores)
        brier = float(np.mean((scores - labels) ** 2))
        history.append({"epoch": epoch, "loss": total_loss / len(train),
                        "validation_auc": auc, "validation_brier": brier})
        key = (auc, -brier)
        if best is None or key > best[0]:
            best = (key, epoch, {name: value.detach().cpu().clone()
                                 for name, value in model.state_dict().items()})
            patience = 12
        else:
            patience -= 1
            if patience == 0:
                break

    assert best is not None
    model.load_state_dict(best[2])
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": 64,
                "relations": len(RELATIONS),
                "training_versions": sorted(training_versions),
                "validation_versions": sorted(validation_versions),
                "validation_fold": validation_fold,
                "fold_count": fold_count}, output / "masking-proxy.pt")
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_target": "同一测试下失败/通过结果翻转，相对于失败/失败和通过/通过控制对",
        "train_versions": len(set(row["version"] for row in train_rows)),
        "validation_versions": len(validation_versions),
        "training_version_names": sorted(training_versions),
        "validation_version_names": sorted(validation_versions),
        "validation_fold": validation_fold,
        "fold_count": fold_count,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_outcome_flip": sum(row["weak_outcome_flip"] for row in train_rows),
        "validation_outcome_flip": sum(row["weak_outcome_flip"] for row in validation_rows),
        "best_epoch": best[1],
        "validation_auc": best[0][0],
        "validation_brier": -best[0][1],
        "gate_threshold_auc": 0.70,
        "gate_passed": best[0][0] >= 0.70,
        "history": history,
    }
    (output / "masking-training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练特征条件化故障屏蔽的弱监督代理任务")
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args(argv)
    report = run(args.graphs, args.output, args.epochs, args.seed,
                 args.validation_fold, args.fold_count)
    print(json.dumps({key: value for key, value in report.items() if key != "history"},
                     ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

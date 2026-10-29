from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - 由命令入口给出可读错误
    raise ImportError("弱监督模型需要安装可选依赖：pip install -e .[learning]") from exc

from .paired_graph import PairedExecutionGraph


class TestEquivalenceNetwork(nn.Module):
    """根据测试内容、调用和判定条件估计行为等价概率。"""

    __test__ = False

    def __init__(self, input_size: int = 10, hidden_size: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class RelationalGraphEncoder(nn.Module):
    """不依赖额外图学习库的关系消息传递编码器。"""

    def __init__(self, input_size: int, hidden_size: int, relations: int, layers: int = 2):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(layers)
        )
        self.relation_layers = nn.ModuleList(
            nn.ModuleList(nn.Linear(hidden_size, hidden_size, bias=False)
                          for _ in range(relations))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(layers))

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor,
                edge_types: torch.Tensor) -> torch.Tensor:
        states = torch.relu(self.input(nodes))
        if edges.numel() == 0:
            return states
        sources, targets = edges[0], edges[1]
        for own, relations, norm in zip(
            self.self_layers, self.relation_layers, self.norms,
        ):
            aggregated = torch.zeros_like(states)
            degrees = torch.zeros(states.size(0), 1, device=states.device, dtype=states.dtype)
            for relation, transform in enumerate(relations):
                mask = edge_types == relation
                if not torch.any(mask):
                    continue
                target = targets[mask]
                messages = transform(states[sources[mask]])
                aggregated.index_add_(0, target, messages)
                degrees.index_add_(0, target, torch.ones(
                    target.size(0), 1, device=states.device, dtype=states.dtype,
                ))
            update = own(states) + aggregated / degrees.clamp_min(1.0)
            states = torch.relu(norm(update))
        return states


@dataclass(frozen=True)
class CounterfactualOutputs:
    test_equivalence_logit: torch.Tensor
    masking_logit: torch.Tensor
    defect_logit: torch.Tensor
    pair_false_passing_logit: torch.Tensor


class CounterfactualGraphNetwork(nn.Module):
    """从配对动态差异图同时产生等价、缺陷和屏蔽因子势。"""

    def __init__(self, node_features: int = 18, hidden_size: int = 64,
                 relations: int = 5, layers: int = 2):
        super().__init__()
        self.encoder = RelationalGraphEncoder(
            node_features, hidden_size, relations, layers,
        )
        self.equivalence = nn.Linear(hidden_size, 1)
        self.masking = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
        )
        self.defect = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
        )
        self.false_passing = nn.Sequential(
            nn.Linear(hidden_size * 3 + 3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
        )

    def forward(self, graph: PairedExecutionGraph, device: torch.device | str) -> CounterfactualOutputs:
        nodes = torch.as_tensor(graph.node_features, device=device)
        edges = torch.as_tensor(graph.edges, device=device)
        edge_types = torch.as_tensor(graph.edge_types, device=device)
        # 测试相似度由独立等价模型处理，屏蔽机制编码器不读取测试上下文边，
        # 防止用“测试是否相似”代替真正的动态路径差异。
        test_relation = graph.relation_names.index("test_context")
        structural = edge_types != test_relation
        states = self.encoder(nodes, edges[:, structural], edge_types[structural])
        positions = {name: position for position, name in enumerate(graph.node_ids)}
        test_state = states[positions["test:pair"]]
        aligned = graph.aligned_statements
        if aligned:
            state_a = torch.stack([states[positions[left]] for left, _right, _source in aligned])
            state_b = torch.stack([states[positions[right]] for _left, right, _source in aligned])
            difference = torch.abs(state_a - state_b)
            pooled_a, pooled_b, pooled_difference = (
                state_a.mean(0), state_b.mean(0), difference.mean(0),
            )
        else:
            zero = torch.zeros_like(test_state)
            pooled_a = pooled_b = pooled_difference = zero
        contrast = torch.cat((pooled_a, pooled_b, pooled_difference))
        equivalence_logit = self.equivalence(test_state).squeeze(-1)
        masking_logit = self.masking(contrast).squeeze(-1)
        defect_logit = self.defect(contrast).squeeze(-1)
        probabilities = torch.stack((
            torch.sigmoid(equivalence_logit),
            torch.sigmoid(masking_logit),
            torch.sigmoid(defect_logit),
        ))
        pair_logit = self.false_passing(torch.cat((contrast, probabilities))).squeeze(-1)
        return CounterfactualOutputs(
            equivalence_logit, masking_logit, defect_logit, pair_logit,
        )

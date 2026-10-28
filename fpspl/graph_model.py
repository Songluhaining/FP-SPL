"""Relation-aware graph neural network for positive/unlabeled test graphs.

The graph remains a directed, typed graph throughout message passing. Only
failed-test and unlabeled passing-test outcomes are used for training.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass

from fpspl.model import TestGraph


def _bucket(value: str, size: int = 2048) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big") % size


def _numeric_summary(values: list[str]) -> tuple[float, float, float]:
    numbers = []
    for value in values:
        kind, _, raw = value.replace("|", ":", 1).partition(":")
        if kind not in {"I", "S", "J", "F", "D", "Z"}:
            continue
        try:
            number = float(raw)
            if math.isfinite(number):
                numbers.append(number)
        except ValueError:
            pass
    if not numbers:
        return 0.0, 0.0, 0.0
    return (sum(number > 0 for number in numbers) / len(numbers),
            sum(number < 0 for number in numbers) / len(numbers),
            min(math.log1p(max(abs(number) for number in numbers)) / 12.0, 1.0))


def graph_arrays(graph: TestGraph) -> tuple[list, list, list, list, list, list]:
    """Return node attributes and directed typed edges without graph flattening."""
    nodes = sorted(graph.nodes) or ["<empty>"]
    positions = {node: index for index, node in enumerate(nodes)}
    location_ids, feature_ids, code_ids, attributes = [], [], [], []
    for node in nodes:
        info = graph.node_info.get(node)
        feature = info.feature if info else "UNKNOWN"
        code = info.code if info else ""
        branch_true = node + ":T" in graph.branches
        branch_false = node + ":F" in graph.branches
        positive, negative, magnitude = _numeric_summary(graph.runtime_values.get(node, []))
        location_ids.append(_bucket(node))
        feature_ids.append(_bucket(feature))
        code_ids.append(_bucket(code.lower()))
        attributes.append([float(feature == "TEST"), float(feature == "UNKNOWN"),
                           float(bool(re.search(r"\b(?:assert\w+|fail)\s*\(", code))),
                           float(branch_true), float(branch_false), positive, negative, magnitude])
    sources, targets, relations = [], [], []
    for source, target, kind in sorted(graph.typed_links):
        if source not in positions or target not in positions or kind not in {"DATA", "CONTROL"}:
            continue
        relation = 0 if kind == "DATA" else 1
        sources.extend([positions[source], positions[target]])
        targets.extend([positions[target], positions[source]])
        relations.extend([relation, relation + 2])
    return location_ids, feature_ids, code_ids, attributes, [sources, targets], relations


def batch_graphs(graphs: list[TestGraph]):
    import torch

    locations, features, codes, attributes = [], [], [], []
    sources, targets, relations, memberships = [], [], [], []
    offset = 0
    for graph_index, graph in enumerate(graphs):
        loc, feat, code, attr, edges, rel = graph_arrays(graph)
        locations.extend(loc)
        features.extend(feat)
        codes.extend(code)
        attributes.extend(attr)
        memberships.extend([graph_index] * len(loc))
        sources.extend(source + offset for source in edges[0])
        targets.extend(target + offset for target in edges[1])
        relations.extend(rel)
        offset += len(loc)
    return (torch.tensor(locations, dtype=torch.long),
            torch.tensor(features, dtype=torch.long),
            torch.tensor(codes, dtype=torch.long),
            torch.tensor(attributes, dtype=torch.float32),
            torch.tensor([sources, targets], dtype=torch.long).reshape(2, -1),
            torch.tensor(relations, dtype=torch.long),
            torch.tensor(memberships, dtype=torch.long), len(graphs))


def _network_class():
    import torch
    from torch import nn

    class DependencyLayer(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.transforms = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(4))
            self.gate = nn.Sequential(nn.Linear(width * 2 + 8, width), nn.ReLU(), nn.Linear(width, 1))
            self.relations = nn.Embedding(4, 8)
            self.norm = nn.LayerNorm(width)
            self.dropout = nn.Dropout(0.15)

        def forward(self, states, edges, relation_ids):
            if relation_ids.numel() == 0:
                return states
            source, target = edges
            messages = torch.zeros((source.numel(), states.size(1)), device=states.device)
            for relation, transform in enumerate(self.transforms):
                selected = relation_ids == relation
                if selected.any():
                    messages[selected] = transform(states[source[selected]])
            gates = torch.sigmoid(self.gate(torch.cat(
                (states[source], states[target], self.relations(relation_ids)), dim=1)))
            aggregated = torch.zeros_like(states)
            aggregated.index_add_(0, target, messages * gates)
            degree = torch.zeros((states.size(0), 1), device=states.device)
            degree.index_add_(0, target, gates)
            return self.norm(states + self.dropout(torch.relu(aggregated / degree.clamp_min(1.0))))

    class DependencyNetwork(nn.Module):
        def __init__(self, width: int = 48, depth: int = 3):
            super().__init__()
            self.location = nn.Embedding(2048, width)
            self.feature = nn.Embedding(2048, width)
            self.code = nn.Embedding(2048, width)
            self.attributes = nn.Linear(8, width)
            self.layers = nn.ModuleList(DependencyLayer(width) for _ in range(depth))
            self.attention = nn.Linear(width + 1, 1)
            self.readout = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(),
                                         nn.Dropout(0.15), nn.Linear(width, 1))

        def forward(self, batch):
            location, feature, code, attributes, edges, relations, membership, count = batch
            states = (self.location(location) + self.feature(feature)
                      + self.code(code) + self.attributes(attributes)) / 4.0
            for layer in self.layers:
                states = layer(states, edges, relations)
            means = torch.zeros((count, states.size(1)), device=states.device)
            means.index_add_(0, membership, states)
            counts = torch.bincount(membership, minlength=count).unsqueeze(1).clamp_min(1)
            means = means / counts
            logits = self.attention(torch.cat((states, attributes[:, 2:3]), dim=1)).squeeze(1)
            pooled = []
            for index in range(count):
                selected = membership == index
                weights = torch.softmax(logits[selected], dim=0)
                pooled.append((states[selected] * weights.unsqueeze(1)).sum(dim=0))
            attention_pool = torch.stack(pooled)
            return self.readout(torch.cat((means, attention_pool), dim=1)).squeeze(1)

    return DependencyNetwork


@dataclass
class TrainedGraphPUModel:
    networks: list
    sampled_variants: list[set[str]]
    positives: int
    unlabeled: int

    def score_all(self, graphs: list[TestGraph]) -> list[float]:
        import torch

        if not graphs:
            return []
        batch = batch_graphs(graphs)
        predictions = []
        with torch.no_grad():
            for network in self.networks:
                network.eval()
                predictions.append(torch.sigmoid(network(batch)).tolist())
        return [sum(predictions[model_index][index] for model_index in eligible) / len(eligible)
                for index, graph in enumerate(graphs)
                for eligible in [[model_index for model_index, sampled in enumerate(self.sampled_variants)
                                  if graph.variant not in sampled] or list(range(len(self.networks)))]]

    def save(self, path) -> None:
        import torch

        torch.save({"network_state_dicts": [network.state_dict() for network in self.networks],
                    "sampled_variants": [sorted(variants) for variants in self.sampled_variants],
                    "positives": self.positives, "unlabeled": self.unlabeled,
                    "architecture": "directed-relational-attention-3layer-v1"}, path)


def train_graph_pu_model(positives: list[TestGraph], unlabeled: list[TestGraph],
                         seed: int = 17, ensembles: int = 4, epochs: int = 40) -> TrainedGraphPUModel | None:
    """Train several graph discriminators; sampled unlabeled graphs are noisy background."""
    if len(positives) < 2 or len(unlabeled) < 2:
        return None
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    random_generator = random.Random(seed)
    variants = sorted({graph.variant for graph in unlabeled})
    networks, sampled_variants = [], []
    Network = _network_class()
    for repetition in range(ensembles):
        selected_variants = set(random_generator.sample(
            variants, min(len(variants), max(1, math.ceil(len(variants) * 0.35)))))
        background = [graph for graph in unlabeled if graph.variant in selected_variants]
        random_generator.shuffle(background)
        background = background[:max(2, len(positives) * 2)]
        selected_variants = {graph.variant for graph in background}
        batch = batch_graphs(positives + background)
        torch.manual_seed(seed + repetition)
        network = Network()
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.01)
        for _ in range(epochs):
            network.train()
            optimizer.zero_grad()
            logits = network(batch)
            positive_logits = logits[:len(positives)]
            background_logits = logits[len(positives):]
            # The background may include false passes, so it has a smaller role.
            loss = (F.softplus(-positive_logits).mean()
                    + 0.35 * F.softplus(background_logits).mean()
                    + 0.15 * F.softplus(background_logits[:, None] - positive_logits[None, :]).mean())
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 2.0)
            optimizer.step()
        network.eval()
        networks.append(network)
        sampled_variants.append(selected_variants)
    return TrainedGraphPUModel(networks, sampled_variants, len(positives), len(unlabeled))

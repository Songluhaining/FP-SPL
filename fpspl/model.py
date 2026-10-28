from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    feature: str
    source_file: str
    source_line: int
    feature_line: int
    kind: str
    code: str = ""
    signature: str = ""


@dataclass
class TestGraph:
    version: str
    variant: str
    test: str
    outcome: str
    nodes: set[str] = field(default_factory=set)
    links: set[tuple[str, str]] = field(default_factory=set)
    typed_links: set[tuple[str, str, str]] = field(default_factory=set)
    branches: set[str] = field(default_factory=set)
    event_count: int = 0
    runtime_values: dict[str, list[str]] = field(default_factory=dict)
    observed_result: dict[str, Any] = field(default_factory=dict)
    observations: tuple[str, ...] = ()
    node_info: dict[str, NodeInfo] = field(default_factory=dict)
    evidence: str = "coverage+slice-reachability"

    @property
    def features(self) -> set[str]:
        return {self.node_info[n].feature for n in self.nodes if n in self.node_info}

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["nodes"] = sorted(self.nodes)
        result["links"] = [list(link) for link in sorted(self.links)]
        result["typed_links"] = [list(link) for link in sorted(self.typed_links)]
        result["branches"] = sorted(self.branches)
        result["features"] = sorted(self.features)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestGraph":
        """Restore a cached runtime graph for model experiments without re-instrumenting."""
        return cls(
            version=data["version"], variant=data["variant"], test=data["test"],
            outcome=data["outcome"], nodes=set(data["nodes"]),
            links={tuple(edge) for edge in data.get("links", [])},
            typed_links={tuple(edge) for edge in data.get("typed_links", [])},
            branches=set(data.get("branches", [])), event_count=data.get("event_count", 0),
            runtime_values=data.get("runtime_values", {}),
            observed_result=data.get("observed_result", {}),
            observations=tuple(data.get("observations", [])),
            node_info={name: NodeInfo(**details) for name, details in data.get("node_info", {}).items()},
            evidence=data.get("evidence", ""),
        )


@dataclass
class FailureTemplate:
    version: str
    source_variant: str
    source_test: str
    nodes: set[str]
    links: set[tuple[str, str]]
    branches: set[str]
    features: set[str]
    observations: tuple[str, ...]
    node_info: dict[str, NodeInfo]
    evidence: str

    @classmethod
    def from_graph(cls, graph: TestGraph) -> "FailureTemplate":
        if graph.outcome != "failed":
            raise ValueError("A failure template requires a failed test")
        return cls(
            version=graph.version,
            source_variant=graph.variant,
            source_test=graph.test,
            nodes=set(graph.nodes),
            links=set(graph.links),
            branches=set(graph.branches),
            features=set(graph.features),
            observations=graph.observations,
            node_info=dict(graph.node_info),
            evidence=graph.evidence,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["nodes"] = sorted(self.nodes)
        result["links"] = [list(link) for link in sorted(self.links)]
        result["branches"] = sorted(self.branches)
        result["features"] = sorted(self.features)
        return result

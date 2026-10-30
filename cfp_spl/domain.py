from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Outcome = Literal["passed", "failed"]


@dataclass(frozen=True)
class Configuration:
    name: str
    features: tuple[int, ...]
    feature_names: tuple[str, ...]
    outcome: Outcome
    variant_dir: Path

    @property
    def selected_features(self) -> tuple[str, ...]:
        return tuple(n for n, v in zip(self.feature_names, self.features) if v)


@dataclass(frozen=True)
class JavaTestCase:
    class_name: str
    method_name: str
    source_path: Path
    body: str
    full_source: str

    @property
    def identity(self) -> str:
        return f"{self.class_name}.{self.method_name}"


@dataclass(frozen=True)
class TestPair:
    failing: JavaTestCase
    passing: JavaTestCase | None
    provenance: Literal["exact", "similar", "model_selected", "generated", "missing"]
    similarity: float
    oracle_similarity: float
    generation_request: Path | None = None
    rationale: str = ""


@dataclass(frozen=True)
class StatementPosterior:
    node_id: str
    raw_name: str
    stable_key: str
    correctness: float
    observed: bool
    reduced: bool

    @property
    def suspiciousness(self) -> float:
        return 1.0 - self.correctness


@dataclass
class FactorGraph:
    statements: dict[str, StatementPosterior] = field(default_factory=dict)
    values: dict[str, dict] = field(default_factory=dict)
    factors: dict[str, dict] = field(default_factory=dict)
    incidences: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class StatementEvidence:
    stable_key: str
    feature_source: str
    failing_suspiciousness: float
    passing_suspiciousness: float
    posterior_gap: float
    present_in_a: bool
    executed_in_a: bool
    causal_in_a: bool


@dataclass
class PairAssessment:
    passing_configuration: str
    failing_configuration: str
    failing_test: str
    passing_test: str | None
    failing_outcome: Outcome
    passing_outcome: Outcome | None
    test_provenance: str
    test_similarity: float
    evidence_level: Literal[
        "confirmed_false_passing", "strong_suspect", "weak_suspect", "insufficient"
    ]
    score: float
    statements: list[StatementEvidence]
    reasons: list[str]

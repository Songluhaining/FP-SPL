"""Read the all-statements backward-slice format produced in the PULP trial.

Each entry contains a transitive backward slice. It is reachability evidence,
not an observed sequence of direct dynamic-dependence edges.
"""

from __future__ import annotations

import json
from pathlib import Path


SLICE_NAME = "slice.all_statements.backward_slicing.log"


def read_backward_slices(variant_dir: Path) -> dict[str, set[str]]:
    path = variant_dir / SLICE_NAME
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a slice mapping in {path}")
    return {
        str(target): {str(item) for item in ancestors}
        for target, ancestors in raw.items()
        if isinstance(ancestors, list)
    }


def observed_reachability_links(
    executed_nodes: set[str], backward_slices: dict[str, set[str]]
) -> set[tuple[str, str]]:
    """Retain slice reachability only when both endpoints ran in this test."""
    return {
        (source, target)
        for target in executed_nodes
        for source in backward_slices.get(target, set()) & executed_nodes
        if source != target
    }

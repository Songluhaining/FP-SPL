"""
Persistent cache for the LLM-based counterfactual test selection.

Motivation (TSE revision / experiment cost):
The matcher in ``pyutils.utils.llm_match_passed_for_failed_in_class`` queries a
large language model (Qwen) to choose, for every *failing* test in a product,
a few counterfactual *passing* tests. That call is the only expensive part of
the pipeline and we re-run the experiments many times. The experiment driver
(``pylib.spl.fl``) also *deletes* each mutant directory right after it finishes
(see ``shutil.rmtree(mutated_project_Path)``), so anything stored inside the
variant/mutant tree would be wiped between runs.

This module stores the LLM decisions in a durable, project-root location that
is independent of the (deleted) variant tree:

    <project_root>/llm_selection_cache/<system>/<mutant>/<variant>.json

Each JSON file holds, for one product (``variant``), the full selection:

    {
      "<estest_class>": {
        "<failed_method>": ["<passed_method>", ...],
        ...
      },
      ...
    }

i.e. exactly the ``{failed_method: [passed_method, ...]}`` mapping that the LLM
matcher returns, grouped per test class. On a later run we reload this file and
skip the LLM entirely for any test class already present.

The cache key (system / mutant / variant) is derived entirely from the variant
root path, which is reproducible across runs:

    <buggy_systems_folder>/<mutant>/variants/<variant>

so ``variant`` is the leaf, ``mutant`` is its grandparent, and ``system`` is the
``buggy_systems_folder`` name (the dataset folder, e.g. ``4wise-BankAccountTP-1BUG-Full``).
No environment variables or extra plumbing are needed.

Set ``BRFL_LLM_CACHE=0`` to disable reading/writing the cache (always re-query
the LLM) -- useful when you deliberately want to refresh the stored decisions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Mapping schema, per variant: {estest_class: {failed_method: [passed_method, ...]}}
VariantSelection = Dict[str, Dict[str, List[str]]]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "llm_selection_cache"

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def cache_enabled() -> bool:
    """Cache is on unless ``BRFL_LLM_CACHE`` is set to a falsy value."""
    return os.environ.get("BRFL_LLM_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def _sanitize(name: str) -> str:
    """Make an arbitrary system/mutant/variant name safe as a path component."""
    name = (name or "").strip()
    if not name:
        return "_"
    return _SANITIZE_RE.sub("_", name)


def derive_cache_identity(variant_root, system: Optional[str] = None) -> Tuple[str, str, str]:
    """Derive ``(system, mutant, variant)`` for the cache key from a variant root.

    Expected layout: ``<buggy_systems_folder>/<mutant>/variants/<variant>`` so:
        variant = leaf, mutant = grandparent, system = buggy_systems_folder name.
    Pass ``system`` explicitly to override the path-derived value.
    """
    vr = Path(variant_root).resolve()
    parts = vr.parts
    variant = vr.name
    # vr.parent is the ``variants`` dir; its parent is the mutant dir.
    mutant = parts[-3] if len(parts) >= 3 else "default"
    if system is None:
        # The dataset folder that holds all mutants of one system.
        system = parts[-4] if len(parts) >= 4 else "default"
    return system, mutant, variant


def cache_path(system: str, mutant: str, variant: str) -> Path:
    return CACHE_ROOT / _sanitize(system) / _sanitize(mutant) / (_sanitize(variant) + ".json")


def load_variant_cache(system: str, mutant: str, variant: str) -> VariantSelection:
    """Load the stored selection for one product; ``{}`` if nothing cached yet."""
    p = cache_path(system, mutant, variant)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        print(f"[llm_cache][WARN] malformed cache (not a dict), ignoring: {p}")
    except Exception as e:  # corrupt/partial file -> behave as a cache miss
        print(f"[llm_cache][WARN] failed to read cache {p}: {e}")
    return {}


def save_variant_cache(system: str, mutant: str, variant: str, data: VariantSelection) -> None:
    """Persist the selection for one product (atomic replace)."""
    p = cache_path(system, mutant, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, p)  # atomic on the same filesystem (Windows & POSIX)
    print(f"[llm_cache] saved {sum(len(v) for v in data.values())} failed-test "
          f"selections across {len(data)} test class(es) -> {p}")

"""
Strategy-aware drop-in replacement for ``pylib.spl.collect_items_passed``.

Original ``collect_items_passed`` is hard-wired to the LLM (Qwen) matcher.
For the TSE revision (reviewer comment: "There is no comparison with simpler
alternatives, such as Jaccard-only filtering, random selection, or
embedding-based similarity.") we need to be able to swap the matcher without
touching the rest of the pipeline.

This module provides ``collect_items_passed_by_strategy(...)`` which mirrors
the original function but takes a ``strategy`` argument:

    - "llm"       -> the original Qwen-based matcher (unchanged behaviour)
    - "jaccard"   -> Jaccard similarity over test-method body tokens
    - "random"    -> uniformly random selection (deterministic per seed)
    - "embedding" -> dense embedding (semantic) cosine similarity
    - "all"       -> skip selection entirely and use every passing test
                     (= the no-counterfactual-filtering baseline)

The strategy can also be picked from the environment variable
``BRFL_TEST_SELECTION``; pass ``strategy=None`` to opt-in to that behaviour.
This keeps the call site in ``pylib.spl.getsplcmdline`` to a single line.

Output: a list of "passed items" with the same dict schema as
``pylib.spl.collect_items``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyutils.utils import (
    group_items_by_class,
    load_test_file_source,
    llm_match_passed_for_failed_in_class,
)
from pyutils.test_selection_baselines import STRATEGY_MATCHERS
from pylib.llm_selection_cache import (
    cache_enabled,
    derive_cache_identity,
    load_variant_cache,
    save_variant_cache,
)


VALID_STRATEGIES = {"llm", "jaccard", "random", "embedding", "all"}


def _resolve_strategy(strategy: Optional[str]) -> str:
    if strategy is None:
        strategy = os.environ.get("BRFL_TEST_SELECTION", "llm").lower()
    strategy = strategy.lower().strip()
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown test-selection strategy: {strategy!r}. "
            f"Valid options: {sorted(VALID_STRATEGIES)}"
        )
    return strategy


def collect_items_passed_by_strategy(
    passed_dir: Path,
    failed_items: List[Dict[str, Any]],
    variant_root: Path,
    strategy: Optional[str] = None,
    top_k: int = 5,
    # LLM-specific knobs (only used when strategy == "llm").
    # Left as None -> taken from llm_config.json / the BRFL_LLM_* env vars
    # (see pyutils/llm_config.py); no key is ever hard-coded here.
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    # Random-specific knob
    random_seed: int = 0,
    # Embedding-specific knob (only used when strategy == "embedding").
    # Defaults to None -> the matcher's own default (all-MiniLM-L6-v2), or the
    # BRFL_EMBEDDING_MODEL env var if set.
    embedding_model: Optional[str] = None,
    # Persistent-cache identity (only used when strategy == "llm").
    # If left as None they are derived from the ``variant_root`` path
    # (system = dataset folder, mutant = grandparent, variant = leaf).
    system: Optional[str] = None,
    mutant: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Select counterfactual passing tests per failing test, using ``strategy``.

    Behaviour matches the original ``pylib.spl.collect_items_passed`` for
    ``strategy="llm"``; for "jaccard" / "random" we substitute the matcher
    from ``pyutils.test_selection_baselines``; for "all" we skip matching and
    return every passing item.
    """
    strategy = _resolve_strategy(strategy)

    # Lazy import to avoid the spl <-> spl_test_selection circular import.
    from pylib.spl import collect_items
    passed_items_all = collect_items(passed_dir)

    if strategy == "all":
        print(f"[collect_items_passed_by_strategy] strategy=all -> "
              f"{len(passed_items_all)} passed items (no filtering)")
        return passed_items_all

    failed_by_class, passed_by_class = group_items_by_class(
        failed_items, passed_items_all
    )

    # ---- persistent LLM cache (so re-runs don't re-pay the model) ----------
    use_cache = strategy == "llm" and cache_enabled()
    variant_cache: Dict[str, Dict[str, List[str]]] = {}
    cache_dirty = False
    if use_cache:
        c_system, c_mutant, c_variant = derive_cache_identity(
            variant_root, system=system
        )
        if mutant:
            c_mutant = mutant
        if variant:
            c_variant = variant
        variant_cache = load_variant_cache(c_system, c_mutant, c_variant)
        if variant_cache:
            print(f"[llm_cache] loaded {len(variant_cache)} cached test class(es) "
                  f"for {c_system}/{c_mutant}/{c_variant}")

    selected_passed: List[Dict[str, Any]] = []
    seen_passed_keys: set = set()

    for estest_class, f_list in failed_by_class.items():
        p_list = passed_by_class.get(estest_class, [])
        if not p_list:
            continue

        print(f"[{strategy}] {estest_class}: {len(f_list)} failing, {len(p_list)} passing")

        # All matchers accept (estest_class, test_source, f_list, p_list, top_k, ...).
        # LLM needs the source for prompting; Jaccard needs it for tokens;
        # random ignores it.
        # Cache hit: reuse the stored LLM decision, skip the model entirely.
        if use_cache and estest_class in variant_cache:
            mapping = variant_cache[estest_class]
            print(f"[llm][cache-hit] {estest_class}: reusing stored selection "
                  f"({sum(len(v) for v in mapping.values())} matches)")
            _apply_mapping(mapping, f_list, p_list, selected_passed, seen_passed_keys,
                           estest_class)
            continue

        test_src = load_test_file_source(variant_root, estest_class)

        try:
            if strategy == "llm":
                mapping = llm_match_passed_for_failed_in_class(
                    estest_class=estest_class,
                    test_source=test_src,
                    failed_items_in_class=f_list,
                    passed_items_in_class=p_list,
                    model_name=model_name,
                    temperature=temperature,
                    top_k=top_k,
                    api_key=api_key,
                )
                if use_cache:
                    variant_cache[estest_class] = mapping
                    cache_dirty = True
            else:
                matcher = STRATEGY_MATCHERS[strategy]
                # Extra knobs are accepted via **_ignored by every matcher, so
                # we can pass the union; each matcher uses only what it needs:
                #   - "random"    uses ``seed``
                #   - "embedding" uses ``embedding_model`` (omitted when unset
                #                 so the matcher's own default applies)
                matcher_kwargs: Dict[str, Any] = {"seed": random_seed}
                eff_embedding_model = embedding_model or os.environ.get("BRFL_EMBEDDING_MODEL")
                if eff_embedding_model:
                    matcher_kwargs["embedding_model"] = eff_embedding_model
                mapping = matcher(
                    estest_class=estest_class,
                    test_source=test_src,
                    failed_items_in_class=f_list,
                    passed_items_in_class=p_list,
                    top_k=top_k,
                    **matcher_kwargs,
                )
        except Exception as e:
            print(f"[{strategy}][WARN] match failed for {estest_class}: {e}")
            continue

        _apply_mapping(mapping, f_list, p_list, selected_passed, seen_passed_keys,
                       estest_class)

    if use_cache and cache_dirty:
        save_variant_cache(c_system, c_mutant, c_variant, variant_cache)

    print(f"[collect_items_passed_by_strategy] strategy={strategy} "
          f"selected {len(selected_passed)} passed tests")
    return selected_passed


def _apply_mapping(
    mapping: Dict[str, List[str]],
    f_list: List[Dict[str, Any]],
    p_list: List[Dict[str, Any]],
    selected_passed: List[Dict[str, Any]],
    seen_passed_keys: set,
    estest_class: str,
) -> None:
    """Resolve a ``{failed_method: [passed_method, ...]}`` mapping into items.

    Appends the selected (deduplicated) passing items to ``selected_passed``.
    Passing method names that no longer exist in ``p_list`` are skipped, so a
    cached decision stays valid even if the regenerated product differs.
    """
    passed_index = {it["test_method"]: it for it in p_list}
    for f_item in f_list:
        fname = f_item["test_method"]
        for mname in mapping.get(fname, []):
            key = (estest_class, mname)
            if key in seen_passed_keys:
                continue
            seen_passed_keys.add(key)
            pi = passed_index.get(mname)
            if pi is not None:
                selected_passed.append(pi)

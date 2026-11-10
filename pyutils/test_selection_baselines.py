"""
Baseline counterfactual-passing-test selectors used for the TSE revision
(comparison against the LLM-based selector in ``utils.llm_match_passed_for_failed_in_class``).

Each public function shares the same calling convention as the LLM version so the
upper layer (``pylib.spl.collect_items_passed_*``) can swap them in transparently:

    matcher(
        estest_class: str,
        test_source: str,
        failed_items_in_class: List[Dict[str, Any]],
        passed_items_in_class: List[Dict[str, Any]],
        top_k: int = 5,
        **strategy_specific_kwargs,
    ) -> Dict[str, List[str]]

The returned mapping is ``{failed_method_name: [passed_method_name, ...]}``,
identical to what the LLM matcher returns, so it plugs into the same
``collect_items_passed`` post-processing logic.

Three strategies are implemented:

* ``jaccard_match_passed_for_failed_in_class`` -- pure Jaccard similarity over
  test-method body token sets. No external service, no DL model.
* ``random_match_passed_for_failed_in_class`` -- uniformly random selection
  (deterministic w.r.t. ``estest_class`` + failed method name so reruns
  reproduce identical splits).
* ``embedding_match_passed_for_failed_in_class`` -- dense embedding similarity:
  every test-method body is encoded with a pretrained sentence/code embedding
  model (Sentence-BERT ``all-MiniLM-L6-v2`` by default) and passing methods are
  ranked by cosine similarity to the failing method. Unlike Jaccard (lexical
  token overlap) this captures *semantic* similarity, which is the
  "embedding-based similarity" baseline requested by the reviewer. It needs the
  ``sentence-transformers`` package; the model is loaded lazily and cached
  in-process, and encoding is deterministic so reruns reproduce identical
  splits.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Set


# ---------------------------------------------------------------------------
# Shared helpers: parsing an EvoSuite *_ESTest.java file into per-method bodies
# ---------------------------------------------------------------------------

# A very small set of Java keywords / EvoSuite scaffolding tokens that carry
# essentially no signal for similarity and would otherwise dominate the union.
_JAVA_STOPWORDS: Set[str] = {
    "public", "private", "protected", "static", "final", "void", "throws",
    "throw", "new", "return", "if", "else", "for", "while", "do", "switch",
    "case", "break", "continue", "true", "false", "null", "this", "super",
    "class", "interface", "extends", "implements", "import", "package",
    "try", "catch", "finally", "instanceof", "int", "long", "short", "byte",
    "char", "float", "double", "boolean", "string", "object",
    # JUnit / EvoSuite scaffolding noise
    "test", "timeout", "throwable", "assert", "assertequals", "asserttrue",
    "assertfalse", "assertnull", "assertnotnull", "assertsame", "assertarrayequals",
    "org", "junit", "evosuite", "runwith", "evosuiterunner",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_METHOD_HEADER_RE = re.compile(
    r"public\s+void\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def _strip_comments(src: str) -> str:
    """Remove // line comments and /* ... */ block comments."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", " ", src)
    return src


def _extract_method_bodies(test_source: str) -> Dict[str, str]:
    """Return ``{test_method_name: body_text}`` for every ``public void`` method.

    Implementation note: we rely on simple brace matching after locating the
    method header. EvoSuite-generated test classes are well-formed enough for
    this to work without a full Java parser.
    """
    src = _strip_comments(test_source)
    bodies: Dict[str, str] = {}

    for m in _METHOD_HEADER_RE.finditer(src):
        name = m.group("name")
        # Find the opening '{' after the parameter list / `throws` clause.
        i = src.find("{", m.end())
        if i < 0:
            continue
        depth = 0
        j = i
        while j < len(src):
            c = src[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    bodies[name] = src[i + 1 : j]
                    break
            j += 1
    return bodies


def _tokenize(body: str) -> Set[str]:
    """Lower-cased identifier-like tokens minus Java/EvoSuite stopwords."""
    toks = {t.lower() for t in _TOKEN_RE.findall(body)}
    return {t for t in toks if t not in _JAVA_STOPWORDS and not t.isdigit()}


# ---------------------------------------------------------------------------
# Strategy 1: Jaccard similarity over test-method body tokens
# ---------------------------------------------------------------------------

def jaccard_match_passed_for_failed_in_class(
    estest_class: str,
    test_source: str,
    failed_items_in_class: List[Dict[str, Any]],
    passed_items_in_class: List[Dict[str, Any]],
    top_k: int = 5,
    min_similarity: float = 0.0,
    **_ignored,
) -> Dict[str, List[str]]:
    """Pick the ``top_k`` passing methods with highest Jaccard token overlap.

    A passing method is only kept if its similarity is strictly greater than
    ``min_similarity`` (default 0.0, so any non-empty overlap qualifies).
    If a failing method has no usable overlap, the entry is still recorded
    with an empty candidate list (matching the LLM matcher's contract for
    "no suitable candidate").
    """
    bodies = _extract_method_bodies(test_source)

    passed_methods = [it["test_method"] for it in passed_items_in_class]
    passed_tokens: Dict[str, Set[str]] = {
        name: _tokenize(bodies.get(name, "")) for name in passed_methods
    }

    mapping: Dict[str, List[str]] = {}
    for f_item in failed_items_in_class:
        fname = f_item["test_method"]
        f_toks = _tokenize(bodies.get(fname, ""))

        scored: List[tuple] = []
        for pname in passed_methods:
            p_toks = passed_tokens[pname]
            if not f_toks and not p_toks:
                sim = 0.0
            else:
                union = f_toks | p_toks
                inter = f_toks & p_toks
                sim = (len(inter) / len(union)) if union else 0.0
            if sim > min_similarity:
                scored.append((sim, pname))

        # Sort by similarity desc, tiebreak on method name for determinism.
        scored.sort(key=lambda x: (-x[0], x[1]))
        mapping[fname] = [name for _, name in scored[:top_k]]

    return mapping


# ---------------------------------------------------------------------------
# Strategy 2: Random selection
# ---------------------------------------------------------------------------

def _seeded_rng(estest_class: str, failed_method: str, salt: int) -> random.Random:
    """Stable RNG so a given (class, failed_method, salt) reproduces the same draw."""
    key = f"{estest_class}::{failed_method}::{salt}".encode("utf-8")
    seed = int(hashlib.md5(key).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def random_match_passed_for_failed_in_class(
    estest_class: str,
    test_source: str,  # accepted for signature parity; not used
    failed_items_in_class: List[Dict[str, Any]],
    passed_items_in_class: List[Dict[str, Any]],
    top_k: int = 5,
    seed: int = 0,
    **_ignored,
) -> Dict[str, List[str]]:
    """Uniformly random selection of ``top_k`` passing methods per failing method.

    Deterministic across reruns thanks to the per-(class, method) seeding.
    Pass a different ``seed`` to obtain an independent random replicate
    (useful for reporting mean / std over multiple random runs).
    """
    passed_methods = [it["test_method"] for it in passed_items_in_class]
    if not passed_methods:
        return {f["test_method"]: [] for f in failed_items_in_class}

    mapping: Dict[str, List[str]] = {}
    for f_item in failed_items_in_class:
        fname = f_item["test_method"]
        rng = _seeded_rng(estest_class, fname, seed)
        k = min(top_k, len(passed_methods))
        mapping[fname] = rng.sample(passed_methods, k)
    return mapping


# ---------------------------------------------------------------------------
# Strategy 3: Dense embedding (semantic) similarity
# ---------------------------------------------------------------------------

# Default Sentence-BERT model: small, fast, no fine-tuning required, and the
# de-facto baseline for general-purpose semantic-text similarity. Override via
# the ``embedding_model`` kwarg (e.g. "microsoft/codebert-base" for a
# code-pretrained encoder) or the ``BRFL_EMBEDDING_MODEL`` env var upstream.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Loaded models are expensive to construct, so cache them per process. The
# matcher is invoked once per test class across many variants, so without this
# we would reload the encoder thousands of times.
_EMBEDDING_MODEL_CACHE: Dict[str, Any] = {}


def _load_embedding_model(model_name: str):
    """Lazily load (and cache) a SentenceTransformer encoder by name."""
    cached = _EMBEDDING_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "embedding-based test selection requires the 'sentence-transformers' "
            "package. Install it with: pip install sentence-transformers"
        ) from e
    model = SentenceTransformer(model_name)
    _EMBEDDING_MODEL_CACHE[model_name] = model
    return model


def embedding_match_passed_for_failed_in_class(
    estest_class: str,
    test_source: str,
    failed_items_in_class: List[Dict[str, Any]],
    passed_items_in_class: List[Dict[str, Any]],
    top_k: int = 5,
    min_similarity: float = 0.0,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    **_ignored,
) -> Dict[str, List[str]]:
    """Pick the ``top_k`` passing methods most semantically similar to each failing one.

    Each test-method body is encoded with a pretrained embedding model and
    passing methods are ranked by cosine similarity to the failing method.
    Mirrors :func:`jaccard_match_passed_for_failed_in_class` but replaces the
    lexical token-overlap score with dense-embedding cosine similarity, so it
    captures *semantic* rather than surface-level resemblance.

    A passing method is kept only if its similarity is strictly greater than
    ``min_similarity``. Empty bodies fall back to the method name so the encoder
    never receives an empty string.
    """
    import numpy as np

    passed_methods = [it["test_method"] for it in passed_items_in_class]
    failed_methods = [it["test_method"] for it in failed_items_in_class]
    if not passed_methods:
        return {fname: [] for fname in failed_methods}

    bodies = _extract_method_bodies(test_source)

    def _text(name: str) -> str:
        body = bodies.get(name, "").strip()
        # Prepend the method name for a little extra signal; fall back to the
        # name alone when the body could not be extracted.
        return f"{name} {body}".strip() if body else name

    model = _load_embedding_model(embedding_model)

    passed_emb = model.encode(
        [_text(n) for n in passed_methods],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    failed_emb = model.encode(
        [_text(n) for n in failed_methods],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Embeddings are L2-normalized, so cosine similarity is just the dot product.
    sim_matrix = failed_emb @ passed_emb.T

    mapping: Dict[str, List[str]] = {}
    for i, fname in enumerate(failed_methods):
        scored = [
            (float(sim_matrix[i, j]), pname)
            for j, pname in enumerate(passed_methods)
            if float(sim_matrix[i, j]) > min_similarity
        ]
        # Sort by similarity desc, tiebreak on method name for determinism.
        scored.sort(key=lambda x: (-x[0], x[1]))
        mapping[fname] = [name for _, name in scored[:top_k]]

    return mapping


# ---------------------------------------------------------------------------
# Dispatcher convenience
# ---------------------------------------------------------------------------

# Maps strategy name -> matcher callable. Kept as a module-level constant so
# callers can introspect available strategies (e.g. in CLI flags).
STRATEGY_MATCHERS = {
    "jaccard":   jaccard_match_passed_for_failed_in_class,
    "random":    random_match_passed_for_failed_in_class,
    "embedding": embedding_match_passed_for_failed_in_class,
}

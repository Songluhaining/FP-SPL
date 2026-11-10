"""Central configuration for the LLM used by the test-selection step.

The LLM matcher (``pyutils.utils.llm_match_passed_for_failed_in_class``) used to
carry a hard-coded API key. Everything is now read from one place so that users
of this repository can plug in their own model / endpoint / key without editing
any source file.

Resolution order (first non-empty wins):

1. explicit function arguments (``model=``, ``api_key=`` ...);
2. environment variables:
   ``BRFL_LLM_PROVIDER`` / ``BRFL_LLM_MODEL`` / ``BRFL_LLM_BASE_URL`` /
   ``BRFL_LLM_API_KEY`` / ``BRFL_LLM_TEMPERATURE`` / ``BRFL_LLM_TOP_K`` /
   ``BRFL_LLM_TIMEOUT``;
   for the key, the provider-specific classics are also accepted:
   ``DASHSCOPE_API_KEY`` (dashscope), ``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY``
   (openai-compatible endpoints);
3. the JSON config file -- ``$BRFL_LLM_CONFIG`` if set, otherwise
   ``<project_root>/llm_config.json``;
4. the built-in defaults below.

Config file format (see ``llm_config.example.json``)::

    {
      "provider": "dashscope",
      "model": "qwen-plus",
      "base_url": "",
      "api_key": "sk-...",
      "temperature": 0.1,
      "top_k": 5,
      "timeout": 60
    }

``provider`` is either:

* ``"dashscope"``  -- Aliyun DashScope native SDK (``dashscope.Generation``);
* ``"openai"``     -- any OpenAI-compatible ``/chat/completions`` endpoint
  (OpenAI, DeepSeek, DashScope compatible-mode, vLLM, Ollama, ...), in which
  case ``base_url`` must point at the API root, e.g.
  ``https://api.deepseek.com/v1``.

NEVER commit a real key: ``llm_config.json`` is git-ignored, only the
``.example`` template is tracked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Project root = parent directory of this ``pyutils`` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_FILENAME = "llm_config.json"
EXAMPLE_CONFIG_FILENAME = "llm_config.example.json"

# Built-in fallbacks (no secret here on purpose).
DEFAULTS: Dict[str, Any] = {
    "provider": "dashscope",
    "model": "qwen-plus",
    "base_url": "",
    "api_key": "",
    "temperature": 0.1,
    "top_k": 5,
    "timeout": 60,
}

# env var -> config key
_ENV_KEYS = {
    "BRFL_LLM_PROVIDER": "provider",
    "BRFL_LLM_MODEL": "model",
    "BRFL_LLM_BASE_URL": "base_url",
    "BRFL_LLM_API_KEY": "api_key",
    "BRFL_LLM_TEMPERATURE": "temperature",
    "BRFL_LLM_TOP_K": "top_k",
    "BRFL_LLM_TIMEOUT": "timeout",
}

# Key-only fallbacks, kept for people who already export the vendor variables.
_PROVIDER_KEY_ENVS = {
    "dashscope": ("DASHSCOPE_API_KEY",),
    "openai": ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"),
}

_CACHE: Optional[Dict[str, Any]] = None


def config_path() -> Path:
    """Path of the JSON config file we would read (it may not exist)."""
    env_path = os.environ.get("BRFL_LLM_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return PROJECT_ROOT / DEFAULT_CONFIG_FILENAME


def _load_config_file() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"[llm_config] cannot read {path}: {e}") from e
    if not isinstance(raw, dict):
        raise RuntimeError(f"[llm_config] {path} must contain a JSON object")
    # Tolerate both {"provider": ...} and {"llm": {"provider": ...}}.
    if "llm" in raw and isinstance(raw["llm"], dict):
        raw = raw["llm"]
    return {k: v for k, v in raw.items() if k in DEFAULTS and v not in (None, "")}


def _coerce(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise types (env values always arrive as strings)."""
    cfg["provider"] = str(cfg["provider"]).strip().lower()
    cfg["model"] = str(cfg["model"]).strip()
    cfg["base_url"] = str(cfg["base_url"]).strip()
    cfg["api_key"] = str(cfg["api_key"]).strip()
    try:
        cfg["temperature"] = float(cfg["temperature"])
    except (TypeError, ValueError):
        cfg["temperature"] = DEFAULTS["temperature"]
    try:
        cfg["top_k"] = int(cfg["top_k"])
    except (TypeError, ValueError):
        cfg["top_k"] = DEFAULTS["top_k"]
    try:
        cfg["timeout"] = float(cfg["timeout"])
    except (TypeError, ValueError):
        cfg["timeout"] = DEFAULTS["timeout"]
    if cfg["provider"] not in ("dashscope", "openai"):
        raise RuntimeError(
            f"[llm_config] unknown provider {cfg['provider']!r}; "
            f"expected 'dashscope' or 'openai'"
        )
    return cfg


def get_llm_config(refresh: bool = False) -> Dict[str, Any]:
    """Return the effective configuration (defaults < file < environment)."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return dict(_CACHE)

    cfg: Dict[str, Any] = dict(DEFAULTS)
    cfg.update(_load_config_file())

    for env_name, key in _ENV_KEYS.items():
        val = os.environ.get(env_name, "").strip()
        if val:
            cfg[key] = val

    cfg = _coerce(cfg)

    # Provider-specific key fallbacks (only if still unset).
    if not cfg["api_key"]:
        for env_name in _PROVIDER_KEY_ENVS.get(cfg["provider"], ()):
            val = os.environ.get(env_name, "").strip()
            if val:
                cfg["api_key"] = val
                break

    _CACHE = dict(cfg)
    return dict(cfg)


def resolve_api_key(explicit: Optional[str] = None) -> str:
    """Return the API key to use, raising a helpful error when missing."""
    if explicit and explicit.strip():
        return explicit.strip()

    cfg = get_llm_config()
    if cfg["api_key"]:
        return cfg["api_key"]

    raise RuntimeError(
        "No LLM API key configured.\n"
        f"  * copy {EXAMPLE_CONFIG_FILENAME} to {config_path()} and fill in "
        f'"api_key", or\n'
        "  * export BRFL_LLM_API_KEY (or DASHSCOPE_API_KEY / OPENAI_API_KEY).\n"
        "The key is never stored in the source tree; llm_config.json is git-ignored."
    )


def mask_key(key: str) -> str:
    """Mask a key for logging: ``sk-abcd...wxyz``."""
    if not key:
        return "<unset>"
    if len(key) <= 10:
        return key[:2] + "***"
    return f"{key[:5]}...{key[-4:]}"


def describe_config() -> str:
    """One-line, secret-free summary -- safe to print in logs."""
    cfg = get_llm_config()
    where = config_path()
    src = str(where) if where.is_file() else "built-in defaults / environment"
    return (
        f"[llm_config] provider={cfg['provider']} model={cfg['model']} "
        f"base_url={cfg['base_url'] or '-'} api_key={mask_key(cfg['api_key'])} "
        f"(from {src})"
    )


if __name__ == "__main__":  # quick self-check: python -m pyutils.llm_config
    print(describe_config())

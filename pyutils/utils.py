import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import textwrap
from collections import defaultdict
from http import HTTPStatus

from pyutils.llm_config import get_llm_config, resolve_api_key

def load_test_file_source(variant_root: Path, estest_class: str) -> str:
    """
    根据 estest_class 找到对应测试 Java 文件源码。
    形如: ElevatorSystem.Elevator_ESTest -> test/ElevatorSystem/Elevator_ESTest.java
    """
    rel = estest_class.replace(".", "/") + ".java"
    candidate = variant_root / "test" / rel
    if not candidate.exists():
        # 这里打个 warning；模型那边可以看到注释，也知道缺失
        return f"// TEST SOURCE NOT FOUND for {estest_class}\n"

    return candidate.read_text(encoding="utf-8", errors="ignore")

def group_items_by_class(
    failed_items: List[Dict[str, Any]],
    passed_items: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    按 estest_class（测试类）分组 failed / passed 测试项。
    """
    failed_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    passed_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for it in failed_items:
        est = it.get("estest_class")
        if est:
            failed_by_class[est].append(it)

    for it in passed_items:
        est = it.get("estest_class")
        if est:
            passed_by_class[est].append(it)

    return failed_by_class, passed_by_class


def llm_match_passed_for_failed_in_class(
    estest_class: str,
    test_source: str,
    failed_items_in_class: List[Dict[str, Any]],
    passed_items_in_class: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    调用大模型，为单个测试类 estest_class 中的每个 failing 测试方法，
    选出若干最相近的 passing 测试方法（反事实候选）。

    模型名 / 温度 / top_k / API Key 若留空，则统一取自 pyutils.llm_config
    （配置文件 llm_config.json 或 BRFL_LLM_* 环境变量）。

    返回:
        mapping: {failed_method_name: [passed_method_name1, ...]}
    """
    cfg = get_llm_config()
    model_name = model_name or cfg["model"]
    temperature = cfg["temperature"] if temperature is None else temperature
    top_k = cfg["top_k"] if top_k is None else top_k
    # 1) 提取方法名列表，方便做合法性检查
    failed_methods = [it["test_method"] for it in failed_items_in_class]
    passed_methods = [it["test_method"] for it in passed_items_in_class]

    meta = {
        "test_class": estest_class,
        "failed_methods": failed_methods,
        "passed_methods": passed_methods,
    }

    # 2) 构造 prompt
    system_prompt = (
        "You are an assistant for software testing.\n"
        "You will receive:\n"
        "  (1) The source code of a Java test class.\n"
        "  (2) A list of failing test method names.\n"
        "  (3) A list of passing test method names.\n\n"
        "For each failing test method, you must select a small number of most similar\n"
        "passing test methods that can serve as counterfactual test cases\n"
        "(similar intention or scenario but passing).\n\n"
        "IMPORTANT CONSTRAINTS:\n"
        "- Only use the provided method names; do not invent new method names.\n"
        "- If no suitable passing candidate exists for a failing method, you may return an empty list.\n"
        "- The output MUST be a single JSON object with the structure:\n"
        "{\n"
        "  \"matches\": [\n"
        "    {\"failed\": \"<failed_method>\", \"candidates\": [\"<passed1>\", \"<passed2>\"]},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "- Do NOT output any explanation or extra text outside this JSON.\n"
    )

    user_prompt = (
        "Here is the Java test class source code:\n"
        "<<<TEST_FILE_BEGIN>>>\n"
        f"{test_source}\n"
        "<<<TEST_FILE_END>>>\n\n"
        "Here is the metadata (in JSON):\n"
        f"{json.dumps(meta, indent=2)}\n\n"
        f"For each failing method, select up to {top_k} most similar passing methods.\n"
        "Return ONLY the JSON object described above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # 3) 调用大模型（Key 由配置文件 / 环境变量提供，不写死在代码里）
    raw = call_llm(
        messages,
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    ).strip()

    # 4) 处理可能出现的 ```json ... ``` 或额外说明，尽量只保留 JSON 部分
    def _extract_json(text: str) -> str:
        # 去掉 ```json / ``` 包裹
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 从第一个 '{' 到最后一个 '}' 截取
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return text[start : end + 1]
        return text  # 尝试原文

    json_str = _extract_json(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 方便调试：你可以在这里打印 raw 或写到日志
        raise RuntimeError(
            f"[LLM] JSON 解析失败: {e}\nRaw response was:\n{raw}"
        ) from e

    # 5) 构造 {failed_method -> [passed_method,...]} 映射
    mapping: Dict[str, List[str]] = {}
    for entry in data.get("matches", []):
        f = entry.get("failed")
        cands = entry.get("candidates") or []
        if not isinstance(f, str):
            continue
        # 只保留真实存在的 passing 方法，并限制 top_k
        valid_cands = [m for m in cands if m in passed_methods]
        if valid_cands:
            mapping[f] = valid_cands[:top_k]

    return mapping

def call_llm(
    messages,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """
    与具体厂商无关的对话补全调用。

    messages: [{"role":"system/user/assistant", "content":"..."}]

    所有留空的参数都从 pyutils.llm_config 读取（配置文件 llm_config.json、
    或 BRFL_LLM_* / DASHSCOPE_API_KEY / OPENAI_API_KEY 等环境变量）。

    支持两种接入方式（由 provider 决定）：
      * "dashscope"：阿里云百炼原生 SDK（dashscope.Generation）；
      * "openai"   ：任何兼容 OpenAI /chat/completions 协议的服务
                     （OpenAI、DeepSeek、百炼 compatible-mode、vLLM、Ollama…），
                     此时必须提供 base_url。
    """
    cfg = get_llm_config()
    provider = (provider or cfg["provider"]).lower()
    model = model or cfg["model"]
    temperature = cfg["temperature"] if temperature is None else temperature
    timeout = cfg["timeout"] if timeout is None else timeout
    base_url = base_url if base_url is not None else cfg["base_url"]
    key = resolve_api_key(api_key)

    if provider == "openai":
        return _call_openai_compatible(
            messages,
            model=model,
            temperature=temperature,
            api_key=key,
            base_url=base_url,
            timeout=timeout,
        )
    if provider == "dashscope":
        return _call_dashscope(
            messages,
            model=model,
            temperature=temperature,
            api_key=key,
        )
    raise RuntimeError(f"不支持的 provider: {provider!r}（可选 'dashscope' / 'openai'）")


def _call_dashscope(messages, model: str, temperature: float, api_key: str) -> str:
    """阿里云百炼原生 SDK 调用；仅在真正用到时才导入 dashscope。"""
    try:
        from dashscope import Generation
    except ImportError as e:
        raise RuntimeError(
            "provider='dashscope' 需要安装 dashscope 包（pip install dashscope），"
            "或者在配置里改用 provider='openai' + base_url。"
        ) from e

    # 使用 Generation，结果格式指定为 message，便于统一解析
    resp = Generation.call(
        model=model,
        messages=messages,
        temperature=temperature,
        result_format="message",
        api_key=api_key,
    )

    status = getattr(resp, "status_code", HTTPStatus.OK)
    if status != HTTPStatus.OK:
        code = getattr(resp, "code", None)
        msg = getattr(resp, "message", None)
        raise RuntimeError(f"DashScope Generation 调用失败: code={code}, message={msg}")

    try:
        return resp.output["choices"][0]["message"]["content"]
    except Exception:
        text = getattr(resp, "output", {}).get("text")
        if text:
            return text
        return json.dumps(getattr(resp, "output", {}), ensure_ascii=False)


def _call_openai_compatible(
    messages,
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
    timeout: float,
) -> str:
    """任何兼容 OpenAI 协议的 /chat/completions 端点。"""
    if not base_url:
        raise RuntimeError(
            "provider='openai' 时必须配置 base_url，例如 "
            "'https://api.deepseek.com/v1' 或 "
            "'https://dashscope.aliyuncs.com/compatible-mode/v1'。"
        )

    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"LLM 调用失败: HTTP {resp.status_code}, body={resp.text[:500]}"
        )
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"无法解析 LLM 返回内容: {e}\nraw={json.dumps(data)[:500]}") from e


def call_qwen(messages, model=None, temperature=None, api_key: Optional[str] = None) -> str:
    """向后兼容的旧接口名，等价于 call_llm（配置决定实际用哪个模型）。"""
    return call_llm(messages, model=model, temperature=temperature, api_key=api_key)
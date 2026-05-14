from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from experiments.evaluation.error_types import ErrorType
from experiments.evaluation.prompt import (
    ERROR_CLASSIFIER_SYSTEM_PROMPT,
    build_error_classifier_user_prompt,
)


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 120.0
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.8
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_env(default_enabled: bool = False) -> "LLMConfig":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "https://api.openai.com/v1"
        model = os.getenv("OPENAI_MODEL") or os.getenv("MODEL") or "gpt-4o-mini"
        enabled = default_enabled and bool(api_key)
        return LLMConfig(enabled=enabled, model=model, api_key=api_key, base_url=base_url)


def _normalize_base_url(base_url: str) -> str:
    url = (base_url or "").strip()
    if not url:
        return "https://api.openai.com/v1"
    return url.rstrip("/")


def _chat_completions_url(base_url: str) -> str:
    url = _normalize_base_url(base_url)
    # base_url is typically something like ".../v1"
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _try_openai_sdk_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Dict[str, Any],
) -> Optional[str]:
    """
    Prefer the OpenAI SDK (if installed). Return None if it's unavailable.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key, base_url=_normalize_base_url(base_url))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body or {},
        stream=False,
    )
    return resp.choices[0].message.content


def _http_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    extra_body: Dict[str, Any],
) -> str:
    url = _chat_completions_url(base_url)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        method="POST",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    return obj["choices"][0]["message"]["content"]


def call_llm_chat(config: LLMConfig, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: (content, error_message)
    """
    if not config.api_key:
        return None, "missing_api_key"

    try:
        content = _try_openai_sdk_chat(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            extra_body=config.extra_body,
        )
        if content is not None:
            return content, None
    except Exception as e:
        return None, f"openai_sdk_error: {e}"

    try:
        content = _http_chat_completion(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            extra_body=config.extra_body,
        )
        return content, None
    except urllib.error.HTTPError as e:
        return None, f"http_error: {e.code} {e.reason}"
    except Exception as e:
        return None, f"http_exception: {e}"


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()

    # Fast path: the entire response is already valid JSON. If parsing fails
    # we deliberately fall through to the brace-scanning fallback below;
    # JSONDecodeError here is expected and does not indicate a bug.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first {...} block
    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = s[first : last + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def classify_error_type_with_llm(
    *,
    config: LLMConfig,
    language: Optional[str],
    expected_diff: str,
    actual_diff: str,
    feature_summary: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Call the LLM to judge the 5 non-OutputInvalid error types; requires strict JSON output.
    Returns: (parsed_json, error_message)
    """
    user_prompt = build_error_classifier_user_prompt(
        language=language,
        expected_diff=expected_diff,
        actual_diff=actual_diff,
        feature_summary_json=json.dumps(feature_summary, ensure_ascii=False),
    )
    messages = [
        {"role": "system", "content": ERROR_CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content, err = call_llm_chat(config, messages)
    if err or not content:
        return None, err or "empty_llm_response"

    parsed = _extract_first_json_object(content)
    if not parsed:
        return None, "llm_json_parse_failed"

    et = parsed.get("error_type")
    if et not in {e.value for e in ErrorType}:
        return None, f"llm_invalid_error_type: {et}"

    # Normalize confidence
    conf = parsed.get("confidence", None)
    try:
        conf_f = float(conf)
        conf_f = max(0.0, min(1.0, conf_f))
    except Exception:
        conf_f = 0.5
    parsed["confidence"] = conf_f

    # Normalize reason_brief
    rb = parsed.get("reason_brief")
    if not isinstance(rb, list):
        parsed["reason_brief"] = []
    else:
        parsed["reason_brief"] = [str(x)[:300] for x in rb[:3]]

    return parsed, None


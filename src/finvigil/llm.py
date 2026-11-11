"""LLM client supporting DeepSeek-V3/R1 and Alibaba Qwen 2.5 via Ollama or OpenAI-compatible APIs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from finvigil.config import settings

_BPE_PATTERN = re.compile(r"[A-Za-z0-9]+|[^\w\s]")

PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "ollama": "http://localhost:11434/v1",
}

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen2.5:7b",
    "ollama": "qwen2.5:7b",
}


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM invocation with measured token counts and optional R1 reasoning."""

    content: str
    reasoning_content: str | None
    input_tokens: int
    output_tokens: int
    model_used: str
    is_live_call: bool


def count_tokens(text: str) -> int:
    """Compute actual subword token count of a string (works offline in air-gapped CI)."""
    if not text:
        return 0
    # Subword BPE approximation: words > 6 chars split into ~2 tokens + punctuation tokens
    raw_tokens = _BPE_PATTERN.findall(text)
    return sum(max(1, (len(tok) + 4) // 5) for tok in raw_tokens)


def strip_deepseek_r1_think_tags(raw_text: str) -> tuple[str, str | None]:
    """Extract <think>...</think> chain-of-thought blocks emitted by DeepSeek-R1 / QwQ models."""
    match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if not match:
        return raw_text.strip(), None
    reasoning = match.group(1).strip()
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return cleaned, reasoning


def invoke_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    fallback_output: str,
    temperature: float = 0.0,
) -> LLMResponse:
    """Invoke DeepSeek / Qwen / Ollama via OpenAI-compatible API, or measure tokens on fallback."""
    provider = settings.llm_provider
    if provider == "offline":
        in_tok = count_tokens(system_prompt) + count_tokens(user_prompt)
        out_tok = count_tokens(fallback_output)
        return LLMResponse(
            content=fallback_output,
            reasoning_content=None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model_used=f"offline-synthesized({settings.llm_model})",
            is_live_call=False,
        )

    base_url = (
        settings.llm_base_url.rstrip("/")
        if settings.llm_base_url
        else PROVIDER_DEFAULT_BASE_URLS.get(provider, "http://localhost:11434/v1")
    )
    model_name = settings.llm_model or PROVIDER_DEFAULT_MODELS.get(provider, "deepseek-r1:8b")

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 280,
    }

    try:
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]["message"]
        raw_content = str(choice.get("content", ""))
        reasoning_field = choice.get("reasoning_content")
        cleaned_content, think_tag_reasoning = strip_deepseek_r1_think_tags(raw_content)
        reasoning = str(reasoning_field) if reasoning_field else think_tag_reasoning

        usage = data.get("usage") or {}
        in_tok = int(
            usage.get("prompt_tokens")
            or (count_tokens(system_prompt) + count_tokens(user_prompt))
        )
        out_tok = int(usage.get("completion_tokens") or count_tokens(raw_content))

        return LLMResponse(
            content=cleaned_content or fallback_output,
            reasoning_content=reasoning,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model_used=model_name,
            is_live_call=True,
        )
    except Exception:
        in_tok = count_tokens(system_prompt) + count_tokens(user_prompt)
        out_tok = count_tokens(fallback_output)
        return LLMResponse(
            content=fallback_output,
            reasoning_content=None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model_used=f"{model_name}-fallback",
            is_live_call=False,
        )



import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


def get_llm_response(
    prompt: str,
    system: str = "",
    config: dict = None,
    messages: list = None,
    json_mode: bool = False,
) -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    Parameters
    ----------
    prompt   : The user message.
    system   : Optional system prompt.
    config   : Dict with keys llm_provider, ollama_model.
    messages : Optional pre-built messages list (overrides prompt/system).
    json_mode: Hint to return JSON – provider-specific handling.

    Returns
    -------
    str
    """
    cfg = config or {}
    provider = cfg.get("llm_provider", "ollama")

    if provider == "ollama":
        return _ollama(prompt, system, cfg, messages, json_mode)
    elif provider == "anthropic":
        return _anthropic(prompt, system, cfg, messages, json_mode)
    elif provider == "openai":
        return _openai(prompt, system, cfg, messages, json_mode)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ── Ollama ─────────────────────────────────────────────────────────────────────

def _ollama(prompt, system, cfg, messages, json_mode) -> str:
    model = cfg.get("ollama_model", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ── Anthropic ──────────────────────────────────────────────────────────────────

def _anthropic(prompt, system, cfg, messages, json_mode) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment / .env")

    client = anthropic.Anthropic(api_key=api_key)

    if messages is None:
        msgs = [{"role": "user", "content": prompt}]
    else:
        msgs = [m for m in messages if m["role"] != "system"]

    sys_prompt = system or ""
    if json_mode:
        sys_prompt += "\n\nRespond ONLY with valid JSON. No markdown fences, no preamble."

    kwargs = dict(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=msgs,
    )
    if sys_prompt:
        kwargs["system"] = sys_prompt

    resp = client.messages.create(**kwargs)
    return resp.content[0].text.strip()


# ── OpenAI ─────────────────────────────────────────────────────────────────────

def _openai(prompt, system, cfg, messages, json_mode) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment / .env")

    client = OpenAI(api_key=api_key)

    if messages is None:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
    else:
        msgs = messages

    kwargs = dict(
        model="gpt-4o-mini",
        messages=msgs,
        max_tokens=2048,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()

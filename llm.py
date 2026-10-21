"""LLM access: one function, two providers (Google and OpenRouter)."""

import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

# The dashboard starts the solver with the user's own keys and ARC_NO_DOTENV=1, so the
# server's .env keys are never mixed in. Command-line runs load .env as usual.
if not os.environ.get("ARC_NO_DOTENV"):
    load_dotenv()

MODELS = {  # default model of each provider
    "google": "gemini-3.5-flash-lite",
    "openrouter": "google/gemini-3.5-flash-lite",
}
# Google text models offered in the dashboard. Selected from the API models list and tested
# with this key: Gemini 2.5 is closed to new users (404) and 3.1 Pro has no quota here (429).
GOOGLE_MODELS = [
    "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview",
]
# These models reject the "minimal" level with HTTP 400.
NO_MINIMAL = {"gemini-3.7-flash", "gemini-3.8-flash"}
REASONING_LEVELS = ["minimal", "low", "medium", "high"]
MAX_ATTEMPTS = 4
RETRY_STATUS = {429, 500, 502, 503, 504}
KEY_STATUS = {401, 403, 429}  # quota exhausted or key rejected: another key may work
TIMEOUT = 600  # high reasoning can take a while


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    reasoning_tokens: int
    answer_tokens: int


def _key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} not found in .env")
    return value


def _google_keys() -> list[tuple[str, str]]:
    """GOOGLE_API_KEY first, then GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ... from the environment."""
    def order(name: str) -> tuple:
        suffix = name.removeprefix("GOOGLE_API_KEY").removeprefix("_")
        return (name != "GOOGLE_API_KEY", int(suffix) if suffix.isdigit() else 0, name)

    names = [n for n in os.environ if n == "GOOGLE_API_KEY" or n.startswith("GOOGLE_API_KEY_")]
    return [(n, os.environ[n].strip()) for n in sorted(names, key=order) if os.environ[n].strip()]


_google_key_pos = 0  # index of the key currently in use


def check_google_key(key: str) -> tuple[bool, str]:
    """Cheap validity check (lists one model; spends no tokens). Returns (ok, message)."""
    try:
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models",
                      params={"pageSize": 1}, headers={"x-goog-api-key": key}, timeout=20)
    except httpx.HTTPError as e:
        return False, f"sem conexão com a API ({type(e).__name__})"
    if r.status_code == 200:
        return True, "ok"
    if r.status_code in (400, 401, 403):
        return False, "a API recusou a chave (inválida ou sem permissão)"
    if r.status_code == 429:
        return True, "ok (no limite de taxa agora)"  # valid key, just busy
    return False, f"resposta inesperada da API (HTTP {r.status_code})"


def _call_google(system: str, prompt: str, reasoning: str, model: str) -> LLMResult:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": reasoning}},
    }
    global _google_key_pos
    keys = _google_keys()
    if not keys:
        raise RuntimeError("GOOGLE_API_KEY not found in .env")
    # Start from the key that worked last. If it is out of quota (429) or rejected
    # (401/403), move on to the next key; with a single key this just raises the error.
    for step in range(len(keys)):
        pos = (_google_key_pos + step) % len(keys)
        name, value = keys[pos]
        r = httpx.post(url, json=body, timeout=TIMEOUT, headers={"x-goog-api-key": value})
        if r.status_code in KEY_STATUS and step < len(keys) - 1:
            print(f"  {name} got HTTP {r.status_code}; trying the next key", flush=True)
            continue
        if pos != _google_key_pos:
            print(f"  now using {name}", flush=True)
        _google_key_pos = pos
        break
    r.raise_for_status()
    data = r.json()
    parts = data["candidates"][0].get("content", {}).get("parts", [])
    # Skip thought summaries, keep only the answer text.
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = data.get("usageMetadata", {})
    return LLMResult(
        text=text,
        input_tokens=u.get("promptTokenCount", 0),
        reasoning_tokens=u.get("thoughtsTokenCount", 0),
        answer_tokens=u.get("candidatesTokenCount", 0),
    )


def _call_openrouter(system: str, prompt: str, reasoning_level: str, model: str) -> LLMResult:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": reasoning_level},
    }
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions", json=body,
        timeout=TIMEOUT,
        headers={"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}"},
    )
    r.raise_for_status()
    data = r.json()
    u = data.get("usage", {})
    reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    # OpenRouter's completion_tokens includes reasoning tokens.
    return LLMResult(
        text=data["choices"][0]["message"].get("content") or "",
        input_tokens=u.get("prompt_tokens", 0),
        reasoning_tokens=reasoning,
        answer_tokens=u.get("completion_tokens", 0) - reasoning,
    )


def _call(system: str, prompt: str, provider: str, reasoning: str, model: str) -> LLMResult:
    if provider == "google":
        return _call_google(system, prompt, reasoning, model)
    if provider == "openrouter":
        return _call_openrouter(system, prompt, reasoning, model)
    raise ValueError(f"Unknown provider: {provider}")


def _is_transient(e: Exception) -> bool:
    """Overload / rate limit / network problems are worth retrying; other errors are not."""
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in RETRY_STATUS
    return isinstance(e, httpx.TransportError)  # timeouts, connection resets


def ask(
    system: str, prompt: str, provider: str = "google", reasoning: str = "high",
    model: str | None = None,
) -> LLMResult:
    """Call the model, retrying transient failures with increasing waits (10 s, 20 s, 40 s)."""
    if reasoning not in REASONING_LEVELS:
        raise ValueError(f"reasoning must be one of {REASONING_LEVELS}")
    model = model or MODELS[provider]
    if reasoning == "minimal" and model in NO_MINIMAL:
        raise ValueError(f"{model} does not support the minimal reasoning level")
    for attempt in range(MAX_ATTEMPTS):
        try:
            return _call(system, prompt, provider, reasoning, model)
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1 or not _is_transient(e):
                raise
            wait = 10 * 2**attempt
            print(f"  transient error ({type(e).__name__}); retrying in {wait}s "
                  f"(attempt {attempt + 2}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(wait)

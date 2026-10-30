"""Target LLM adapters behind one interface.

- MockProvider: deterministic, simulated responses for pipeline tests. Never evidence.
- OllamaProvider: local models. Sets num_ctx explicitly (Ollama's default context is
  small and silently truncates) and refuses prompts that would not fit.
- OpenAICompatProvider: any /chat/completions endpoint (OpenAI, OpenRouter, Groq,
  Together, vLLM, Gemini's OpenAI-compatible endpoint). Key read from an env var.

Adapters never substitute a different model: the requested model id is sent as-is and
the provider-reported model/version is recorded next to it.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from instruction_drift.config import ProviderConfig
from instruction_drift.schemas import GenerationParams
from instruction_drift.tokenization import Tokenizer, reference_tokenizer

Messages = list[dict[str, str]]


@dataclass
class LLMResult:
    text: str
    provider_input_tokens: int | None
    provider_output_tokens: int | None
    latency_ms: float
    model_version: str | None
    request_id: str | None = None
    finish_reason: str | None = None
    retry_count: int = 0
    error: str | None = None
    raw_meta: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Non-retryable failure (bad request, auth, context overflow)."""


class RetryableError(RuntimeError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}", _retry_after(resp))
    if resp.status_code >= 400:
        raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:500]}")


class BaseProvider:
    kind = "base"

    def __init__(self, cfg: ProviderConfig, *, sleep: Callable[[float], None] = time.sleep,
                 tokenizer: Tokenizer | None = None):
        self.cfg = cfg
        self.model_id = cfg.model_id
        self._sleep = sleep
        self._last_call = 0.0
        self.tok = tokenizer or reference_tokenizer()

    def model_version(self) -> str | None:
        return None

    def _call(self, messages: Messages, params: GenerationParams) -> LLMResult:
        raise NotImplementedError

    def generate(self, messages: Messages, params: GenerationParams) -> LLMResult:
        """Bounded retries with exponential backoff and jitter; never raises."""
        attempts = self.cfg.max_retries + 1
        for attempt in range(attempts):
            self._throttle()
            try:
                res = self._call(messages, params)
                res.retry_count = attempt
                return res
            except ProviderError as e:
                return self._failure(str(e), attempt)
            except (RetryableError, httpx.TimeoutException, httpx.TransportError) as e:
                if attempt == attempts - 1:
                    return self._failure(f"retries exhausted: {e!r}", attempt)
                delay = min(60.0, 2.0 ** attempt + random.random())
                if isinstance(e, RetryableError) and e.retry_after is not None:
                    delay = min(120.0, max(delay, e.retry_after))
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _failure(self, msg: str, attempt: int) -> LLMResult:
        return LLMResult(text="", provider_input_tokens=None, provider_output_tokens=None,
                         latency_ms=0.0, model_version=self.model_version(),
                         retry_count=attempt, error=msg)

    def _throttle(self) -> None:
        if self.cfg.requests_per_minute:
            wait = 60.0 / self.cfg.requests_per_minute - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = time.monotonic()


class OllamaProvider(BaseProvider):
    kind = "ollama"
    # Reference-tokenizer counts can undercount a model's own tokenizer and omit
    # chat-template tokens, so prompts must fit with this margin.
    CONTEXT_MARGIN = 1.25
    PER_MESSAGE_OVERHEAD = 8

    def __init__(self, cfg: ProviderConfig, *, client: httpx.Client | None = None, **kw):
        super().__init__(cfg, **kw)
        self.base_url = (cfg.base_url or "http://localhost:11434").rstrip("/")
        self.client = client or httpx.Client(timeout=cfg.timeout_s)
        self._version: str | None = None

    def model_version(self) -> str | None:
        if self._version is None:
            try:
                tags = self.client.get(f"{self.base_url}/api/tags").json()["models"]
                m = next(m for m in tags if m["name"] == self.model_id)
                det = m.get("details", {})
                self._version = (f"{m['digest']} {det.get('parameter_size', '?')} "
                                 f"{det.get('quantization_level', '?')}")
            except (httpx.HTTPError, KeyError, StopIteration, ValueError):
                return None
        return self._version

    def estimated_prompt_tokens(self, messages: Messages) -> int:
        return sum(self.tok.count(m["content"]) + self.PER_MESSAGE_OVERHEAD for m in messages)

    def _call(self, messages: Messages, params: GenerationParams) -> LLMResult:
        if params.context_window is None:
            raise ProviderError("Ollama requires generation.context_window (num_ctx)")
        est = self.estimated_prompt_tokens(messages)
        need = math.ceil(est * self.CONTEXT_MARGIN) + params.max_output_tokens
        if need > params.context_window:
            raise ProviderError(f"context overflow: ~{est} prompt tokens + "
                                f"{params.max_output_tokens} output exceed num_ctx="
                                f"{params.context_window} with margin")
        options = {"temperature": params.temperature, "num_predict": params.max_output_tokens,
                   "num_ctx": params.context_window}
        if params.seed is not None:
            options["seed"] = params.seed
        if params.top_p is not None:
            options["top_p"] = params.top_p
        body = {"model": self.model_id, "messages": messages, "stream": False,
                "options": options, "keep_alive": "30m"}
        t0 = time.perf_counter()
        resp = self.client.post(f"{self.base_url}/api/chat", json=body)
        latency = (time.perf_counter() - t0) * 1000
        _raise_for_status(resp)
        data = resp.json()
        p_in = data.get("prompt_eval_count")
        if p_in is not None and p_in >= params.context_window - params.max_output_tokens:
            raise ProviderError(f"possible prompt truncation: prompt_eval_count={p_in} "
                                f"reaches num_ctx={params.context_window}")
        meta = {k: data.get(k) for k in ("total_duration", "load_duration",
                                          "prompt_eval_duration", "eval_duration", "model")}
        return LLMResult(
            text=data["message"]["content"], provider_input_tokens=p_in,
            provider_output_tokens=data.get("eval_count"), latency_ms=latency,
            model_version=self.model_version(), finish_reason=data.get("done_reason"),
            raw_meta=meta)


class OpenAICompatProvider(BaseProvider):
    kind = "openai_compat"

    def __init__(self, cfg: ProviderConfig, *, client: httpx.Client | None = None, **kw):
        super().__init__(cfg, **kw)
        if not cfg.base_url:
            raise ValueError("openai_compat requires provider.base_url")
        if not cfg.api_key_env:
            raise ValueError("openai_compat requires provider.api_key_env")
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise ValueError(f"Environment variable {cfg.api_key_env} is not set")
        self.base_url = cfg.base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=cfg.timeout_s)
        self._headers = {"Authorization": f"Bearer {key}"}
        self._last_model: str | None = None

    def model_version(self) -> str | None:
        return self._last_model

    def _call(self, messages: Messages, params: GenerationParams) -> LLMResult:
        body = {"model": self.model_id, "messages": messages,
                "temperature": params.temperature, "max_tokens": params.max_output_tokens}
        if params.seed is not None:
            body["seed"] = params.seed
        if params.top_p is not None:
            body["top_p"] = params.top_p
        t0 = time.perf_counter()
        resp = self.client.post(f"{self.base_url}/chat/completions", json=body,
                                headers=self._headers)
        latency = (time.perf_counter() - t0) * 1000
        _raise_for_status(resp)
        data = resp.json()
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RetryableError(f"malformed response: {e!r}") from e
        usage = data.get("usage") or {}
        self._last_model = data.get("model")
        return LLMResult(
            text=text, provider_input_tokens=usage.get("prompt_tokens"),
            provider_output_tokens=usage.get("completion_tokens"), latency_ms=latency,
            model_version=self._last_model, request_id=data.get("id"),
            finish_reason=choice.get("finish_reason"),
            raw_meta={"system_fingerprint": data.get("system_fingerprint")})


class MockProvider(BaseProvider):
    """Simulated target model for pipeline tests. Outputs are synthetic.

    For a known checkpoint message it returns the scripted compliant reply or the
    scripted violating reply. Violation probability rises with prompt length:
    p = min(max_p, base_rate + slope_per_1k * prompt_tokens / 1000), drawn from a
    hash of (seed, message content) so runs are reproducible.
    """

    kind = "mock"
    VERSION = "mock-1.0"

    def __init__(self, cfg: ProviderConfig, replies: dict[str, tuple[str, str | None]], *,
                 seed: int = 0, base_rate: float = 0.05, slope_per_1k: float = 0.1,
                 max_p: float = 0.9, **kw):
        super().__init__(cfg, **kw)
        self.replies = replies
        self.seed, self.base_rate, self.slope, self.max_p = seed, base_rate, slope_per_1k, max_p

    def model_version(self) -> str | None:
        return self.VERSION

    def _call(self, messages: Messages, params: GenerationParams) -> LLMResult:
        user = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user" and m["content"] in self.replies), None)
        if user is None:
            raise ProviderError("mock has no scripted reply for this prompt")
        compliant, violating = self.replies[user]
        tokens = sum(self.tok.count(m["content"]) for m in messages)
        p = min(self.max_p, self.base_rate + self.slope * tokens / 1000)
        h = hashlib.sha256(f"{self.seed}|{len(messages)}|{user}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        text = violating if (violating is not None and u < p) else compliant
        return LLMResult(text=text, provider_input_tokens=None, provider_output_tokens=None,
                         latency_ms=0.0, model_version=self.VERSION, finish_reason="stop",
                         raw_meta={"simulated": True, "p_violation": p})


def make_provider(cfg: ProviderConfig, **kw) -> BaseProvider:
    if cfg.kind == "ollama":
        return OllamaProvider(cfg, **kw)
    if cfg.kind == "openai_compat":
        return OpenAICompatProvider(cfg, **kw)
    if cfg.kind == "mock":
        return MockProvider(cfg, **kw)
    raise ValueError(cfg.kind)

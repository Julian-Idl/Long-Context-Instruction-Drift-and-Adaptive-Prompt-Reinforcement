import json

import httpx
import pytest

from instruction_drift.config import ProviderConfig
from instruction_drift.llm import MockProvider, OllamaProvider, OpenAICompatProvider
from instruction_drift.schemas import GenerationParams

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ollama(handler, retries=2):
    cfg = ProviderConfig(kind="ollama", model_id="qwen2.5:7b", max_retries=retries)
    return OllamaProvider(cfg, client=_client(handler), sleep=lambda s: None)


def _ok_ollama(request):
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b", "digest": "abc",
                                                     "details": {"parameter_size": "7.6B",
                                                                 "quantization_level": "Q4_K_M"}}]})
    body = json.loads(request.content)
    assert body["options"]["num_ctx"] == 4096 and body["options"]["seed"] == 7
    assert body["options"]["temperature"] == 0.0 and body["stream"] is False
    return httpx.Response(200, json={"message": {"content": "hi"}, "prompt_eval_count": 12,
                                     "eval_count": 3, "done_reason": "stop"})


def test_ollama_success_records_usage_and_version():
    res = _ollama(_ok_ollama).generate(MSGS, GenerationParams(seed=7, context_window=4096))
    assert res.error is None and res.text == "hi"
    assert (res.provider_input_tokens, res.provider_output_tokens) == (12, 3)
    assert res.model_version == "abc 7.6B Q4_K_M"


def test_ollama_requires_context_window_and_refuses_overflow():
    p = _ollama(_ok_ollama)
    assert "context_window" in p.generate(MSGS, GenerationParams()).error
    long = [{"role": "user", "content": "word " * 5000}]
    res = p.generate(long, GenerationParams(context_window=4096))
    assert "context overflow" in res.error and res.retry_count == 0


def test_ollama_flags_possible_truncation():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "x"},
                                         "prompt_eval_count": 4000, "eval_count": 1})
    res = _ollama(handler).generate(MSGS, GenerationParams(context_window=4096,
                                                           max_output_tokens=200))
    assert "truncation" in res.error


def test_retries_are_bounded_and_honour_retry_after():
    calls, sleeps = [], []

    def handler(request):
        if request.url.path == "/api/chat":
            calls.append(1)
        return httpx.Response(429, headers={"retry-after": "3"}, text="slow down")
    cfg = ProviderConfig(kind="ollama", model_id="m", max_retries=2)
    p = OllamaProvider(cfg, client=_client(handler), sleep=sleeps.append)
    res = p.generate(MSGS, GenerationParams(context_window=4096))
    assert len(calls) == 3 and res.retry_count == 2 and "retries exhausted" in res.error
    assert len(sleeps) == 2 and all(s >= 3 for s in sleeps)


def test_client_errors_are_not_retried_and_timeouts_are():
    calls = []

    def bad_request(request):
        if request.url.path == "/api/chat":
            calls.append(1)
        return httpx.Response(400, text="bad")
    assert "HTTP 400" in _ollama(bad_request).generate(
        MSGS, GenerationParams(context_window=4096)).error
    assert len(calls) == 1

    attempts = []

    def flaky(request):
        if request.url.path == "/api/chat":
            attempts.append(1)
        if len(attempts) == 1 and request.url.path == "/api/chat":
            raise httpx.ReadTimeout("timeout")
        return _ok_ollama(request)
    res = _ollama(flaky).generate(MSGS, GenerationParams(seed=7, context_window=4096))
    assert res.error is None and res.retry_count == 1


def _openai(handler, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    cfg = ProviderConfig(kind="openai_compat", model_id="gpt-x", base_url="https://api.test/v1",
                         api_key_env="TEST_KEY", max_retries=1)
    return OpenAICompatProvider(cfg, client=_client(handler), sleep=lambda s: None)


def test_openai_compat_parses_response_and_tolerates_missing_usage(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == "Bearer sk-test"
        assert json.loads(request.content)["model"] == "gpt-x"
        return httpx.Response(200, json={"id": "req1", "model": "gpt-x-2026-01-01",
                                         "choices": [{"message": {"content": "ok"},
                                                      "finish_reason": "stop"}]})
    res = _openai(handler, monkeypatch).generate(MSGS, GenerationParams())
    assert res.text == "ok" and res.request_id == "req1"
    assert res.model_version == "gpt-x-2026-01-01"
    assert res.provider_input_tokens is None  # missing usage stays None, not 0


def test_openai_compat_malformed_output_retried_then_reported(monkeypatch):
    res = _openai(lambda r: httpx.Response(200, json={"choices": []}), monkeypatch).generate(
        MSGS, GenerationParams())
    assert res.error and "malformed" in res.error and res.retry_count == 1


def test_openai_compat_requires_env_key(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = ProviderConfig(kind="openai_compat", model_id="m", base_url="https://x",
                         api_key_env="MISSING_KEY")
    with pytest.raises(ValueError, match="MISSING_KEY"):
        OpenAICompatProvider(cfg)


def test_mock_is_deterministic_and_length_sensitive():
    cfg = ProviderConfig(kind="mock", model_id="mock")
    replies = {"hello": ("good", "bad")}
    p = MockProvider(cfg, replies, seed=1, base_rate=0.0, slope_per_1k=0.0)
    assert p.generate(MSGS, GenerationParams()).text == "good"
    always = MockProvider(cfg, replies, seed=1, base_rate=1.0, max_p=1.0)
    assert always.generate(MSGS, GenerationParams()).text == "bad"
    a = MockProvider(cfg, replies, seed=3).generate(MSGS, GenerationParams()).text
    assert a == MockProvider(cfg, replies, seed=3).generate(MSGS, GenerationParams()).text
    unknown = p.generate([{"role": "user", "content": "?"}], GenerationParams())
    assert unknown.error

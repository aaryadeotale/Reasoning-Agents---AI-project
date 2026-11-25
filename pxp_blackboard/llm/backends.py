"""LLM backends. Anything with .generate(messages, ...) -> LLMResponse works.

- MockBackend          deterministic, for tests/demos (no model needed)
- OpenAICompatBackend  talks to any OpenAI-compatible server: vLLM, Ollama (/v1), llama.cpp
- HFBackend            in-process Hugging Face transformers (lazy import; lock-guarded)
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence


class BackendError(RuntimeError):
    """Network / server failure while calling a model."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough fallback when a server reports no usage


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str


class LLMBackend(Protocol):
    name: str

    def generate(self, messages: Sequence[dict], *, max_tokens: int = 512,
                 temperature: float = 0.2, json_mode: bool = True) -> LLMResponse: ...


class MockBackend:
    def __init__(self, responder: Callable[[list], str], name: str = "mock", latency_s: float = 0.0,
                 jitter: float = 0.0):
        self.responder, self.name, self.latency_s, self.jitter = responder, name, latency_s, jitter

    def generate(self, messages, *, max_tokens=512, temperature=0.2, json_mode=True) -> LLMResponse:
        t0 = time.perf_counter()
        if self.latency_s:  # simulated "thinking" time so demos play out live
            time.sleep(max(0.0, self.latency_s * (1 + random.uniform(-self.jitter, self.jitter))))
        text = self.responder(list(messages))
        return LLMResponse(text, sum(estimate_tokens(m["content"]) for m in messages),
                           estimate_tokens(text), time.perf_counter() - t0, self.name)


class OpenAICompatBackend:
    """Examples: vLLM  http://localhost:8000/v1   |   Ollama  http://localhost:11434/v1"""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: float = 120.0):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.timeout = api_key, timeout
        self.name = model
        self._json_mode_ok = True  # flips to False if the server rejects response_format

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def generate(self, messages, *, max_tokens=512, temperature=0.2, json_mode=True) -> LLMResponse:
        payload = {"model": self.model, "messages": list(messages),
                   "max_tokens": max_tokens, "temperature": temperature}
        use_json = json_mode and self._json_mode_ok
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            try:
                data = self._post(payload)
            except urllib.error.HTTPError as e:
                if use_json and e.code in (400, 422):  # server doesn't support JSON mode
                    self._json_mode_ok = False
                    payload.pop("response_format")
                    data = self._post(payload)
                else:
                    raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise BackendError(f"{self.base_url}: {e}") from e
        text = (data["choices"][0]["message"].get("content") or "")
        usage = data.get("usage") or {}
        return LLMResponse(
            text,
            usage.get("prompt_tokens") or sum(estimate_tokens(m["content"]) for m in messages),
            usage.get("completion_tokens") or estimate_tokens(text),
            time.perf_counter() - t0, self.model)


class HFBackend:
    """In-process transformers model (needs `pip install transformers torch accelerate`).
    One model instance => generation is serialized with a lock. Untested in CI (no GPU)."""

    def __init__(self, model_id: str, device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
        self.name = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype="auto")
        self._lock = threading.Lock()

    def generate(self, messages, *, max_tokens=512, temperature=0.2, json_mode=True) -> LLMResponse:
        t0 = time.perf_counter()
        ids = self.tok.apply_chat_template(list(messages), add_generation_prompt=True,
                                           return_tensors="pt").to(self.model.device)
        kw = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kw["temperature"] = temperature
        with self._lock:
            out = self.model.generate(ids, **kw)
        new = out[0][ids.shape[-1]:]
        return LLMResponse(self.tok.decode(new, skip_special_tokens=True),
                           int(ids.shape[-1]), int(new.shape[-1]), time.perf_counter() - t0, self.name)

"""Unified LLM client wrapping OpenAI-compatible APIs."""

from __future__ import annotations

import os
import time

from openai import OpenAI

from ..config.llm import LLMConfig


def _env(name: str, default: str) -> str:
    """Read ``MASKILLS_<name>``, falling back to the LangMARL-era ``LANGMARL_``
    spelling so shell setups from before the split keep working."""
    return os.environ.get(f"MASKILLS_{name}") or os.environ.get(f"LANGMARL_{name}", default)


_MAX_RETRIES = int(_env("LLM_MAX_RETRIES", "4"))
_RETRY_BACKOFF = float(_env("LLM_RETRY_BACKOFF", "2.0"))


def _is_retryable(exc: BaseException) -> bool:
    """Heuristic: which API failures should be retried?

    OpenRouter under load returns transient errors with shapes like:
      * ``json.JSONDecodeError`` — upstream sent HTML instead of JSON.
      * ``TypeError: 'NoneType' object is not subscriptable`` — content
        field came back null.
      * ``openai.APIConnectionError`` / ``APITimeoutError`` / ``RateLimitError``
        / 5xx ``InternalServerError`` — standard transport failures.

    All of these are retryable; anything else (auth, schema, etc.) is not.
    """
    name = type(exc).__name__
    if name in (
        "JSONDecodeError",
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "APIStatusError",
    ):
        return True
    if isinstance(exc, TypeError):
        msg = str(exc)
        # ``response.choices[0].message.content is None`` (raw subscript) or
        # the explicit re-raise from ``_create_with_retry`` below.
        if "NoneType" in msg or "subscriptable" in msg or "no usable content" in msg:
            return True
    return False


def _create_with_retry(client_create_fn, **params):
    """Call ``chat.completions.create`` with bounded retries on transient errors."""
    last_exc: BaseException = RuntimeError("unreachable")
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            response = client_create_fn(**params)
            # Some upstreams return choices=None or message.content=None;
            # treat that the same as a transport hiccup and retry.
            if not response.choices or response.choices[0].message.content is None:
                raise TypeError(
                    "LLM response has no usable content: "
                    f"choices={response.choices!r}"
                )
            return response
        except BaseException as e:
            if not _is_retryable(e) or attempt > _MAX_RETRIES:
                raise
            last_exc = e
            sleep_s = _RETRY_BACKOFF * (2 ** (attempt - 1))
            time.sleep(min(sleep_s, 30.0))
    raise last_exc


class LLMClient:
    """Unified LLM client for all OpenAI-compatible providers."""

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        api_key = llm_config.get_api_key()
        # Client-side timeout: without this, an upstream server that closes a
        # connection (CLOSE_WAIT sockets) leaves worker threads hung forever
        # because the openai SDK / httpx default does not abort on stalled
        # reads.  120s is generous for legitimate completions and fails
        # fast on dead sockets.
        timeout_s = float(_env("LLM_TIMEOUT", "120"))
        if llm_config.base_url:
            self._client = OpenAI(
                base_url=llm_config.base_url, api_key=api_key, timeout=timeout_s,
            )
        else:
            self._client = OpenAI(api_key=api_key, timeout=timeout_s)
        self.model = llm_config.model_string

    @property
    def raw_client(self) -> OpenAI:
        """Access the underlying OpenAI client."""
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        max_tokens = max_tokens or self.config.max_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        params = {"model": self.model, "messages": messages}
        model_lower = self.model.lower()
        if "o1" in model_lower or "o3" in model_lower or "gpt-5" in model_lower:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = _create_with_retry(self._client.chat.completions.create, **params)
        return response.choices[0].message.content.strip()

    def chat_with_usage(
        self,
        system_prompt: str,
        user_input: str,
        max_tokens: int = None,
    ) -> tuple[str, dict[str, int]]:
        """Chat and return (response_text, {input: N, output: N})."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self.chat_messages_with_usage(messages, max_tokens=max_tokens)

    def chat_messages_with_usage(
        self,
        messages: list[dict],
        max_tokens: int = None,
    ) -> tuple[str, dict[str, int]]:
        """Multi-turn chat with a pre-built message list."""
        max_tokens = max_tokens or self.config.max_tokens

        params = {"model": self.model, "messages": messages}
        model_lower = self.model.lower()
        if "o1" in model_lower or "o3" in model_lower or "gpt-5" in model_lower:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = _create_with_retry(self._client.chat.completions.create, **params)
        text = response.choices[0].message.content.strip()

        if hasattr(response, 'usage') and response.usage:
            tokens = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
            }
        else:
            joined = " ".join(m.get("content", "") for m in messages)
            tokens = {
                "input": len(joined.split()) * 2,
                "output": len(text.split()) * 2,
            }
        return text, tokens

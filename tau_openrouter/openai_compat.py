import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, TypeVar

from openai import OpenAI, RateLimitError

T = TypeVar("T")


def provider_api_key(provider: str) -> str:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for provider=openai.")
        return api_key
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required for provider=openrouter.")
        return api_key
    raise SystemExit(f"Unsupported provider: {provider}")


def provider_base_url(provider: str) -> str | None:
    if provider == "openai":
        return os.getenv("OPENAI_API_BASE")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    raise SystemExit(f"Unsupported provider: {provider}")


def provider_headers(provider: str) -> dict[str, str] | None:
    if provider != "openrouter":
        return None
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers or None


def build_client(provider: str) -> OpenAI:
    return OpenAI(
        api_key=provider_api_key(provider),
        base_url=provider_base_url(provider),
        default_headers=provider_headers(provider),
        max_retries=env_int("TAU_BENCH_SDK_MAX_RETRIES") or 8,
    )


def env_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


def env_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else None


def env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_str(name: str) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else None


def env_name(prefix: str, field: str) -> str:
    return f"{prefix}{field}"


def chat_kwargs(prefix: str = "TAU_BENCH_", temperature: float | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    top_p = env_float(env_name(prefix, "TOP_P")) or env_float("TAU_BENCH_TOP_P")
    if top_p is not None:
        kwargs["top_p"] = top_p
    max_tokens = env_int(env_name(prefix, "MAX_TOKENS")) or env_int("TAU_BENCH_MAX_TOKENS")
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    timeout = env_float(env_name(prefix, "TIMEOUT")) or env_float("TAU_BENCH_TIMEOUT")
    if timeout is not None:
        kwargs["timeout"] = timeout
    extra_body = build_extra_body(prefix)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def build_extra_body(prefix: str = "TAU_BENCH_") -> dict[str, Any] | None:
    chat_template_kwargs: dict[str, Any] = {}
    thinking = env_bool(env_name(prefix, "THINKING"))
    if thinking is None:
        thinking = env_bool("TAU_BENCH_THINKING")
    if thinking is not None:
        chat_template_kwargs["thinking"] = thinking
    reasoning_effort = os.getenv(env_name(prefix, "REASONING_EFFORT")) or os.getenv("TAU_BENCH_REASONING_EFFORT")
    # NVIDIA's OpenAI-compatible endpoint accepts reasoning controls, but sending
    # them while "thinking" is disabled can still route requests through a slower
    # reasoning path. Keep the user simulator on the simple path by omitting
    # reasoning settings unless thinking is explicitly enabled.
    if reasoning_effort and thinking:
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs:
        return {"chat_template_kwargs": chat_template_kwargs}
    return None


def request_settings(prefix: str = "TAU_BENCH_", temperature: float | None = None) -> dict[str, Any]:
    thinking = env_bool(env_name(prefix, "THINKING"))
    if thinking is None:
        thinking = env_bool("TAU_BENCH_THINKING")
    reasoning_effort = env_str(env_name(prefix, "REASONING_EFFORT")) or env_str("TAU_BENCH_REASONING_EFFORT")
    return {
        "temperature": temperature,
        "top_p": env_float(env_name(prefix, "TOP_P")) or env_float("TAU_BENCH_TOP_P"),
        "max_tokens": env_int(env_name(prefix, "MAX_TOKENS")) or env_int("TAU_BENCH_MAX_TOKENS"),
        "timeout": env_float(env_name(prefix, "TIMEOUT")) or env_float("TAU_BENCH_TIMEOUT"),
        "thinking": thinking,
        "reasoning_effort": reasoning_effort if thinking else None,
        "retries": env_int(env_name(prefix, "RETRIES")) or env_int("TAU_BENCH_RETRIES") or 0,
    }


def maybe_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


class _RequestLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self._interval_s = 60.0 / requests_per_minute
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    def wait_for_turn(self, context: str) -> None:
        with self._lock:
            now = time.monotonic()
            reserved_at = max(now, self._next_allowed_at)
            self._next_allowed_at = reserved_at + self._interval_s
        delay_s = reserved_at - now
        if delay_s > 0:
            print(f"Rate limiter sleeping {delay_s:.1f}s before {context} request")
            time.sleep(delay_s)


_REQUEST_LIMITER: _RequestLimiter | None = None


def requests_per_minute() -> int:
    value = env_int("TAU_BENCH_REQUESTS_PER_MINUTE")
    return 12 if value in (None, 0) else value


def wait_for_rate_limit(context: str) -> None:
    global _REQUEST_LIMITER
    rpm = requests_per_minute()
    if rpm < 1:
        return
    if _REQUEST_LIMITER is None:
        _REQUEST_LIMITER = _RequestLimiter(requests_per_minute=rpm)
    _REQUEST_LIMITER.wait_for_turn(context)


def should_log_api_headers() -> bool:
    enabled = env_bool("TAU_BENCH_LOG_API_HEADERS")
    return True if enabled is None else enabled


def error_header_snapshot(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    if response is None or getattr(response, "headers", None) is None:
        return {}
    headers = response.headers
    interesting = [
        "retry-after",
        "x-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "server",
        "date",
    ]
    snapshot = {key: headers.get(key) for key in interesting if headers.get(key) is not None}
    request_id = getattr(exc, "request_id", None)
    if request_id and "x-request-id" not in snapshot:
        snapshot["x-request-id"] = request_id
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        snapshot["status_code"] = status_code
    return snapshot


def log_api_error(context: str, exc: Exception) -> None:
    if not should_log_api_headers():
        return
    snapshot = error_header_snapshot(exc)
    if snapshot:
        print(f"{context} API error headers: {snapshot}")
    else:
        print(f"{context} API error headers: <none available>")


def rate_limit_retry_count() -> int:
    value = env_int("TAU_BENCH_RATE_LIMIT_RETRIES")
    return 6 if value is None else value


def rate_limit_backoff_base_s() -> float:
    value = env_float("TAU_BENCH_RATE_LIMIT_BACKOFF_BASE")
    return 5.0 if value is None else value


def rate_limit_backoff_max_s() -> float:
    value = env_float("TAU_BENCH_RATE_LIMIT_BACKOFF_MAX")
    return 60.0 if value is None else value


def rate_limit_backoff_jitter_s() -> float:
    value = env_float("TAU_BENCH_RATE_LIMIT_BACKOFF_JITTER")
    return 1.0 if value is None else value


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("retry-after")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def call_with_rate_limit_retry(context: str, fn: Callable[[], T]) -> T:
    max_retries = rate_limit_retry_count()
    base_s = rate_limit_backoff_base_s()
    max_s = rate_limit_backoff_max_s()
    jitter_s = rate_limit_backoff_jitter_s()
    attempt = 0
    while True:
        try:
            return fn()
        except RateLimitError as exc:
            log_api_error(context, exc)
            if attempt >= max_retries:
                raise
            delay_s = min(max_s, base_s * (2**attempt))
            header_delay_s = retry_after_seconds(exc)
            if header_delay_s is not None:
                delay_s = max(delay_s, header_delay_s)
            if jitter_s > 0:
                delay_s += random.uniform(0.0, jitter_s)
            print(
                f"{context} hit rate limit on attempt {attempt + 1}/{max_retries + 1}; "
                f"retrying in {delay_s:.1f}s"
            )
            time.sleep(delay_s)
            attempt += 1

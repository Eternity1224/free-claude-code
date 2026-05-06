"""OpenAI-style chat base for :class:`OpenAIChatTransport` (NIM, etc.).

All transient errors (network, empty HTTP 200, 5xx, 429) are retried forever,
with delays: 1s, 2s, 3s, 5s, 10s, then 15s (with jitter).
Never exposes network errors to the user.
"""

import asyncio
import json
import random
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI, APIConnectionError, RateLimitError

from core.anthropic import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    map_stop_reason,
)
from providers.base import BaseProvider, ProviderConfig
from providers.rate_limit import GlobalRateLimiter


def _iter_heuristic_tool_use_sse(
    sse: SSEBuilder, tool_use: dict[str, Any]
) -> Iterator[str]:
    """Emit SSE for one heuristic tool_use block (closes open text/thinking first)."""
    if tool_use.get("name") == "Task" and isinstance(tool_use.get("input"), dict):
        task_input = tool_use["input"]
        if task_input.get("run_in_background") is not False:
            task_input["run_in_background"] = False
    yield from sse.close_content_blocks()
    block_idx = sse.blocks.allocate_index()
    yield sse.content_block_start(
        block_idx,
        "tool_use",
        id=tool_use["id"],
        name=tool_use["name"],
    )
    yield sse.content_block_delta(
        block_idx,
        "input_json_delta",
        json.dumps(tool_use["input"]),
    )
    yield sse.content_block_stop(block_idx)


class OpenAIChatTransport(BaseProvider):
    """Base for OpenAI-compatible ``/chat/completions`` adapters (NIM, …)."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            provider_name.lower(),
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        http_client = None
        if config.proxy:
            http_client = httpx.AsyncClient(
                proxy=config.proxy,
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    read=config.http_read_timeout,
                    write=config.http_write_timeout,
                ),
            )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,  # We handle all retries ourselves
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
            http_client=http_client,
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        client = getattr(self, "_client", None)
        if client is not None:
            await client.aclose()

    @abstractmethod
    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build request body. Must be implemented by subclasses."""

    def _handle_extra_reasoning(
        self, delta: Any, sse: SSEBuilder, *, thinking_enabled: bool
    ) -> Iterator[str]:
        """Hook for provider-specific reasoning (e.g. OpenRouter reasoning_details)."""
        return iter(())

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Return a modified request body for one retry, or None."""
        return None

    def _is_thinking_enabled(self, request: Any, override: bool | None) -> bool:
        """Determine if thinking should be enabled for this request."""
        if override is not None:
            return override
        return getattr(request, "thinking_enabled", False)

    # --------------------------------------------------------------------------
    #  Retry forever with delays: 1s, 2s, 3s, 5s, 10s, then 15s (with jitter)
    # --------------------------------------------------------------------------
    async def _create_stream_with_retry(self, body: dict) -> tuple[Any, dict]:
        """Create a streaming chat completion, retrying forever on retryable errors.

        Delays after each failure:
        - 1st retry: ~1s (±50% jitter)
        - 2nd: ~2s
        - 3rd: ~3s
        - 4th: ~5s
        - 5th: ~10s
        - 6th and further: ~15s
        """
        # Fixed delays in seconds (without jitter)
        DELAYS = [1, 2, 3, 5, 10]
        # After these, use 15s
        DEFAULT_DELAY = 15

        attempt = 0  # number of retries already performed (0 = first try)
        last_error = None

        while True:
            try:
                if attempt > 0:
                    # Compute delay based on attempt count
                    idx = attempt - 1  # 0-based index into DELAYS
                    if idx < len(DELAYS):
                        delay = DELAYS[idx]
                    else:
                        delay = DEFAULT_DELAY
                    # Add jitter: random factor between 0.5 and 1.5
                    delay = delay * (0.5 + random.random())
                    logger.warning(
                        f"[{self._provider_name}] 🔄 Retry #{attempt} "
                        f"dans {delay:.1f}s après erreur: {last_error}"
                    )
                    await asyncio.sleep(delay)

                # Attempt to create the stream
                stream = await self._global_rate_limiter.execute_with_retry(
                    self._client.chat.completions.create, **body, stream=True
                )
                return stream, body

            except Exception as e:
                last_error = e
                error_lower = str(e).lower()
                status_code = getattr(e, 'status_code', None)
                is_rate_limit = (
                    isinstance(e, RateLimitError) or "429" in error_lower or "rate limit" in error_lower
                )
                # Detect HTTP 200 empty/malformed responses (broken proxies)
                is_empty_200 = (
                    status_code == 200
                    and any(phrase in error_lower for phrase in [
                        "empty", "malformed", "incomplete", "unexpected end", "no content"
                    ])
                )
                # Retryable conditions
                is_retryable = (
                    isinstance(e, APIConnectionError)
                    or "timeout" in error_lower
                    or "connection" in error_lower
                    or "incomplete chunked read" in error_lower
                    or "peer closed connection" in error_lower
                    or "remote disconnected" in error_lower
                    or "socket" in error_lower
                    or "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e)
                    or is_rate_limit
                    or is_empty_200
                )

                # If not retryable, try to modify request body (once)
                if not is_retryable:
                    retry_body = self._get_retry_request_body(e, body)
                    if retry_body is not None:
                        try:
                            logger.warning(
                                f"[{self._provider_name}] Tentative avec body modifié "
                                f"(après erreur {status_code or '?'})"
                            )
                            stream = await self._global_rate_limiter.execute_with_retry(
                                self._client.chat.completions.create, **retry_body, stream=True
                            )
                            return stream, retry_body
                        except Exception as e2:
                            last_error = e2
                            # Fall through: increment attempt and retry normally
                            attempt += 1
                            continue

                if not is_retryable:
                    # Non‑retryable error (e.g., 400 bad request, auth error) – give up
                    logger.error(f"[{self._provider_name}] Non‑retryable error, giving up: {e}")
                    raise

                # Otherwise, increment attempt counter and retry
                attempt += 1
                # Continue the loop (will wait and retry)

    # --------------------------------------------------------------------------
    #  Tool argument handling (unchanged)
    # --------------------------------------------------------------------------
    def _emit_tool_arg_delta(
        self, sse: SSEBuilder, tc_index: int, args: str
    ) -> Iterator[str]:
        if not args:
            return
        state = sse.blocks.tool_states.get(tc_index)
        if state is None:
            return
        if state.name == "Task":
            parsed = sse.blocks.buffer_task_args(tc_index, args)
            if parsed is not None:
                yield sse.emit_tool_delta(tc_index, json.dumps(parsed))
            return
        yield sse.emit_tool_delta(tc_index, args)

    def _process_tool_call(self, tc: dict, sse: SSEBuilder) -> Iterator[str]:
        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_states)

        fn_delta = tc.get("function", {})
        incoming_name = fn_delta.get("name")
        arguments = fn_delta.get("arguments", "") or ""

        if tc.get("id") is not None:
            sse.blocks.set_stream_tool_id(tc_index, tc.get("id"))

        if incoming_name is not None:
            sse.blocks.register_tool_name(tc_index, incoming_name)

        state = sse.blocks.tool_states.get(tc_index)
        resolved_id = (state.tool_id if state and state.tool_id else None) or tc.get("id")
        resolved_name = (state.name if state else "") or ""

        if not state or not state.started:
            name_ok = bool((resolved_name or "").strip())
            if name_ok:
                tool_id = str(resolved_id) if resolved_id else f"tool_{uuid.uuid4()}"
                display_name = (resolved_name or "").strip() or "tool_call"
                yield sse.start_tool_block(tc_index, tool_id, display_name)
                state = sse.blocks.tool_states[tc_index]
                if state.pre_start_args:
                    pre = state.pre_start_args
                    state.pre_start_args = ""
                    yield from self._emit_tool_arg_delta(sse, tc_index, pre)

        state = sse.blocks.tool_states.get(tc_index)
        if not arguments:
            return
        if state is None or not state.started:
            state = sse.blocks.ensure_tool_state(tc_index)
            if not (resolved_name or "").strip():
                state.pre_start_args += arguments
                return

        yield from self._emit_tool_arg_delta(sse, tc_index, arguments)

    def _flush_task_arg_buffers(self, sse: SSEBuilder) -> Iterator[str]:
        for tool_index, out in sse.blocks.flush_task_arg_buffers():
            yield sse.emit_tool_delta(tool_index, out)

    # --------------------------------------------------------------------------
    #  Main streaming entry point (never exposes network errors to user)
    # --------------------------------------------------------------------------
    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format, retrying forever on failures."""
        with logger.contextualize(request_id=request_id):
            async for event in self._stream_response_impl(
                request, input_tokens, request_id, thinking_enabled=thinking_enabled
            ):
                yield event

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        """Internal streaming that never exposes socket errors – retries forever."""
        tag = self._provider_name
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(
            message_id,
            request.model,
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        body = self._build_request_body(request, thinking_enabled=thinking_enabled)
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "{}_STREAM:{} model={} msgs={} tools={}",
            tag,
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        finish_reason = None
        usage_info = None

        # Acquire rate limiter slot, then create stream (will retry forever internally)
        async with self._global_rate_limiter.concurrency_slot():
            # _create_stream_with_retry loops forever; only non-retryable errors raise.
            try:
                stream, _ = await self._create_stream_with_retry(body)
            except Exception as e:
                # Non-retryable error (e.g., 400, 401). Send generic error to user.
                logger.error(f"[{tag}] Non-retryable error, sending fallback message: {e}")
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(
                    "I'm sorry, but your request cannot be processed due to an invalid configuration. "
                    "Please contact support."
                )
                for event in sse.close_all_blocks():
                    yield event
                yield sse.message_delta("error", 0)
                yield sse.message_stop()
                return

            # Process the stream, catching mid-stream interruptions gracefully
            try:
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage_info = chunk.usage

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is None:
                        continue

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        logger.debug("{} finish_reason: {}", tag, finish_reason)

                    reasoning = getattr(delta, "reasoning_content", None)
                    if thinking_enabled and reasoning:
                        for event in sse.ensure_thinking_block():
                            yield event
                        yield sse.emit_thinking_delta(reasoning)

                    for event in self._handle_extra_reasoning(
                        delta, sse, thinking_enabled=thinking_enabled
                    ):
                        yield event

                    if delta.content:
                        for part in think_parser.feed(delta.content):
                            if part.type == ContentType.THINKING:
                                if not thinking_enabled:
                                    continue
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(part.content)
                            else:
                                filtered_text, detected_tools = heuristic_parser.feed(part.content)
                                if filtered_text:
                                    for event in sse.ensure_text_block():
                                        yield event
                                    yield sse.emit_text_delta(filtered_text)
                                for tool_use in detected_tools:
                                    for event in _iter_heuristic_tool_use_sse(sse, tool_use):
                                        yield event

                    if delta.tool_calls:
                        for event in sse.close_content_blocks():
                            yield event
                        for tc in delta.tool_calls:
                            tc_info = {
                                "index": tc.index,
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for event in self._process_tool_call(tc_info, sse):
                                yield event

            except (asyncio.CancelledError, GeneratorExit):
                # Control flow: propagate
                raise
            except Exception as e:
                error_lower = str(e).lower()
                # Mid‑stream network interruption – finish gracefully with what we have
                is_network_error = any(phrase in error_lower for phrase in [
                    "incomplete chunked read",
                    "peer closed connection",
                    "remote disconnected",
                    "socket connection",
                    "empty",
                    "malformed",
                    "connection reset",
                    "timeout",
                ])
                if is_network_error:
                    logger.warning(f"[{tag}] ⚠ Stream interrupted mid‑flight, finishing early: {e}")
                    if finish_reason is None:
                        finish_reason = "error"
                else:
                    # Non‑network error – complete with error stop reason
                    logger.error(f"[{tag}] Unexpected streaming error: {e}")
                    if finish_reason is None:
                        finish_reason = "error"

        # Flush remaining content from parsers
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                if thinking_enabled:
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(remaining.content)
            elif remaining.type == ContentType.TEXT:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(remaining.content)

        for tool_use in heuristic_parser.flush():
            for event in _iter_heuristic_tool_use_sse(sse, tool_use):
                yield event

        # Ensure at least one content block
        has_started_tool = any(s.started for s in sse.blocks.tool_states.values())
        has_content_blocks = (
            sse.blocks.text_index != -1
            or sse.blocks.thinking_index != -1
            or has_started_tool
        )
        if not has_content_blocks:
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")
        elif not has_started_tool and not sse.accumulated_text.strip() and sse.accumulated_reasoning.strip():
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        for event in self._flush_task_arg_buffers(sse):
            yield event

        for event in sse.close_all_blocks():
            yield event

        # Token accounting
        if usage_info and hasattr(usage_info, "completion_tokens"):
            output_tokens = usage_info.completion_tokens
        else:
            output_tokens = sse.estimate_output_tokens()

        if usage_info and hasattr(usage_info, "prompt_tokens"):
            provider_input = usage_info.prompt_tokens
            if isinstance(provider_input, int):
                logger.debug(
                    "TOKEN_ESTIMATE: our={} provider={} diff={:+d}",
                    input_tokens,
                    provider_input,
                    provider_input - input_tokens,
                )

        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()

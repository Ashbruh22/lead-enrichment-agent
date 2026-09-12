"""Gemini client: structured output, retries, and token/cost accounting.

Every call goes through :meth:`GeminiClient.structured`, which hands Gemini a
Pydantic model as its ``response_schema``. That means the model is constrained
at decode time rather than asked nicely in a prompt, so we get parseable JSON
instead of markdown-fenced hope.

Usage is accumulated in a :class:`UsageTracker` so the CLI can report tokens
and estimated cost per domain.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from .config import Settings, estimate_cost

log = logging.getLogger(__name__)


class LlmUnavailableError(RuntimeError):
    """Raised when no API key is configured."""


@dataclass
class UsageTracker:
    """Running token/cost totals for one domain (or a whole run)."""

    model: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)

    def record(self, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
        self.calls += 1
        if usage is None:
            return
        self.prompt_tokens += usage.prompt_token_count or 0
        # Thinking tokens bill as output, so count them where the money is.
        self.completion_tokens += (usage.candidates_token_count or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )

    def merge(self, other: UsageTracker) -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.errors.extend(other.errors)


def minimal_thinking(model: str) -> types.ThinkingConfig | None:
    """The least deliberation this model family will accept.

    These are extraction tasks over text we already supply, not reasoning
    problems, and thinking tokens bill at the output rate -- so we ask for as
    little as the API allows. Gemini 2.x takes a numeric ``thinking_budget``;
    Gemini 3 replaced it with a coarse ``thinking_level`` and rejects the old
    field with a 400. Version-shaped names are matched explicitly, and anything
    unrecognised (``gemini-flash-latest``) gets no thinking config at all rather
    than a guess -- :meth:`GeminiClient.structured` also degrades on a 400, so a
    wrong guess here costs a wasted call rather than the run.
    """
    match = re.match(r"(?:models/)?gemini-(\d+)", model)
    if match is None:
        return None
    return (
        types.ThinkingConfig(thinking_level="low")
        if int(match.group(1)) >= 3
        else types.ThinkingConfig(thinking_budget=0)
    )


def _is_invalid_argument(exc: BaseException) -> bool:
    """A 400 from the API -- usually an option this model does not support."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 400 or "INVALID_ARGUMENT" in str(exc)


def _is_retryable(exc: BaseException) -> bool:
    """Rate limits, overload and transient 5xx are worth another try."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429", "resource_exhausted", "rate limit", "overloaded", "unavailable", "503",
        )
    )


class GeminiClient:
    """Thin async wrapper around the Gemini SDK."""

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise LlmUnavailableError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://aistudio.google.com/apikey"
            )
        self.settings = settings
        self.model = settings.gemini_model
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._thinking = minimal_thinking(self.model)

    async def structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        system: str,
        usage: UsageTracker,
        temperature: float = 0.1,
    ) -> BaseModel | None:
        """One schema-constrained call. Returns ``None`` rather than raising."""

        def build(thinking: types.ThinkingConfig | None) -> types.GenerateContentConfig:
            return types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
                thinking_config=thinking,
            )

        try:
            response = await self._call_with_retries(prompt, build(self._thinking))
        except Exception as exc:
            # A 400 while a thinking config is set is almost always the model
            # refusing that knob (the 2.x/3.x split). Drop it, retry once, and
            # remember, so an unknown model costs one wasted call per run.
            retryable = self._thinking is not None and _is_invalid_argument(exc)
            if not retryable:
                message = f"LLM call failed: {type(exc).__name__}: {str(exc)[:200]}"
                log.warning(message)
                usage.errors.append(message)
                usage.calls += 1
                return None

            log.info("%s rejected the thinking config; retrying without it", self.model)
            self._thinking = None
            try:
                response = await self._call_with_retries(prompt, build(None))
            except Exception as retry_exc:
                message = f"LLM call failed: {type(retry_exc).__name__}: {str(retry_exc)[:200]}"
                log.warning(message)
                usage.errors.append(message)
                usage.calls += 1
                return None

        usage.record(response.usage_metadata)
        parsed = self._coerce(response, schema)
        if parsed is None:
            usage.errors.append("LLM returned output that did not satisfy the schema")
        return parsed

    async def _call_with_retries(self, prompt: str, config: types.GenerateContentConfig):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.llm_attempts),
            # Capped at 45s rather than 30s: a free-tier 429 is a per-minute
            # quota, and a backoff that tops out below that window just burns
            # the remaining attempts inside the same minute and gives up.
            wait=wait_exponential(multiplier=3, min=3, max=45),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                return await self._client.aio.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )

    @staticmethod
    def _coerce(response, schema: type[BaseModel]) -> BaseModel | None:
        """Prefer the SDK's parsed object; fall back to parsing the raw text."""
        if isinstance(response.parsed, schema):
            return response.parsed
        raw = (response.text or "").strip()
        if not raw:
            return None
        if raw.startswith("```"):  # defensive: strip a stray markdown fence
            raw = raw.strip("`").removeprefix("json").strip()
        try:
            return schema.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("Could not coerce LLM output into %s: %s", schema.__name__, exc)
            return None

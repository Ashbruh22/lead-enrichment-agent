"""Tests for model-specific call configuration and cost accounting.

The thinking-config split is the kind of thing that only shows up as a 400 in
production, so it is pinned here: Gemini 2.x takes a numeric ``thinking_budget``
and Gemini 3 rejects that field outright in favour of ``thinking_level``.
"""

from google.genai.types import ThinkingLevel

from lead_agent.config import PRICING, estimate_cost
from lead_agent.llm import UsageTracker, _is_invalid_argument, minimal_thinking


class TestMinimalThinking:
    def test_gemini_3_uses_thinking_level(self) -> None:
        config = minimal_thinking("gemini-3.6-flash")
        assert config is not None
        assert config.thinking_level is ThinkingLevel.LOW
        assert config.thinking_budget is None

    def test_gemini_2_uses_thinking_budget(self) -> None:
        config = minimal_thinking("gemini-2.5-flash")
        assert config is not None
        assert config.thinking_budget == 0
        assert config.thinking_level is None

    def test_handles_the_models_prefix(self) -> None:
        assert minimal_thinking("models/gemini-3.6-flash").thinking_level is ThinkingLevel.LOW

    def test_future_major_versions_stay_on_thinking_level(self) -> None:
        assert minimal_thinking("gemini-12.0-flash").thinking_level is ThinkingLevel.LOW

    def test_unversioned_alias_gets_no_thinking_config(self) -> None:
        # Guessing wrong costs a wasted call; sending nothing is always valid.
        assert minimal_thinking("gemini-flash-latest") is None


class TestIsInvalidArgument:
    def test_detects_status_name(self) -> None:
        assert _is_invalid_argument(RuntimeError("400 INVALID_ARGUMENT. {...}"))

    def test_detects_code_attribute(self) -> None:
        exc = RuntimeError("something went wrong")
        exc.code = 400
        assert _is_invalid_argument(exc)

    def test_ignores_rate_limits(self) -> None:
        assert not _is_invalid_argument(RuntimeError("429 RESOURCE_EXHAUSTED"))


class TestUsageTracker:
    def test_totals_and_cost_follow_the_price_table(self) -> None:
        tracker = UsageTracker(model="gemini-3.6-flash")
        tracker.prompt_tokens = 1_000_000
        tracker.completion_tokens = 1_000_000

        price_in, price_out = PRICING["gemini-3.6-flash"]
        assert tracker.total_tokens == 2_000_000
        assert tracker.cost_usd == price_in + price_out

    def test_merge_accumulates_both_sides(self) -> None:
        a = UsageTracker(model="gemini-3.6-flash", calls=1, prompt_tokens=10, completion_tokens=2)
        b = UsageTracker(model="gemini-3.6-flash", calls=2, prompt_tokens=5, completion_tokens=3)
        b.errors.append("boom")

        a.merge(b)

        assert (a.calls, a.prompt_tokens, a.completion_tokens) == (3, 15, 5)
        assert a.errors == ["boom"]

    def test_unknown_model_still_costs_something(self) -> None:
        # An unpriced model must not silently report a free run.
        assert estimate_cost("gemini-99-experimental", 1_000_000, 1_000_000) > 0

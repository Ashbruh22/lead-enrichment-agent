"""Runtime configuration.

Everything tunable lives here so no magic numbers are buried in the pipeline.
Values come from the environment (a local ``.env`` is loaded automatically) and
fall back to the defaults below.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Paid-tier rates in USD per 1M tokens, from https://ai.google.dev/gemini-api/docs/pricing
# (verified 2026-09-11). The free tier bills nothing, but we still report what the
# run *would* have cost -- that is the number that matters when this is scaled up.
PRICING: dict[str, tuple[float, float]] = {
    # model: (input $/1M, output $/1M)
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}
DEFAULT_PRICE = (0.30, 2.50)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class Settings(BaseSettings):
    """Environment-driven settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash")
    tavily_api_key: str = Field(default="")

    # --- crawling -------------------------------------------------------- #
    page_timeout_ms: int = 25_000
    settle_timeout_ms: int = 3_000
    """How long to wait for network to go quiet after DOM load (SPA content)."""
    max_pages_per_domain: int = 5
    max_link_candidates: int = 25
    page_concurrency: int = 3
    domain_concurrency: int = 2
    request_delay_seconds: float = 0.5
    respect_robots: bool = True

    # --- token budgets --------------------------------------------------- #
    max_chars_per_page: int = 6_000
    max_chars_total: int = 30_000

    # --- resilience ------------------------------------------------------ #
    fetch_attempts: int = 2
    llm_attempts: int = 3
    blocked_text_threshold: int = 200
    """Pages yielding less clean text than this are treated as bot-blocked."""

    @property
    def price_per_million(self) -> tuple[float, float]:
        return PRICING.get(self.gemini_model, DEFAULT_PRICE)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of a call at published paid-tier rates."""
    price_in, price_out = PRICING.get(model, DEFAULT_PRICE)
    return (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out


def load_settings() -> Settings:
    """Load settings, preferring a ``.env`` next to the project root."""
    from dotenv import load_dotenv

    load_dotenv(override=False)
    return Settings()

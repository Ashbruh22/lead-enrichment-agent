"""Pydantic schemas.

Two families of models live here, and the split is deliberate:

* ``Llm*`` models are the *wire* schema handed to Gemini as ``response_schema``.
  Gemini accepts only the OpenAPI scalar subset (str / int / float / bool /
  enum / list / nested object / nullable), so these use plain ``str`` for URLs
  and e-mails.
* The remaining models are the *strict* schema used for our own output. They
  carry real ``HttpUrl`` / ``EmailStr`` validation.

Converting from the first to the second is where hallucinated contact data gets
dropped -- see :mod:`lead_agent.pipeline`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator

# --------------------------------------------------------------------------- #
# Wire schema (sent to the LLM)
# --------------------------------------------------------------------------- #


class LlmTeamMember(BaseModel):
    """A leadership/team member as reported by the LLM."""

    name: str = Field(description="Full name of the person, exactly as written on the page.")
    role: str | None = Field(
        default=None, description="Job title, e.g. 'Co-Founder & CEO'. Null if not stated."
    )
    linkedin_url: str | None = Field(
        default=None,
        description=(
            "The person's LinkedIn profile URL, but ONLY if it appears verbatim in the "
            "provided content. Never guess or construct one."
        ),
    )


class LlmCompanyIntel(BaseModel):
    """The structured payload we ask Gemini to produce."""

    company_overview: str = Field(
        description="Exactly two sentences describing what the company does."
    )
    target_audience: str = Field(
        description=(
            "One sentence naming the ideal customer profile, e.g. "
            "'Developers building and testing backend APIs'."
        )
    )
    emails: list[str] = Field(
        description=(
            "Generic/public e-mail addresses found in the content (contact@, sales@, "
            "support@, ...). Copy them verbatim; do not invent addresses."
        )
    )
    leadership: list[LlmTeamMember] = Field(
        description="Named founders, executives or team members mentioned in the content."
    )
    data_confidence_score: float = Field(
        description=(
            "0.0-1.0 estimate of how complete and trustworthy this extraction is, "
            "given how much relevant information the content actually contained."
        )
    )


class SelectedPages(BaseModel):
    """LLM link-selection result (the dynamic-navigation step)."""

    urls: list[str] = Field(description="The chosen URLs, copied verbatim from the candidate list.")
    reasoning: str = Field(description="One short sentence explaining the choice.")


# --------------------------------------------------------------------------- #
# Strict output schema
# --------------------------------------------------------------------------- #


class TeamMember(BaseModel):
    """One person, plus how we came by their profile URL -- or why we did not.

    ``source`` deliberately distinguishes the three empty cases. A reviewer
    reading the output should be able to tell a feature that was switched off
    from one that ran and found nothing, and both from one that could not run
    because every search provider was blocked. Collapsing them all into
    "unknown" hides whether the lookup works at all.
    """

    name: str
    role: str | None = None
    linkedin_url: HttpUrl | None = None
    source: Literal[
        "website",           # the URL was on the company's own site
        "search",            # found by external search, name-matched
        "searched_not_found",  # we looked and there was no confident match
        "search_unavailable",  # every provider was blocked or unconfigured
        "not_searched",      # lookup disabled for this run
    ] = "website"


class ContactPoints(BaseModel):
    emails: list[EmailStr] = Field(default_factory=list)
    contact_page_url: HttpUrl | None = None


class CompanyIntel(BaseModel):
    company_overview: str
    target_audience: str
    contact_points: ContactPoints = Field(default_factory=ContactPoints)
    leadership: list[TeamMember] = Field(default_factory=list)
    data_confidence_score: float = Field(ge=0.0, le=1.0)

    # Kept alongside the blended score so a reviewer can see how it was reached.
    llm_self_score: float | None = None
    completeness_score: float | None = None

    @field_validator("data_confidence_score", mode="before")
    @classmethod
    def _clamp(cls, v: float) -> float:
        """Clamp before the range check.

        Models do occasionally return 1.2 or -0.1 for a "0.0 to 1.0" field, and
        an out-of-range score is not a good reason to throw away an otherwise
        complete extraction.
        """
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class PageRecord(BaseModel):
    """Provenance for a single fetched page."""

    url: str
    ok: bool
    chars: int = 0
    note: str | None = None


class RunMetrics(BaseModel):
    pages_fetched: int = 0
    pages_failed: int = 0
    agent_rounds: int = 0
    """Assess-and-continue rounds run for this domain; 1 means no follow-up."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    duration_seconds: float = 0.0


class DomainResult(BaseModel):
    domain: str
    url: str
    status: Literal["ok", "partial", "failed"]
    intel: CompanyIntel | None = None
    pages: list[PageRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    scraped_at: datetime = Field(default_factory=datetime.now)


class RunReport(BaseModel):
    """Top level of ``output.json``."""

    generated_at: datetime = Field(default_factory=datetime.now)
    model: str
    results: list[DomainResult]
    totals: RunMetrics

"""Data confidence scoring.

The LLM is asked for a self-assessed confidence, but self-reported confidence
is a weak signal on its own -- models tend to be uniformly optimistic. So the
final score blends it with a completeness score we compute ourselves from
things that are objectively checkable: how many pages we actually retrieved,
whether we found contact e-mails, whether we found named people, and whether
those people have verifiable LinkedIn URLs.

The deterministic half carries more weight because it cannot be talked into a
number it has not earned.
"""

from __future__ import annotations

from .models import CompanyIntel

LLM_WEIGHT = 0.35
COMPLETENESS_WEIGHT = 0.65


def completeness(intel: CompanyIntel, pages_ok: int, pages_attempted: int) -> float:
    """Objective 0.0-1.0 measure of how much we actually came away with."""
    score = 0.0

    # Coverage: did the crawl work at all?
    if pages_attempted:
        score += 0.20 * min(1.0, pages_ok / min(pages_attempted, 3))

    # Core narrative fields.
    if len(intel.company_overview.strip()) > 60:
        score += 0.20
    if len(intel.target_audience.strip()) > 20:
        score += 0.15

    # Contact points.
    if intel.contact_points.emails:
        score += 0.20 if len(intel.contact_points.emails) > 1 else 0.13

    # People, weighted towards people we can actually reach.
    if intel.leadership:
        score += 0.13
        if any(member.linkedin_url for member in intel.leadership):
            score += 0.12

    return round(min(1.0, score), 3)


def blended_confidence(
    intel: CompanyIntel, pages_ok: int, pages_attempted: int
) -> tuple[float, float]:
    """Final confidence and the completeness component it was blended from."""
    objective = completeness(intel, pages_ok, pages_attempted)
    llm_score = max(0.0, min(1.0, intel.data_confidence_score))
    blended = LLM_WEIGHT * llm_score + COMPLETENESS_WEIGHT * objective
    return round(min(1.0, blended), 3), objective

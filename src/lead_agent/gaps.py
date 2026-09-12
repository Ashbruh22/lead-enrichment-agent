"""What the extraction failed to find, and whether it is worth another round.

This is the "assess" half of the agent loop. After each extraction the result
is inspected for the fields that actually matter to a sales researcher; if any
are missing, the gaps are handed back to the navigator so it can go looking for
pages that would fill them specifically.

Kept deterministic on purpose. Asking the model "did you miss anything?" invites
it to invent work; checking the object we just built cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CompanyIntel

# Below this, an overview is a tagline rather than a description.
_MIN_OVERVIEW_CHARS = 60
_MIN_AUDIENCE_CHARS = 20


@dataclass(frozen=True)
class Gap:
    """One missing piece, phrased for the follow-up prompt."""

    key: str
    description: str
    """What to go looking for, in words the navigator can act on."""


GAP_LEADERSHIP = Gap(
    "leadership",
    "the names and job titles of founders, executives or team members",
)
GAP_EMAILS = Gap(
    "emails",
    "a public contact e-mail address (contact@, sales@, support@, hello@)",
)
GAP_OVERVIEW = Gap(
    "overview",
    "a clear description of what the company actually builds and sells",
)
GAP_AUDIENCE = Gap(
    "audience",
    "who the product is built for -- the customers or industries it targets",
)


def identify_gaps(intel: CompanyIntel) -> list[Gap]:
    """The missing fields worth spending another round on, most valuable first.

    Leadership leads because it is both the hardest field to find and the one a
    lead-enrichment user is most often actually after; a company with no named
    people is a much weaker lead than one missing a support e-mail.
    """
    gaps: list[Gap] = []
    if not intel.leadership:
        gaps.append(GAP_LEADERSHIP)
    if not intel.contact_points.emails:
        gaps.append(GAP_EMAILS)
    if len(intel.company_overview.strip()) < _MIN_OVERVIEW_CHARS:
        gaps.append(GAP_OVERVIEW)
    if len(intel.target_audience.strip()) < _MIN_AUDIENCE_CHARS:
        gaps.append(GAP_AUDIENCE)
    return gaps


def format_gaps(gaps: list[Gap]) -> str:
    """Gaps as a bulleted list for the follow-up prompt."""
    return "\n".join(f"- {gap.description}" for gap in gaps)


def summarise_gaps(gaps: list[Gap]) -> str:
    """Short human-readable form for CLI progress output."""
    return ", ".join(gap.key for gap in gaps)

"""Tests for the assess half of the agent loop.

Gap detection decides whether a domain costs one LLM round or three, so the
conditions are pinned here rather than left to behave-as-observed.
"""

from lead_agent.gaps import (
    GAP_AUDIENCE,
    GAP_EMAILS,
    GAP_LEADERSHIP,
    GAP_OVERVIEW,
    format_gaps,
    identify_gaps,
    summarise_gaps,
)
from lead_agent.models import CompanyIntel, ContactPoints, TeamMember


def make_intel(
    *,
    overview: str = "A" * 100,
    audience: str = "B" * 40,
    emails: list[str] | None = None,
    leadership: list[TeamMember] | None = None,
) -> CompanyIntel:
    return CompanyIntel(
        company_overview=overview,
        target_audience=audience,
        contact_points=ContactPoints(emails=emails or []),
        leadership=leadership or [],
        data_confidence_score=0.5,
    )


class TestIdentifyGaps:
    def test_a_complete_extraction_has_no_gaps(self) -> None:
        intel = make_intel(
            emails=["hello@example.org"],
            leadership=[TeamMember(name="Jane Doe", role="CEO")],
        )
        assert identify_gaps(intel) == []

    def test_missing_people_is_a_gap(self) -> None:
        intel = make_intel(emails=["hello@example.org"])
        assert GAP_LEADERSHIP in identify_gaps(intel)

    def test_missing_emails_is_a_gap(self) -> None:
        intel = make_intel(leadership=[TeamMember(name="Jane Doe")])
        assert GAP_EMAILS in identify_gaps(intel)

    def test_a_tagline_overview_is_a_gap(self) -> None:
        intel = make_intel(
            overview="We do things.",
            emails=["hello@example.org"],
            leadership=[TeamMember(name="Jane Doe")],
        )
        assert GAP_OVERVIEW in identify_gaps(intel)

    def test_a_thin_audience_is_a_gap(self) -> None:
        intel = make_intel(
            audience="Everyone",
            emails=["hello@example.org"],
            leadership=[TeamMember(name="Jane Doe")],
        )
        assert GAP_AUDIENCE in identify_gaps(intel)

    def test_whitespace_does_not_count_as_content(self) -> None:
        intel = make_intel(
            overview="   ",
            audience="   ",
            emails=["hello@example.org"],
            leadership=[TeamMember(name="Jane Doe")],
        )
        assert {GAP_OVERVIEW, GAP_AUDIENCE} <= set(identify_gaps(intel))

    def test_leadership_is_reported_first(self) -> None:
        # The loop spends its follow-up budget on the earliest gaps, and a
        # company with no named people is the weakest kind of lead.
        gaps = identify_gaps(make_intel(overview="short", audience="x"))
        assert gaps[0] is GAP_LEADERSHIP

    def test_an_empty_extraction_reports_everything(self) -> None:
        assert len(identify_gaps(make_intel(overview="", audience=""))) == 4


class TestFormatting:
    def test_format_gaps_is_a_bulleted_list(self) -> None:
        rendered = format_gaps([GAP_LEADERSHIP, GAP_EMAILS])
        assert rendered.startswith("- ")
        assert rendered.count("\n") == 1

    def test_summarise_gaps_is_short_keys(self) -> None:
        assert summarise_gaps([GAP_LEADERSHIP, GAP_EMAILS]) == "leadership, emails"

    def test_no_gaps_formats_to_nothing(self) -> None:
        assert format_gaps([]) == ""
        assert summarise_gaps([]) == ""

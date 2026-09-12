"""Tests for domain normalisation and the LLM-output validation gate."""

from lead_agent.models import LlmCompanyIntel, LlmTeamMember
from lead_agent.pipeline import _build_intel, _verify_against_source, normalise_domain

SOURCE = """
Contact us at hello@acmerobotics.com or sales@acmerobotics.com.
Dana Whitfield, Co-Founder & CEO: https://www.linkedin.com/in/dana-whitfield-7b32a1
"""
KNOWN_EMAILS = ["hello@acmerobotics.com", "sales@acmerobotics.com"]
KNOWN_LINKEDINS = ["https://www.linkedin.com/in/dana-whitfield-7b32a1"]


def build(raw: LlmCompanyIntel, contact_page: str | None = None):
    return _build_intel(
        raw,
        source_blob=SOURCE,
        known_emails=KNOWN_EMAILS,
        known_linkedins=KNOWN_LINKEDINS,
        contact_page=contact_page,
    )


def llm_payload(**overrides) -> LlmCompanyIntel:
    data = {
        "company_overview": "Acme builds robots. They are used in warehouses.",
        "target_audience": "Warehouse operations teams.",
        "emails": [],
        "leadership": [],
        "data_confidence_score": 0.8,
    }
    data.update(overrides)
    return LlmCompanyIntel(**data)


class TestNormaliseDomain:
    def test_strips_scheme_path_and_www(self) -> None:
        assert normalise_domain("https://www.Postman.com/pricing") == "postman.com"

    def test_passes_through_bare_domain(self) -> None:
        assert normalise_domain("  vapi.ai  ") == "vapi.ai"

    def test_keeps_meaningful_subdomain(self) -> None:
        assert normalise_domain("https://docs.vapi.ai") == "docs.vapi.ai"


class TestVerifyAgainstSource:
    def test_keeps_strings_present_in_source(self) -> None:
        assert _verify_against_source(["hello@acmerobotics.com"], SOURCE) == [
            "hello@acmerobotics.com"
        ]

    def test_drops_strings_absent_from_source(self) -> None:
        assert _verify_against_source(["invented@acmerobotics.com"], SOURCE) == []

    def test_is_case_insensitive(self) -> None:
        assert _verify_against_source(["HELLO@ACMEROBOTICS.COM"], SOURCE) != []


class TestBuildIntel:
    def test_drops_hallucinated_emails(self) -> None:
        intel = build(llm_payload(emails=["ceo@acmerobotics.com"]))
        addresses = {str(e) for e in intel.contact_points.emails}
        assert "ceo@acmerobotics.com" not in addresses
        assert addresses == set(KNOWN_EMAILS)

    def test_keeps_regex_found_emails_even_if_llm_missed_them(self) -> None:
        intel = build(llm_payload(emails=[]))
        assert len(intel.contact_points.emails) == 2

    def test_keeps_a_verified_linkedin_url(self) -> None:
        intel = build(
            llm_payload(
                leadership=[
                    LlmTeamMember(
                        name="Dana Whitfield",
                        role="CEO",
                        linkedin_url=KNOWN_LINKEDINS[0],
                    )
                ]
            )
        )
        assert str(intel.leadership[0].linkedin_url) == KNOWN_LINKEDINS[0]
        assert intel.leadership[0].source == "website"

    def test_strips_an_unverified_linkedin_url_but_keeps_the_person(self) -> None:
        intel = build(
            llm_payload(
                leadership=[
                    LlmTeamMember(
                        name="Marcus Oyelaran",
                        role="CTO",
                        linkedin_url="https://linkedin.com/in/made-up-profile",
                    )
                ]
            )
        )
        assert intel.leadership[0].name == "Marcus Oyelaran"
        assert intel.leadership[0].linkedin_url is None
        # Not yet searched: _build_intel runs before the external lookup.
        assert intel.leadership[0].source == "not_searched"

    def test_skips_nameless_entries(self) -> None:
        intel = build(llm_payload(leadership=[LlmTeamMember(name="   ", role="CEO")]))
        assert intel.leadership == []

    def test_records_the_llm_self_score_separately(self) -> None:
        intel = build(llm_payload(data_confidence_score=0.8))
        assert intel.llm_self_score == 0.8

    def test_clamps_an_out_of_range_score(self) -> None:
        intel = build(llm_payload(data_confidence_score=4.2))
        assert intel.data_confidence_score == 1.0

    def test_rejects_a_malformed_contact_page_url(self) -> None:
        intel = build(llm_payload(), contact_page="not a url")
        assert intel.contact_points.contact_page_url is None

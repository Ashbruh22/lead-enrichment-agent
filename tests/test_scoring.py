"""Tests for confidence scoring and the LinkedIn name-match guard."""

from lead_agent.models import CompanyIntel, ContactPoints, TeamMember
from lead_agent.scoring import blended_confidence, completeness
from lead_agent.search import looks_blocked, name_tokens, profile_matches_name


def make_intel(
    *,
    overview: str = "A" * 100,
    audience: str = "B" * 40,
    emails: list[str] | None = None,
    leadership: list[TeamMember] | None = None,
    llm_score: float = 0.5,
) -> CompanyIntel:
    return CompanyIntel(
        company_overview=overview,
        target_audience=audience,
        contact_points=ContactPoints(emails=emails or []),
        leadership=leadership or [],
        data_confidence_score=llm_score,
    )


class TestCompleteness:
    def test_empty_extraction_scores_low(self) -> None:
        intel = make_intel(overview="", audience="")
        assert completeness(intel, 0, 0) < 0.2

    def test_full_extraction_scores_high(self) -> None:
        intel = make_intel(
            emails=["a@b.com", "c@b.com"],
            leadership=[
                TeamMember(name="Dana Whitfield", linkedin_url="https://linkedin.com/in/dana")
            ],
        )
        assert completeness(intel, 5, 5) > 0.85

    def test_more_data_never_scores_lower(self) -> None:
        thin = make_intel()
        rich = make_intel(emails=["a@b.com"], leadership=[TeamMember(name="Dana")])
        assert completeness(rich, 3, 3) > completeness(thin, 3, 3)

    def test_people_with_linkedin_beat_people_without(self) -> None:
        without = make_intel(leadership=[TeamMember(name="Dana")])
        with_url = make_intel(
            leadership=[TeamMember(name="Dana", linkedin_url="https://linkedin.com/in/dana")]
        )
        assert completeness(with_url, 3, 3) > completeness(without, 3, 3)

    def test_stays_in_range(self) -> None:
        intel = make_intel(
            emails=["a@b.com", "c@b.com", "d@b.com"],
            leadership=[
                TeamMember(name=f"P{i}", linkedin_url="https://linkedin.com/in/p") for i in range(9)
            ],
        )
        assert 0.0 <= completeness(intel, 9, 9) <= 1.0


class TestBlendedConfidence:
    def test_deterministic_half_dominates_an_optimistic_model(self) -> None:
        # Model claims near-certainty about an extraction that found nothing.
        intel = make_intel(overview="", audience="", llm_score=1.0)
        blended, objective = blended_confidence(intel, 0, 0)
        assert blended < 0.5
        assert objective < blended  # the LLM pulls it up, but only so far

    def test_agreement_keeps_the_score_high(self) -> None:
        intel = make_intel(
            emails=["a@b.com", "c@b.com"],
            leadership=[TeamMember(name="Dana", linkedin_url="https://linkedin.com/in/dana")],
            llm_score=0.9,
        )
        blended, _ = blended_confidence(intel, 5, 5)
        assert blended > 0.85

    def test_result_is_bounded(self) -> None:
        intel = make_intel(llm_score=1.0)
        blended, _ = blended_confidence(intel, 5, 5)
        assert 0.0 <= blended <= 1.0


class TestNameTokens:
    def test_drops_honorifics_and_suffixes(self) -> None:
        assert name_tokens("Dr. Dana Whitfield Jr.") == ["dana", "whitfield"]

    def test_ignores_punctuation(self) -> None:
        assert name_tokens("Marcus O'Reilly-Smith") == ["marcus", "reilly", "smith"]


class TestProfileMatchesName:
    def test_accepts_matching_slug_with_hash_suffix(self) -> None:
        assert profile_matches_name(
            "https://www.linkedin.com/in/dana-whitfield-7b32a1", "Dana Whitfield"
        )

    def test_accepts_concatenated_slug(self) -> None:
        assert profile_matches_name("https://linkedin.com/in/marcusoyelaran", "Marcus Oyelaran")

    def test_rejects_a_different_person(self) -> None:
        # This is the guard that stops a stranger's profile being attached.
        assert not profile_matches_name(
            "https://www.linkedin.com/in/someone-else-1234", "Dana Whitfield"
        )

    def test_rejects_surname_only_match(self) -> None:
        assert not profile_matches_name("https://linkedin.com/in/bob-whitfield", "Dana Whitfield")

    def test_rejects_empty_name(self) -> None:
        assert not profile_matches_name("https://linkedin.com/in/dana-whitfield", "")

    def test_handles_url_encoded_slugs(self) -> None:
        assert profile_matches_name("https://linkedin.com/in/dana%2Dwhitfield", "Dana Whitfield")


class TestLooksBlocked:
    """A blocked engine must be recognised, or it costs ~10s per person."""

    def test_short_error_stub_is_blocked(self) -> None:
        # The real DuckDuckGo refusal: a 273-byte body with no error wording.
        assert looks_blocked("<html><body>If this persists, please email us.</body></html>")

    def test_challenge_title_is_blocked_even_when_long(self) -> None:
        html = "<html><head><title>Unusual traffic detected</title></head><body>"
        assert looks_blocked(html + "x" * 5_000 + "</body></html>")

    def test_a_real_result_page_is_not_blocked(self) -> None:
        html = "<html><head><title>jane doe - Brave Search</title></head><body>"
        assert not looks_blocked(html + "result " * 1_000 + "</body></html>")

    def test_challenge_words_in_the_body_do_not_block(self) -> None:
        # Regression: Brave ships an i18n table containing "Switch to
        # traditional captcha". Matching the body marked every successful
        # search as blocked and silently disabled the whole feature.
        html = (
            "<html><head><title>abhinav asthana - Brave Search</title></head><body>"
            + "x" * 5_000
            + '{"Switch to traditional captcha":"Switch to traditional CAPTCHA"}'
            + "</body></html>"
        )
        assert not looks_blocked(html)

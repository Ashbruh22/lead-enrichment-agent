"""Tests for link normalisation, scoring and candidate extraction."""

from lead_agent.discovery import (
    extract_candidates,
    normalise,
    registrable,
    score_link,
    top_urls,
)

BASE = "https://acmerobotics.com/about"


class TestNormalise:
    def test_strips_fragment_and_query(self) -> None:
        assert normalise("https://a.com/x?utm=1#top") == "https://a.com/x"

    def test_strips_trailing_slash(self) -> None:
        assert normalise("https://a.com/about/") == "https://a.com/about"

    def test_folds_www(self) -> None:
        assert normalise("https://www.a.com/about") == normalise("https://a.com/about")

    def test_root_stays_a_slash(self) -> None:
        assert normalise("https://a.com") == "https://a.com/"


class TestRegistrable:
    def test_reduces_subdomains(self) -> None:
        assert registrable("docs.vapi.ai") == "vapi.ai"
        assert registrable("www.postman.com") == "postman.com"

    def test_passes_through_bare_host(self) -> None:
        assert registrable("localhost") == "localhost"


class TestScoreLink:
    def test_about_beats_docs(self) -> None:
        assert score_link("https://a.com/about", "About") > score_link("https://a.com/docs", "Docs")

    def test_legal_pages_score_negative(self) -> None:
        assert score_link("https://a.com/legal/privacy", "Privacy") < 0

    def test_dated_article_urls_are_penalised(self) -> None:
        dated = score_link("https://a.com/2024/03/post", "A post")
        assert dated < score_link("https://a.com/post", "A post")

    def test_deep_paths_are_penalised(self) -> None:
        shallow = score_link("https://a.com/team", "Team")
        deep = score_link("https://a.com/x/y/z/team", "Team")
        assert deep < shallow

    def test_anchor_text_counts_when_slug_is_opaque(self) -> None:
        assert score_link("https://a.com/p/2931", "Meet the team") > 0


class TestExtractCandidates:
    def test_finds_internal_links_only(self, company_html: str) -> None:
        urls = [c.url for c in extract_candidates(company_html, BASE)]
        assert any("/company/careers" in u for u in urls)
        assert not any("twitter.com" in u for u in urls)
        assert not any("linkedin.com" in u for u in urls)

    def test_skips_anchors_mailto_and_assets(self, company_html: str) -> None:
        urls = [c.url for c in extract_candidates(company_html, BASE)]
        assert not any(u.endswith(".pdf") for u in urls)
        assert not any("#top" in u for u in urls)
        assert not any(u.startswith("mailto") for u in urls)

    def test_deduplicates_www_variants(self, company_html: str) -> None:
        urls = [c.url for c in extract_candidates(company_html, BASE)]
        assert len(urls) == len(set(urls))
        assert sum(1 for u in urls if u.endswith("/about")) <= 1

    def test_honours_exclude_set(self, company_html: str) -> None:
        excluded = "https://acmerobotics.com/pricing"
        urls = [c.url for c in extract_candidates(company_html, BASE, exclude={excluded})]
        assert excluded not in urls

    def test_respects_limit(self, company_html: str) -> None:
        assert len(extract_candidates(company_html, BASE, limit=2)) == 2

    def test_ranks_best_first(self, company_html: str) -> None:
        scores = [c.score for c in extract_candidates(company_html, BASE)]
        assert scores == sorted(scores, reverse=True)

    def test_empty_html_yields_nothing(self) -> None:
        assert extract_candidates("", BASE) == []


class TestTopUrls:
    def test_drops_negatively_scored_links(self, company_html: str) -> None:
        candidates = extract_candidates(company_html, BASE)
        chosen = top_urls(candidates, 10)
        assert not any("privacy" in u for u in chosen)

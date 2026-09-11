"""Tests for HTML cleaning, contact harvesting and token budgeting."""

from lead_agent.extract import (
    build_corpus,
    clean_html,
    find_emails,
    find_linkedin_urls,
    page_title,
    truncate,
)


class TestCleanHtml:
    def test_keeps_real_prose(self, company_html: str) -> None:
        text = clean_html(company_html)
        assert "autonomous mobile robots" in text
        assert "Dana Whitfield" in text

    def test_strips_scripts_and_styles(self, company_html: str) -> None:
        text = clean_html(company_html)
        assert "window.analytics" not in text
        assert "color: red" not in text
        assert "<svg" not in text and "M0 0h24v24H0z" not in text

    def test_handles_empty_input(self) -> None:
        assert clean_html("") == ""

    def test_survives_malformed_html(self) -> None:
        assert "Hello" in clean_html("<div><p>Hello<div><span>unclosed")

    def test_collapses_runaway_whitespace(self) -> None:
        text = clean_html("<body><p>one</p>\n\n\n\n\n<p>two</p></body>")
        assert "\n\n\n" not in text


class TestPageTitle:
    def test_reads_title(self, company_html: str) -> None:
        assert page_title(company_html).startswith("About Acme Robotics")

    def test_missing_title_is_empty(self) -> None:
        assert page_title("<html><body>hi</body></html>") == ""


class TestFindEmails:
    def test_finds_all_public_addresses(self, company_html: str) -> None:
        emails = find_emails(company_html)
        assert set(emails) == {
            "hello@acmerobotics.com",
            "sales@acmerobotics.com",
            "support@acmerobotics.com",
        }

    def test_rejects_retina_image_names(self) -> None:
        # `logo@2x.png` matches the e-mail shape but is an asset filename.
        assert find_emails('<img src="logo@2x.png">') == []

    def test_rejects_placeholder_domains(self) -> None:
        assert find_emails("mail us at you@example.com") == []

    def test_rejects_asset_hashes(self) -> None:
        assert find_emails("a3f9c21e8b7d4a6f9c2e@cdn.net") == []

    def test_deduplicates_case_insensitively(self) -> None:
        assert find_emails("Hi@Acme.com and hi@acme.com") == ["hi@acme.com"]


class TestFindLinkedIn:
    def test_splits_profiles_from_company_pages(self, company_html: str) -> None:
        found = find_linkedin_urls(company_html)
        assert "https://www.linkedin.com/in/dana-whitfield-7b32a1" in found["profiles"]
        assert "https://linkedin.com/in/marcusoyelaran" in found["profiles"]
        assert found["companies"] == ["https://www.linkedin.com/company/acme-robotics"]

    def test_normalises_scheme_and_trailing_slash(self) -> None:
        found = find_linkedin_urls("http://linkedin.com/in/someone/")
        assert found["profiles"] == ["https://linkedin.com/in/someone"]


class TestTruncate:
    def test_leaves_short_text_alone(self) -> None:
        assert truncate("short", 100) == "short"

    def test_marks_truncation(self) -> None:
        result = truncate("word " * 500, 100)
        assert result.endswith("[truncated]")
        assert len(result) <= 100 + len("\n\n[truncated]")

    def test_prefers_paragraph_boundary(self) -> None:
        text = "a" * 70 + "\n\n" + "b" * 200
        assert truncate(text, 100).startswith("a" * 70)
        assert "b" not in truncate(text, 100)


class TestBuildCorpus:
    def test_labels_each_source(self) -> None:
        corpus = build_corpus([("https://a.com", "alpha"), ("https://b.com", "beta")], 100, 1000)
        assert "## SOURCE: https://a.com" in corpus
        assert "## SOURCE: https://b.com" in corpus

    def test_respects_total_budget(self) -> None:
        pages = [(f"https://x.com/{i}", "z" * 5_000) for i in range(10)]
        assert len(build_corpus(pages, 2_000, 5_000)) < 6_000

    def test_skips_empty_pages(self) -> None:
        corpus = build_corpus([("https://a.com", ""), ("https://b.com", "real")], 100, 1000)
        assert "a.com" not in corpus and "b.com" in corpus

    def test_empty_input_is_empty_output(self) -> None:
        assert build_corpus([], 100, 1000) == ""

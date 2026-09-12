"""Tests for LLM page selection, including the follow-up round.

The guard that matters here is that a URL the model invented is never fetched.
It is enforced by intersecting the model's answer with the candidate list, so
these tests drive that intersection directly with a stub client rather than
going near the network.
"""

from lead_agent.discovery import Candidate
from lead_agent.gaps import GAP_EMAILS, GAP_LEADERSHIP
from lead_agent.llm import UsageTracker
from lead_agent.models import SelectedPages
from lead_agent.navigator import select_followup_pages, select_pages


class StubClient:
    """Stands in for GeminiClient, returning a canned selection."""

    def __init__(self, result: SelectedPages | None) -> None:
        self.result = result
        self.calls = 0
        self.last_prompt = ""

    async def structured(self, *, prompt: str, usage: UsageTracker, **_: object):
        self.calls += 1
        self.last_prompt = prompt
        usage.calls += 1
        return self.result


def candidates() -> list[Candidate]:
    return [
        Candidate(url="https://x.com/about", anchor="About", score=10),
        Candidate(url="https://x.com/team", anchor="Team", score=10),
        Candidate(url="https://x.com/contact", anchor="Contact", score=8),
    ]


def tracker() -> UsageTracker:
    return UsageTracker(model="gemini-3.6-flash")


class TestSelectPages:
    async def test_keeps_urls_that_exist(self) -> None:
        client = StubClient(SelectedPages(urls=["https://x.com/about"], reasoning="about page"))
        urls, reason = await select_pages(
            client, domain="x.com", homepage_url="https://x.com", homepage_text="hi",
            candidates=candidates(), max_pages=3, usage=tracker(),
        )
        assert urls == ["https://x.com/about"]
        assert reason == "about page"

    async def test_discards_an_invented_url(self) -> None:
        client = StubClient(
            SelectedPages(urls=["https://x.com/about", "https://x.com/made-up"], reasoning="r")
        )
        urls, reason = await select_pages(
            client, domain="x.com", homepage_url="https://x.com", homepage_text="hi",
            candidates=candidates(), max_pages=3, usage=tracker(),
        )
        assert urls == ["https://x.com/about"]
        assert "1 invented URL(s) discarded" in reason

    async def test_falls_back_to_heuristics_when_the_call_fails(self) -> None:
        urls, reason = await select_pages(
            StubClient(None), domain="x.com", homepage_url="https://x.com", homepage_text="hi",
            candidates=candidates(), max_pages=2, usage=tracker(),
        )
        assert urls == ["https://x.com/about", "https://x.com/team"]
        assert "heuristic" in reason

    async def test_respects_the_page_budget(self) -> None:
        client = StubClient(
            SelectedPages(
                urls=["https://x.com/about", "https://x.com/team", "https://x.com/contact"],
                reasoning="r",
            )
        )
        urls, _ = await select_pages(
            client, domain="x.com", homepage_url="https://x.com", homepage_text="hi",
            candidates=candidates(), max_pages=2, usage=tracker(),
        )
        assert len(urls) == 2


class TestSelectFollowupPages:
    async def test_asks_for_pages_that_match_the_gap(self) -> None:
        client = StubClient(SelectedPages(urls=["https://x.com/team"], reasoning="team page"))
        urls, reason = await select_followup_pages(
            client, domain="x.com", gaps=[GAP_LEADERSHIP], visited=["https://x.com"],
            candidates=candidates(), max_pages=2, usage=tracker(),
        )
        assert urls == ["https://x.com/team"]
        assert reason == "team page"
        # The gap description has to reach the model, or it cannot target it.
        assert "founders" in client.last_prompt

    async def test_no_gaps_means_no_call_at_all(self) -> None:
        client = StubClient(SelectedPages(urls=["https://x.com/team"], reasoning="r"))
        urls, _ = await select_followup_pages(
            client, domain="x.com", gaps=[], visited=[], candidates=candidates(),
            max_pages=2, usage=tracker(),
        )
        assert urls == []
        assert client.calls == 0

    async def test_no_unread_candidates_means_no_call(self) -> None:
        client = StubClient(SelectedPages(urls=[], reasoning="r"))
        urls, _ = await select_followup_pages(
            client, domain="x.com", gaps=[GAP_EMAILS], visited=[], candidates=[],
            max_pages=2, usage=tracker(),
        )
        assert urls == []
        assert client.calls == 0

    async def test_declining_to_choose_stops_the_loop(self) -> None:
        # Unlike the first round there is no heuristic fallback: returning
        # nothing must end the loop, not fetch the next-best link anyway.
        client = StubClient(SelectedPages(urls=[], reasoning="nothing relevant"))
        urls, reason = await select_followup_pages(
            client, domain="x.com", gaps=[GAP_LEADERSHIP], visited=[],
            candidates=candidates(), max_pages=2, usage=tracker(),
        )
        assert urls == []
        assert reason == "nothing left worth reading"

    async def test_a_failed_call_stops_the_loop(self) -> None:
        urls, reason = await select_followup_pages(
            StubClient(None), domain="x.com", gaps=[GAP_LEADERSHIP], visited=[],
            candidates=candidates(), max_pages=2, usage=tracker(),
        )
        assert urls == []
        assert reason == "follow-up selection failed"

    async def test_invented_urls_are_discarded_here_too(self) -> None:
        client = StubClient(SelectedPages(urls=["https://x.com/leadership"], reasoning="r"))
        urls, _ = await select_followup_pages(
            client, domain="x.com", gaps=[GAP_LEADERSHIP], visited=[],
            candidates=candidates(), max_pages=2, usage=tracker(),
        )
        assert urls == []

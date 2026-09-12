"""Tests for provider ordering, caching and the name guard.

No network. Providers are a Protocol precisely so the finder can be driven with
stubs, and the behaviour worth pinning is all in the finder: which provider is
asked first, what retires one, and that a miss is never paid for twice.
"""

from lead_agent.config import Settings
from lead_agent.search import (
    BrowserProvider,
    LinkedInFinder,
    Lookup,
    TavilyProvider,
    profile_urls_in,
)

PROFILE = "https://www.linkedin.com/in/jane-doe"
OTHER = "https://www.linkedin.com/in/someone-else"


class StubProvider:
    """Returns canned results and counts how often it was asked."""

    def __init__(self, name: str, results: list[str] | None) -> None:
        self.name = name
        self.results = results
        self.calls = 0

    async def profile_urls(self, query: str) -> list[str] | None:
        self.calls += 1
        return self.results


def finder(*providers: StubProvider, tavily: str = "") -> LinkedInFinder:
    settings = Settings(gemini_api_key="x", tavily_api_key=tavily)
    return LinkedInFinder(settings=settings, providers=list(providers))


class TestProviderOrder:
    def test_tavily_comes_first_when_configured(self) -> None:
        built = LinkedInFinder.build(Settings(gemini_api_key="x", tavily_api_key="tvly-abc"))
        assert isinstance(built.providers[0], TavilyProvider)
        assert isinstance(built.providers[1], BrowserProvider)

    def test_without_a_key_only_browser_providers_are_built(self) -> None:
        built = LinkedInFinder.build(Settings(gemini_api_key="x", tavily_api_key=""))
        assert built.providers
        assert not any(isinstance(p, TavilyProvider) for p in built.providers)

    async def test_a_later_provider_is_not_asked_after_a_hit(self) -> None:
        first, second = StubProvider("first", [PROFILE]), StubProvider("second", [PROFILE])
        result = await finder(first, second).find("Jane Doe", "acme")
        assert result == Lookup(url=PROFILE, status="found", provider="first")
        assert second.calls == 0

    async def test_falls_through_to_the_next_provider(self) -> None:
        first, second = StubProvider("first", []), StubProvider("second", [PROFILE])
        result = await finder(first, second).find("Jane Doe", "acme")
        assert result.status == "found"
        assert result.provider == "second"


class TestRetiring:
    async def test_a_failed_provider_is_retired_for_the_run(self) -> None:
        broken, working = StubProvider("broken", None), StubProvider("working", [PROFILE])
        f = finder(broken, working)

        await f.find("Jane Doe", "acme")
        await f.find("John Roe", "acme")

        assert broken.calls == 1, "a retired provider must not be asked again"
        assert [p.name for p in f.live_providers] == ["working"]

    async def test_all_providers_failing_reports_unavailable(self) -> None:
        result = await finder(StubProvider("a", None), StubProvider("b", None)).find(
            "Jane Doe", "acme"
        )
        assert result.status == "unavailable"
        assert result.url is None

    async def test_no_providers_at_all_is_unavailable_not_not_found(self) -> None:
        # "We could not look" and "we looked and found nothing" are different
        # claims, and the output reports them differently.
        assert (await finder().find("Jane Doe", "acme")).status == "unavailable"

    async def test_a_working_provider_returning_nothing_is_not_found(self) -> None:
        result = await finder(StubProvider("a", [])).find("Jane Doe", "acme")
        assert result.status == "not_found"


class TestCaching:
    async def test_a_hit_is_cached(self) -> None:
        provider = StubProvider("p", [PROFILE])
        f = finder(provider)
        assert (await f.find("Jane Doe", "acme")).url == PROFILE
        assert (await f.find("Jane Doe", "acme")).url == PROFILE
        assert provider.calls == 1

    async def test_a_miss_is_cached_too(self) -> None:
        # Failure is the common case; re-paying for it per domain is what made
        # the original implementation slow.
        provider = StubProvider("p", [])
        f = finder(provider)
        await f.find("Jane Doe", "acme")
        await f.find("Jane Doe", "acme")
        assert provider.calls == 1

    async def test_the_cache_is_keyed_on_person_and_company(self) -> None:
        provider = StubProvider("p", [])
        f = finder(provider)
        await f.find("Jane Doe", "acme")
        await f.find("Jane Doe", "other")
        assert provider.calls == 2

    async def test_the_cache_ignores_case_and_padding(self) -> None:
        provider = StubProvider("p", [])
        f = finder(provider)
        await f.find("Jane Doe", "acme")
        await f.find("  jane doe ", "ACME")
        assert provider.calls == 1


class TestNameGuard:
    async def test_a_stranger_is_rejected_even_when_returned_first(self) -> None:
        result = await finder(StubProvider("p", [OTHER, PROFILE])).find("Jane Doe", "acme")
        assert result.url == PROFILE

    async def test_only_a_stranger_means_not_found(self) -> None:
        result = await finder(StubProvider("p", [OTHER])).find("Jane Doe", "acme")
        assert result.status == "not_found"
        assert result.url is None

    async def test_a_blank_name_is_never_searched(self) -> None:
        provider = StubProvider("p", [PROFILE])
        assert (await finder(provider).find("   ", "acme")).status == "not_found"
        assert provider.calls == 0


class TestProfileUrlsIn:
    def test_extracts_and_deduplicates_profiles(self) -> None:
        html = f'<a href="{PROFILE}">x</a><a href="{PROFILE}/">y</a>'
        assert profile_urls_in(html) == [PROFILE]

    def test_ignores_company_pages(self) -> None:
        assert profile_urls_in('<a href="https://linkedin.com/company/acme">x</a>') == []

    def test_upgrades_http_to_https(self) -> None:
        assert profile_urls_in('<a href="http://linkedin.com/in/jane-doe">x</a>') == [
            "https://linkedin.com/in/jane-doe"
        ]

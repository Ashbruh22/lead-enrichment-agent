"""External LinkedIn lookup for people whose profile is not on the website.

Most company sites name their founders but never link their LinkedIn profiles,
so this closes the most common gap in the output.

The layout reflects one fact: **unauthenticated search scraping is unreliable
by design.** Brave and DuckDuckGo are actively trying to stop automated
queries, and any cleverness added here is a rate limiter's next target. So
providers are ordered by how much we can trust them -- Tavily's API first when
a key is configured, browser scraping only as a fallback -- and a provider that
stops cooperating is dropped for the whole run rather than re-probed per
person.

Two safeguards matter more than the providers themselves:

* :func:`profile_matches_name` -- a search engine will happily return *a*
  LinkedIn profile for any query. A result is accepted only when the profile
  slug corresponds to the person's name. Without this the feature would quietly
  attach strangers to your leads, which is worse than finding nothing. The
  guard lives in the finder, never in a provider, so every path is checked by
  the same code.
* Negative caching -- "we looked for this person and found nothing" is cached
  as firmly as a hit. Failure is the common case and re-paying for it per
  domain is what made the original implementation slow.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol
from urllib.parse import quote_plus, unquote, urlparse

import httpx
from playwright.async_api import BrowserContext
from playwright.async_api import Error as PlaywrightError

from .config import Settings
from .extract import LINKEDIN_RE, page_title

log = logging.getLogger(__name__)

_NAME_NOISE = {"dr", "mr", "ms", "mrs", "jr", "sr", "ii", "iii", "phd", "md"}

# Browser engines, tried in this order. Brave is first because it is the one
# that still serves a real result page to an automated browser; DuckDuckGo's
# endpoints now answer most automated queries with a short error stub, so they
# sit behind it as backups -- which engine is blocked varies by network.
BROWSER_ENGINES: tuple[tuple[str, str], ...] = (
    ("brave", "https://search.brave.com/search?q={query}"),
    ("duckduckgo", "https://html.duckduckgo.com/html/?q={query}"),
    ("duckduckgo-lite", "https://lite.duckduckgo.com/lite/?q={query}"),
)

# A real result page is tens of kilobytes. Anything this small is an error stub
# or a challenge page, whatever it claims in the body.
_MIN_RESULT_PAGE_CHARS = 2_000
_BLOCK_TITLE_RE = re.compile(
    r"unusual traffic|anomaly|are you a robot|captcha|access denied|"
    r"just a moment|verify you are human|too many requests",
    re.IGNORECASE,
)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


# --------------------------------------------------------------------------- #
# Name matching
# --------------------------------------------------------------------------- #


def name_tokens(name: str) -> list[str]:
    """Lowercase alphabetic name parts worth matching on."""
    parts = re.findall(r"[A-Za-z]{2,}", name.lower())
    return [p for p in parts if p not in _NAME_NOISE]


def profile_matches_name(url: str, name: str) -> bool:
    """True if a ``linkedin.com/in/<slug>`` URL plausibly belongs to ``name``.

    Requires the surname, plus the first name when the person has one, to
    appear in the slug. ``/in/jane-doe-8a41b2`` matches "Jane Doe"; a random
    profile does not.
    """
    tokens = name_tokens(name)
    if not tokens:
        return False
    slug = unquote(urlparse(url).path.rsplit("/", 1)[-1]).lower()
    if not slug:
        return False
    surname = tokens[-1]
    if surname not in slug:
        return False
    return len(tokens) < 2 or tokens[0] in slug


def looks_blocked(html: str) -> bool:
    """True if a search response is a challenge or error stub rather than results.

    Two signals, and only two. Size catches the error stubs, which is how
    DuckDuckGo now refuses automated queries -- a couple of hundred bytes with
    no explanation in the body.

    Challenge wording is matched against the ``<title>`` alone, never the body.
    A search results page is a quarter-megabyte of bundled JavaScript that
    routinely mentions these words in passing: Brave ships a translation table
    containing "Switch to traditional captcha", which is emphatically not the
    same as being served one. Scanning the body for "captcha" marked every
    successful Brave search as blocked.
    """
    if len(html) < _MIN_RESULT_PAGE_CHARS:
        return True
    return bool(_BLOCK_TITLE_RE.search(page_title(html)))


def profile_urls_in(html: str) -> list[str]:
    """Every ``linkedin.com/in/`` URL in a blob, normalised and deduplicated."""
    found: dict[str, None] = {}
    for match in LINKEDIN_RE.finditer(html):
        if match.group(1).lower() != "in":
            continue
        url = unquote(match.group(0)).rstrip("/.,)\"'").replace("http://", "https://")
        found.setdefault(url, None)
    return list(found)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class SearchProvider(Protocol):
    """A way of turning a query into candidate LinkedIn profile URLs.

    ``profile_urls`` returns ``None`` to mean *the provider itself failed* --
    blocked, rate limited, misconfigured -- which retires it for the run. An
    empty list means the provider worked and genuinely found nothing, which
    says nothing about its health.
    """

    name: str

    async def profile_urls(self, query: str) -> list[str] | None: ...


@dataclass
class TavilyProvider:
    """Tavily's search API. The reliable path, and the reason to configure a key."""

    api_key: str
    name: str = "tavily"
    max_results: int = 10
    timeout: float = 20.0

    async def profile_urls(self, query: str) -> list[str] | None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    TAVILY_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "query": query,
                        "max_results": self.max_results,
                        "include_domains": ["linkedin.com"],
                    },
                )
        except Exception as exc:
            log.debug("tavily request failed: %s", exc)
            return None

        if response.status_code != 200:
            log.info("tavily returned HTTP %s; retiring it for this run", response.status_code)
            return None

        try:
            results = response.json().get("results", [])
        except ValueError:
            return None

        # Tavily returns posts and articles alongside profiles; keep only the
        # profile URLs and let the finder's name guard judge them.
        urls: list[str] = []
        for item in results:
            url = (item.get("url") or "").rstrip("/")
            if "/in/" in url:
                urls.append(url.replace("http://", "https://"))
        return urls


@dataclass
class BrowserProvider:
    """A search engine driven through the Playwright browser already running."""

    name: str
    url_template: str
    context: BrowserContext | None = None
    timeout_ms: int = 15_000

    async def profile_urls(self, query: str) -> list[str] | None:
        if self.context is None:
            return None
        page = None
        try:
            page = await self.context.new_page()
            await page.goto(
                self.url_template.format(query=quote_plus(query)),
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
            html = await page.content()
            if looks_blocked(html):
                log.info("%s is not serving results; retiring it for this run", self.name)
                return None
            return profile_urls_in(html)
        except PlaywrightError as exc:
            # A timeout or navigation failure is the engine refusing us just as
            # surely as a challenge page is.
            log.debug("%s search failed: %s", self.name, exc)
            return None
        finally:
            if page is not None:
                try:
                    await page.close()
                except PlaywrightError:
                    pass


# --------------------------------------------------------------------------- #
# The finder
# --------------------------------------------------------------------------- #

LookupStatus = Literal["found", "not_found", "unavailable"]


@dataclass(frozen=True)
class Lookup:
    """Outcome of one profile search, including why it came back empty."""

    url: str | None = None
    status: LookupStatus = "not_found"
    provider: str | None = None


@dataclass
class LinkedInFinder:
    """Looks up missing LinkedIn profile URLs across a whole run.

    One instance per run, not per domain: the cache and the set of retired
    providers are only useful if they outlive a single site. :meth:`bind` swaps
    in each domain's browser context as the crawl moves on.
    """

    settings: Settings
    providers: list[SearchProvider] = field(default_factory=list)
    _cache: dict[tuple[str, str], Lookup] = field(default_factory=dict, repr=False)
    _retired: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def build(cls, settings: Settings, context: BrowserContext | None = None) -> LinkedInFinder:
        """Providers in order of trustworthiness: API first, scraping second."""
        providers: list[SearchProvider] = []
        if settings.tavily_api_key:
            providers.append(
                TavilyProvider(
                    api_key=settings.tavily_api_key,
                    max_results=settings.tavily_max_results,
                )
            )
        providers.extend(
            BrowserProvider(name=name, url_template=template, context=context)
            for name, template in BROWSER_ENGINES
        )
        return cls(settings=settings, providers=providers)

    def bind(self, context: BrowserContext | None) -> None:
        """Point the browser-backed providers at the current domain's context."""
        for provider in self.providers:
            if isinstance(provider, BrowserProvider):
                provider.context = context

    @property
    def live_providers(self) -> list[SearchProvider]:
        return [p for p in self.providers if p.name not in self._retired]

    async def find(self, name: str, company: str) -> Lookup:
        """Best-effort profile for ``name`` at ``company``.

        Returns a :class:`Lookup` rather than a bare URL so the caller can tell
        "searched and found nothing" from "had nothing to search with" -- a
        distinction that ends up in the output as provenance.
        """
        if not name.strip():
            return Lookup(status="not_found")

        key = (name.strip().lower(), company.strip().lower())
        if key in self._cache:
            return self._cache[key]

        query = f'"{name}" {company} linkedin'
        result = await self._search(query, name)
        self._cache[key] = result  # negatives cached too; failure is the common case
        return result

    async def _search(self, query: str, name: str) -> Lookup:
        providers = self.live_providers
        if not providers:
            return Lookup(status="unavailable")

        for provider in providers:
            urls = await provider.profile_urls(query)
            if urls is None:
                self._retired.add(provider.name)
                continue
            for url in urls:
                if profile_matches_name(url, name):
                    return Lookup(url=url, status="found", provider=provider.name)

        # Every provider either retired or came back empty. If none survive,
        # the lookup was never really performed and should not be reported as
        # a negative result.
        return Lookup(status="not_found" if self.live_providers else "unavailable")

"""External LinkedIn lookup for people whose profile is not on the website.

Most company sites name their founders but never link their LinkedIn profiles,
so this closes the most common gap in the output.

It reuses the Playwright browser already running for the crawl, so it needs no
extra API key -- which is the point. Engines are tried in order and an engine
that stops cooperating is dropped for the rest of the run, because a blocked
engine costs ten seconds per person and returns nothing. A Tavily path is
available behind ``TAVILY_API_KEY`` as a final fallback.

The important safeguard is :func:`profile_matches_name`: a search engine will
happily return *a* LinkedIn profile for any query, so a result is only accepted
when the profile slug actually corresponds to the person's name. Without that
check this feature would quietly attach strangers to your leads.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus, unquote, urlparse

import httpx
from playwright.async_api import BrowserContext
from playwright.async_api import Error as PlaywrightError

from .config import USER_AGENT, Settings
from .extract import LINKEDIN_RE, page_title

log = logging.getLogger(__name__)

# Tried in order. Brave is first because it is the one that still serves a real
# result page to an automated browser; DuckDuckGo's endpoints now answer most
# automated queries with a short error stub, so they sit behind it as backups
# rather than being removed -- which engine is blocked varies by network.
SEARCH_ENGINES: tuple[tuple[str, str], ...] = (
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

_NAME_NOISE = {"dr", "mr", "ms", "mrs", "jr", "sr", "ii", "iii", "phd", "md"}


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


def _first_matching_profile(html: str, name: str) -> str | None:
    for match in LINKEDIN_RE.finditer(html):
        if match.group(1).lower() != "in":
            continue
        url = unquote(match.group(0)).rstrip("/.,)\"'").replace("http://", "https://")
        if profile_matches_name(url, name):
            return url
    return None


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


class LinkedInFinder:
    """Looks up missing LinkedIn profile URLs.

    One instance per domain. Engines that return a block page are dropped for
    the lifetime of the instance, so a blocked engine is paid for once rather
    than once per person.
    """

    def __init__(self, settings: Settings, context: BrowserContext | None) -> None:
        self.settings = settings
        self.context = context
        self._blocked: set[str] = set()

    @property
    def _live_engines(self) -> list[tuple[str, str]]:
        return [(n, t) for n, t in SEARCH_ENGINES if n not in self._blocked]

    async def find(self, name: str, company: str) -> str | None:
        """Best-effort profile URL for ``name`` at ``company``; ``None`` if unsure."""
        if not name.strip():
            return None
        query = f'"{name}" {company} linkedin'

        for engine, template in self._live_engines:
            url = await self._search_browser(engine, template, query, name)
            if url:
                return url

        if self.settings.tavily_api_key:
            return await self._search_tavily(query, name)
        return None

    async def _search_browser(
        self, engine: str, template: str, query: str, name: str
    ) -> str | None:
        if self.context is None:
            return None
        page = None
        try:
            page = await self.context.new_page()
            await page.goto(
                template.format(query=quote_plus(query)),
                wait_until="domcontentloaded",
                timeout=15_000,
            )
            html = await page.content()
            if looks_blocked(html):
                log.info("%s is not serving results; dropping it for this run", engine)
                self._blocked.add(engine)
                return None
            return _first_matching_profile(html, name)
        except PlaywrightError as exc:
            # A timeout or navigation failure is the engine refusing us just as
            # surely as a challenge page is.
            log.debug("%s search failed for %r: %s", engine, name, exc)
            self._blocked.add(engine)
            return None
        finally:
            if page is not None:
                try:
                    await page.close()
                except PlaywrightError:
                    pass

    async def _search_tavily(self, query: str, name: str) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.settings.tavily_api_key,
                        "query": query,
                        "max_results": 5,
                        "include_domains": ["linkedin.com"],
                    },
                    headers={"User-Agent": USER_AGENT},
                )
            if response.status_code != 200:
                return None
            for item in response.json().get("results", []):
                url = (item.get("url") or "").rstrip("/")
                if "/in/" in url and profile_matches_name(url, name):
                    return url
        except Exception as exc:
            log.debug("tavily search failed for %r: %s", name, exc)
        return None

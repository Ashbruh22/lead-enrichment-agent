"""External LinkedIn lookup for people whose profile is not on the website.

Most company sites name their founders but never link their LinkedIn profiles,
so this closes the most common gap in the output.

It reuses the Playwright browser already running for the crawl and queries
DuckDuckGo's HTML endpoint -- no extra API key, which is the point. A Tavily
path is available behind ``TAVILY_API_KEY`` for when DuckDuckGo rate-limits.

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
from .extract import LINKEDIN_RE

log = logging.getLogger(__name__)

_DDG_HTML = "https://html.duckduckgo.com/html/?q={query}"
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


class LinkedInFinder:
    """Looks up missing LinkedIn profile URLs."""

    def __init__(self, settings: Settings, context: BrowserContext | None) -> None:
        self.settings = settings
        self.context = context
        self._exhausted = False  # set once the search route stops cooperating

    async def find(self, name: str, company: str) -> str | None:
        """Best-effort profile URL for ``name`` at ``company``; ``None`` if unsure."""
        if self._exhausted or not name.strip():
            return None
        query = f'"{name}" {company} linkedin'

        url = await self._search_browser(query, name)
        if url:
            return url
        if self.settings.tavily_api_key:
            return await self._search_tavily(query, name)
        return None

    async def _search_browser(self, query: str, name: str) -> str | None:
        if self.context is None:
            return None
        page = None
        try:
            page = await self.context.new_page()
            await page.goto(
                _DDG_HTML.format(query=quote_plus(query)),
                wait_until="domcontentloaded",
                timeout=15_000,
            )
            html = await page.content()
            if "anomaly" in html.lower() or "unusual traffic" in html.lower():
                log.info("Search engine rate-limited; disabling LinkedIn lookup for this run")
                self._exhausted = True
                return None
            return _first_matching_profile(html, name)
        except PlaywrightError as exc:
            log.debug("browser search failed for %r: %s", name, exc)
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

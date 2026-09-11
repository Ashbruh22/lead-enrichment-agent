"""Headless browser layer.

One Chromium instance is shared by the whole run; each domain gets a fresh
``BrowserContext`` so cookies and storage never leak between targets.

Three things here matter for speed and for not falling over:

* **Resource blocking.** Images, fonts, media and stylesheets are aborted at
  the network layer. We only ever read the DOM, so downloading a 2 MB hero
  image is pure latency.
* **SPA settling.** We wait for ``domcontentloaded`` and then add a short,
  bounded ``networkidle`` wait. Sites like vapi.ai render their content with
  JS, but analytics beacons mean ``networkidle`` may never fire -- so that
  wait timing out is expected, not an error.
* **Graceful degradation.** Nav failures retry with backoff; bot-blocked or
  JS-hostile pages fall back to a plain HTTP GET before we give up.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

import httpx
from playwright.async_api import Browser, BrowserContext, async_playwright
from playwright.async_api import Error as PlaywrightError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import USER_AGENT, Settings
from .extract import clean_html, page_title
from .robots import RobotsCache

log = logging.getLogger(__name__)

_BLOCKED_RESOURCES = {"image", "font", "media", "stylesheet"}
_BLOCK_TITLE_RE = re.compile(
    r"just a moment|attention required|access denied|are you a robot|"
    r"security check|verify you are human|403 forbidden",
    re.IGNORECASE,
)


@dataclass
class FetchResult:
    """Outcome of fetching one URL. Never raises -- failure is a value."""

    url: str
    ok: bool = False
    html: str = ""
    text: str = ""
    title: str = ""
    status: int | None = None
    via: str = "playwright"
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def looks_blocked(title: str, text: str, threshold: int) -> bool:
    """Heuristic bot-wall / empty-render detection."""
    if _BLOCK_TITLE_RE.search(title or ""):
        return True
    return len(text.strip()) < threshold


class BrowserSession:
    """Async context manager owning the Playwright lifecycle."""

    def __init__(self, settings: Settings, headless: bool = True) -> None:
        self.settings = settings
        self.headless = headless
        self.robots = RobotsCache(enabled=settings.respect_robots)
        self._playwright = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> BrowserSession:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def new_context(self) -> BrowserContext:
        """A clean, resource-blocked context for a single domain."""
        assert self._browser is not None, "BrowserSession used outside its context manager"
        context = await self._browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            ignore_https_errors=True,
        )
        context.set_default_timeout(self.settings.page_timeout_ms)
        await context.route("**/*", self._block_heavy_resources)
        return context

    @staticmethod
    async def _block_heavy_resources(route, request) -> None:
        try:
            if request.resource_type in _BLOCKED_RESOURCES:
                await route.abort()
            else:
                await route.continue_()
        except PlaywrightError:  # context torn down mid-flight
            pass

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #

    async def fetch(self, context: BrowserContext, url: str) -> FetchResult:
        """Fetch one page. Returns a FetchResult even on total failure."""
        result = FetchResult(url=url)

        if not await self.robots.allowed(url):
            result.error = "disallowed by robots.txt"
            result.notes.append("skipped: robots.txt")
            return result

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.settings.fetch_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=6),
                retry=retry_if_exception_type((PlaywrightError, asyncio.TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    await self._render(context, url, result)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:180]}"
            log.debug("playwright failed for %s: %s", url, result.error)

        if not result.ok or looks_blocked(
            result.title, result.text, self.settings.blocked_text_threshold
        ):
            if result.ok:
                result.notes.append("render looked blocked/empty; retried over plain HTTP")
            await self._http_fallback(url, result)

        return result

    async def _render(self, context: BrowserContext, url: str, result: FetchResult) -> None:
        page = await context.new_page()
        try:
            response = await page.goto(
                url, wait_until="domcontentloaded", timeout=self.settings.page_timeout_ms
            )
            result.status = response.status if response else None

            if result.status and result.status >= 400:
                result.error = f"HTTP {result.status}"
                result.ok = False
                return

            # Bounded settle for client-rendered content; timing out here is fine.
            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=self.settings.settle_timeout_ms
                )
            except PlaywrightError:
                result.notes.append("network never went idle; captured DOM as-is")

            result.html = await page.content()
            result.title = page_title(result.html)
            result.text = clean_html(result.html)
            result.ok = bool(result.text)
            result.error = None
        finally:
            await page.close()

    async def _http_fallback(self, url: str, result: FetchResult) -> None:
        """Plain GET for pages the browser could not usefully render."""
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            ) as client:
                response = await client.get(url)
            text = clean_html(response.text)
            if response.status_code < 400 and len(text) > len(result.text):
                result.html = response.text
                result.text = text
                result.title = page_title(response.text)
                result.status = response.status_code
                result.via = "httpx-fallback"
                result.ok = True
                result.error = None
                result.notes.append("recovered via HTTP fallback")
            elif not result.ok:
                result.error = result.error or f"HTTP {response.status_code}"
        except Exception as exc:
            if not result.ok:
                result.error = result.error or f"{type(exc).__name__}: {str(exc)[:120]}"

    async def fetch_many(
        self, context: BrowserContext, urls: list[str], concurrency: int
    ) -> list[FetchResult]:
        """Fetch several pages with bounded concurrency and a politeness delay."""
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def worker(target: str) -> FetchResult:
            async with semaphore:
                await asyncio.sleep(self.settings.request_delay_seconds)
                return await self.fetch(context, target)

        return list(await asyncio.gather(*(worker(u) for u in urls)))

"""robots.txt awareness.

Crawling someone's public marketing site is ordinary behaviour, but honouring
robots.txt costs one cheap request per domain and is the right default. A
missing or unreachable robots.txt means "no restrictions stated" -- we allow.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from .config import USER_AGENT

log = logging.getLogger(__name__)


class RobotsCache:
    """One parsed robots.txt per origin, fetched at most once."""

    def __init__(self, enabled: bool = True, timeout: float = 8.0) -> None:
        self.enabled = enabled
        self.timeout = timeout
        self._cache: dict[str, RobotFileParser | None] = {}

    async def allowed(self, url: str) -> bool:
        if not self.enabled:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._cache:
            self._cache[origin] = await self._load(origin)
        parser = self._cache[origin]
        return True if parser is None else parser.can_fetch(USER_AGENT, url)

    async def _load(self, origin: str) -> RobotFileParser | None:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
            ) as client:
                response = await client.get(f"{origin}/robots.txt")
            if response.status_code != 200:
                return None
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            return parser
        except Exception as exc:  # unreachable robots.txt must never block a run
            log.debug("robots.txt unavailable for %s: %s", origin, exc)
            return None

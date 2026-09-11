"""Link discovery and heuristic scoring.

Produces the candidate list that the LLM navigator chooses from. Keeping this
step deterministic means the LLM only ever sees real, same-site, already-
normalised URLs -- it cannot send the crawler somewhere that does not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

# Slug/anchor keywords worth crawling, weighted by how much company
# intelligence such a page usually carries.
POSITIVE = {
    "about": 10, "team": 10, "leadership": 10, "founders": 10, "company": 9,
    "our-story": 8, "who-we-are": 8, "contact": 8, "management": 8,
    "people": 6, "careers": 4, "jobs": 3, "pricing": 6, "customers": 5,
    "solutions": 4, "product": 4, "platform": 4, "use-cases": 4, "why": 3,
}
# Pages that burn tokens without adding company intelligence.
NEGATIVE = {
    "blog": -8, "docs": -10, "documentation": -10, "changelog": -8, "guides": -6,
    "privacy": -10, "terms": -10, "legal": -10, "cookie": -10, "security": -4,
    "login": -10, "signin": -10, "signup": -10, "register": -10, "status": -8,
    "download": -6, "api-reference": -10, "learn": -4, "community": -4,
    "events": -5, "webinar": -5, "press": -2, "news": -4, "support": -1,
}
_SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4",
    ".dmg", ".exe", ".xml", ".rss", ".json", ".css", ".js",
)


@dataclass(frozen=True)
class Candidate:
    """A crawlable link plus why we think it is worth fetching."""

    url: str
    anchor: str
    score: int


def registrable(host: str) -> str:
    """Best-effort registrable domain (``www.blog.foo.com`` -> ``foo.com``)."""
    parts = host.lower().lstrip(".").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.lower()


def normalise(url: str) -> str:
    """Canonical form of a URL, so the same page is never fetched twice.

    Drops fragments, query strings and trailing slashes, and folds ``www.``
    away -- sites routinely link to both ``example.com/about`` and
    ``www.example.com/about``, which are one page and one token budget.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, host, path, "", "", ""))


def score_link(url: str, anchor: str) -> int:
    """Heuristic usefulness score for a candidate page."""
    slug = urlparse(url).path.lower()
    text = anchor.lower()[:80]
    depth = len([p for p in slug.split("/") if p])

    score = 0
    for keyword, weight in POSITIVE.items():
        if keyword in slug:
            score += weight
        elif keyword in text:
            score += weight // 2
    for keyword, weight in NEGATIVE.items():
        if keyword in slug or keyword in text:
            score += weight

    score -= max(0, depth - 1) * 2  # shallow pages are the informative ones
    if re.search(r"/\d{4}/\d{2}/", slug):  # dated article URLs
        score -= 6
    return score


def extract_candidates(
    html: str, base_url: str, limit: int = 25, exclude: set[str] | None = None
) -> list[Candidate]:
    """Same-site links from ``html``, scored and ranked best-first."""
    exclude = {normalise(u) for u in (exclude or set())}
    base_host = registrable(urlparse(base_url).netloc)
    soup = BeautifulSoup(html or "", "html.parser")

    seen: dict[str, Candidate] = {}
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        url = normalise(urljoin(base_url, href))
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        if registrable(parsed.netloc) != base_host:
            continue
        if parsed.path.lower().endswith(_SKIP_EXTENSIONS):
            continue
        if url in exclude or url in seen:
            continue

        anchor = tag.get_text(" ", strip=True)[:80]
        seen[url] = Candidate(url=url, anchor=anchor, score=score_link(url, anchor))

    ranked = sorted(seen.values(), key=lambda c: (-c.score, len(c.url)))
    return ranked[:limit]


def top_urls(candidates: list[Candidate], count: int) -> list[str]:
    """Pure-heuristic fallback used when the LLM navigator is off or fails."""
    return [c.url for c in candidates[:count] if c.score > 0]

"""HTML -> clean text, plus deterministic harvesting of contact data.

Two jobs:

1. Turn a raw DOM into compact prose the LLM can read cheaply. Scripts, CSS,
   SVG and nav/footer boilerplate are stripped before a single token is spent.
2. Find e-mails and LinkedIn URLs with regex rather than asking the LLM. These
   are exact strings that appear in the source; a regex cannot hallucinate one,
   so the deterministic pass is treated as ground truth and the LLM's own
   findings are checked against it.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify

_STRIP_TAGS = ("script", "style", "svg", "noscript", "iframe", "canvas", "template")

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b")
LINKEDIN_RE = re.compile(
    r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/(in|company)/[A-Za-z0-9\-_%.]+",
    re.IGNORECASE,
)

# `logo@2x.png`-style asset names and tracking noise match the e-mail shape but
# are obviously not addresses.
_ASSET_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".mp4", ".ico",
)
# Only genuinely reserved or boilerplate placeholders belong here. Domains that
# merely *look* like placeholders (acme.com, test.com) are really registered, and
# silently dropping a real lead is worse than keeping a dud one.
_EMAIL_NOISE = (
    "example.com", "example.org", "example.net", "domain.com", "yourcompany",
    "yourdomain", "email.com", "sentry.io", "wixpress.com", "@2x", "@3x",
)


# Below this many characters, a trafilatura result is assumed to have thrown
# away real content rather than boilerplate, and we cross-check the fallback.
_RECALL_THRESHOLD = 1_500


def clean_html(html: str) -> str:
    """Extract readable prose from a raw HTML document.

    trafilatura is excellent on article-shaped pages but treats marketing
    homepages -- which are mostly headings and CTA blocks -- as boilerplate and
    can return almost nothing. So when its output looks suspiciously thin we
    also run a manual strip and keep whichever is richer.
    """
    if not html:
        return ""

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_recall=True,
    )
    primary = _squash(extracted) if extracted else ""
    if len(primary) >= _RECALL_THRESHOLD:
        return primary

    fallback = _strip_to_text(html)
    return fallback if len(fallback) > len(primary) else primary


def _strip_to_text(html: str) -> str:
    """Manual strip of chrome and non-content tags, converted to markdown."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "header"]):
        tag.decompose()
    body = soup.body or soup
    return _squash(markdownify(str(body), strip=["img", "a"], heading_style="ATX"))


def page_title(html: str) -> str:
    """The document title, or an empty string."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.title.get_text(strip=True) if soup.title else ""


def _squash(text: str) -> str:
    """Collapse runaway whitespace -- pure token savings."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def find_emails(*sources: str) -> list[str]:
    """Public e-mail addresses found across the given HTML/text blobs."""
    found: dict[str, None] = {}
    for source in sources:
        if not source:
            continue
        for raw in EMAIL_RE.findall(source):
            candidate = unquote(raw).strip(".,;:").lower()
            if _is_plausible_email(candidate):
                found.setdefault(candidate, None)
    return list(found)


def _is_plausible_email(value: str) -> bool:
    if value.endswith(_ASSET_SUFFIXES):
        return False
    if any(noise in value for noise in _EMAIL_NOISE):
        return False
    local, _, domain = value.partition("@")
    if not local or not domain or "." not in domain:
        return False
    # Long hex-looking locals are almost always asset hashes or message IDs.
    return not (len(local) > 40 or re.fullmatch(r"[0-9a-f]{16,}", local))


def find_linkedin_urls(*sources: str) -> dict[str, list[str]]:
    """LinkedIn URLs split into personal profiles and company pages."""
    profiles: dict[str, None] = {}
    companies: dict[str, None] = {}
    for source in sources:
        if not source:
            continue
        for match in LINKEDIN_RE.finditer(source):
            url = match.group(0).rstrip("/.,)\"'").replace("http://", "https://")
            target = profiles if match.group(1).lower() == "in" else companies
            target.setdefault(url, None)
    return {"profiles": list(profiles), "companies": list(companies)}


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` chars on a paragraph boundary where possible."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = window.rfind("\n\n")
    if cut < limit * 0.6:  # no sensible paragraph break -- fall back to a word break
        cut = window.rfind(" ")
    if cut <= 0:
        cut = limit
    return window[:cut].rstrip() + "\n\n[truncated]"


def build_corpus(
    pages: list[tuple[str, str]], max_chars_per_page: int, max_chars_total: int
) -> str:
    """Join per-page text into one budgeted prompt corpus.

    ``pages`` is a list of ``(url, clean_text)``. Each page is capped, then the
    whole corpus is capped again, so prompt size stays bounded no matter how
    many or how large the pages are.
    """
    chunks: list[str] = []
    used = 0
    for url, text in pages:
        if not text:
            continue
        remaining = max_chars_total - used
        if remaining <= 200:
            break
        chunk = truncate(text, min(max_chars_per_page, remaining))
        block = f"## SOURCE: {url}\n\n{chunk}"
        chunks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(chunks)

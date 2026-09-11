"""Dynamic navigation: let the LLM decide which pages are worth crawling.

A fixed path list (``/about``, ``/team``, ``/company`` ...) guesses at URLs
that often do not exist -- vapi.ai has no ``/about``, for instance -- and burns
requests on 404s. Instead we show the model the links the homepage actually
has, and it picks.

The safety property that makes this cheap to trust: the returned URLs are
intersected with the candidate set, so a hallucinated path is discarded rather
than fetched. If the call fails entirely we fall back to the heuristic ranking,
so navigation degrades instead of breaking.
"""

from __future__ import annotations

import logging

from .discovery import Candidate, top_urls
from .llm import GeminiClient, UsageTracker
from .models import SelectedPages
from .prompts import NAVIGATOR_SYSTEM, NAVIGATOR_USER

log = logging.getLogger(__name__)


def _format_candidates(candidates: list[Candidate]) -> str:
    return "\n".join(f"{c.score:>4} | {c.url} | {c.anchor or '-'}" for c in candidates)


async def select_pages(
    client: GeminiClient | None,
    *,
    domain: str,
    homepage_url: str,
    homepage_text: str,
    candidates: list[Candidate],
    max_pages: int,
    usage: UsageTracker,
) -> tuple[list[str], str]:
    """Choose up to ``max_pages`` URLs to crawl next.

    Returns ``(urls, reason)`` where ``reason`` explains how the choice was
    made -- useful in the CLI output and when auditing a run afterwards.
    """
    if not candidates:
        return [], "no candidate links found on the homepage"

    if client is None:
        return top_urls(candidates, max_pages), "heuristic ranking (LLM navigation disabled)"

    prompt = NAVIGATOR_USER.format(
        url=homepage_url,
        domain=domain,
        summary=homepage_text[:1_200] or "(homepage text unavailable)",
        candidates=_format_candidates(candidates),
    )
    selection = await client.structured(
        prompt=prompt,
        schema=SelectedPages,
        system=NAVIGATOR_SYSTEM.format(max_pages=max_pages),
        usage=usage,
        temperature=0.0,
    )

    if selection is None:
        return top_urls(candidates, max_pages), "heuristic ranking (LLM navigation failed)"

    allowed = {c.url: c for c in candidates}
    chosen: list[str] = []
    rejected = 0
    for url in selection.urls:
        cleaned = url.strip().rstrip("/")
        match = allowed.get(cleaned) or allowed.get(url.strip())
        if match is None:
            rejected += 1
            continue
        if match.url not in chosen:
            chosen.append(match.url)

    if rejected:
        log.debug("discarded %d URL(s) not present in the candidate list", rejected)
    if not chosen:
        return top_urls(candidates, max_pages), "heuristic ranking (LLM returned no valid URLs)"

    reason = selection.reasoning.strip() or "LLM selection"
    if rejected:
        reason += f" ({rejected} invented URL(s) discarded)"
    return chosen[:max_pages], reason

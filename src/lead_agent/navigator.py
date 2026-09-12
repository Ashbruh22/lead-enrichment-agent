"""Dynamic navigation: let the LLM decide which pages are worth crawling.

A fixed path list (``/about``, ``/team``, ``/company`` ...) guesses at URLs
that often do not exist -- vapi.ai has no ``/about``, for instance -- and burns
requests on 404s. Instead we show the model the links the homepage actually
has, and it picks.

The safety property that makes this cheap to trust: the returned URLs are
intersected with the candidate set, so a hallucinated path is discarded rather
than fetched. If the call fails entirely we fall back to the heuristic ranking,
so navigation degrades instead of breaking.

There are two entry points, and together they form the agent loop.
:func:`select_pages` plans the first crawl from the homepage.
:func:`select_followup_pages` runs afterwards, once we know what the extraction
actually failed to find, and goes looking for pages that would fill exactly
those gaps.
"""

from __future__ import annotations

import logging

from .discovery import Candidate, top_urls
from .gaps import Gap, format_gaps
from .llm import GeminiClient, UsageTracker
from .models import SelectedPages
from .prompts import (
    FOLLOWUP_SYSTEM,
    FOLLOWUP_USER,
    NAVIGATOR_SYSTEM,
    NAVIGATOR_USER,
)

log = logging.getLogger(__name__)


def _format_candidates(candidates: list[Candidate]) -> str:
    return "\n".join(f"{c.score:>4} | {c.url} | {c.anchor or '-'}" for c in candidates)


def _keep_real_urls(
    selection: SelectedPages, candidates: list[Candidate], limit: int
) -> tuple[list[str], int]:
    """Intersect the model's choices with links that actually exist.

    This is the safety property that makes LLM-chosen navigation cheap to
    trust: a URL the model invented is simply not in the candidate map, so it
    is dropped rather than fetched. Returns the surviving URLs and how many
    were discarded.
    """
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
    return chosen[:limit], rejected


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

    chosen, rejected = _keep_real_urls(selection, candidates, max_pages)

    if rejected:
        log.debug("discarded %d URL(s) not present in the candidate list", rejected)
    if not chosen:
        return top_urls(candidates, max_pages), "heuristic ranking (LLM returned no valid URLs)"

    reason = selection.reasoning.strip() or "LLM selection"
    if rejected:
        reason += f" ({rejected} invented URL(s) discarded)"
    return chosen, reason


async def select_followup_pages(
    client: GeminiClient | None,
    *,
    domain: str,
    gaps: list[Gap],
    visited: list[str],
    candidates: list[Candidate],
    max_pages: int,
    usage: UsageTracker,
) -> tuple[list[str], str]:
    """Choose unread pages that might fill ``gaps``.

    The second half of the agent loop. Unlike :func:`select_pages` there is no
    heuristic fallback: a follow-up round is optional, so if the model cannot
    help we stop rather than spend fetches on the next-highest-scoring link,
    which the first round already declined to read.
    """
    if client is None or not candidates or not gaps:
        return [], "no follow-up attempted"

    prompt = FOLLOWUP_USER.format(
        domain=domain,
        visited="\n".join(f"- {url}" for url in visited) or "(nothing)",
        gaps=format_gaps(gaps),
        candidates=_format_candidates(candidates),
    )
    selection = await client.structured(
        prompt=prompt,
        schema=SelectedPages,
        system=FOLLOWUP_SYSTEM.format(max_pages=max_pages),
        usage=usage,
        temperature=0.0,
    )
    if selection is None:
        return [], "follow-up selection failed"

    chosen, rejected = _keep_real_urls(selection, candidates, max_pages)
    if not chosen:
        return [], "nothing left worth reading"

    reason = selection.reasoning.strip() or "follow-up selection"
    if rejected:
        reason += f" ({rejected} invented URL(s) discarded)"
    return chosen, reason

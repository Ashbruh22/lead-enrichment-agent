"""Per-domain orchestration.

The flow for one domain:

    homepage -> candidate links -> LLM picks pages -> fetch them ->
    clean + budget the text -> regex pass for contacts -> LLM extraction ->
    validate against source ->
        assess what is missing -> if anything is, pick unread pages that would
        fill it and go round again (bounded)
    -> fill missing LinkedIn URLs -> score

The loop is what makes this an agent rather than a pipeline: the second round
is not planned in advance, it happens because the first round's *output* was
short of something. Rounds, follow-up pages and total pages are all capped in
:mod:`lead_agent.config`, so the loop always terminates.

Every domain is wrapped in its own error boundary, and each result is written
to disk the moment it is ready, so one site failing -- or the run being
interrupted -- never costs you the domains that already succeeded.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from .browser import BrowserSession, FetchResult
from .config import Settings
from .discovery import extract_candidates, normalise
from .extract import build_corpus, find_emails, find_linkedin_urls
from .gaps import identify_gaps, summarise_gaps
from .llm import GeminiClient, UsageTracker
from .models import (
    CompanyIntel,
    ContactPoints,
    DomainResult,
    LlmCompanyIntel,
    PageRecord,
    RunMetrics,
    RunReport,
    TeamMember,
)
from .navigator import select_followup_pages, select_pages
from .prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from .scoring import blended_confidence
from .search import LinkedInFinder

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, str], None]
"""Callback taking (domain, human-readable status line)."""


@dataclass
class RunOptions:
    """Per-run switches, all surfaced as CLI flags."""

    use_llm_navigation: bool = True
    use_search: bool = True
    output_dir: Path = Path("outputs")


def normalise_domain(raw: str) -> str:
    """``https://www.Foo.com/bar`` -> ``foo.com``."""
    value = raw.strip().lower()
    if "//" in value:
        value = urlparse(value).netloc or value.split("//", 1)[1]
    value = value.split("/")[0].strip()
    return value[4:] if value.startswith("www.") else value


async def _fetch_homepage(
    session: BrowserSession, context, domain: str
) -> tuple[FetchResult, str]:
    """Try the bare domain, then ``www.`` -- some hosts only serve one."""
    primary = f"https://{domain}"
    result = await session.fetch(context, primary)
    if result.ok:
        return result, primary
    alternate = f"https://www.{domain}"
    retry = await session.fetch(context, alternate)
    if retry.ok:
        retry.notes.append("bare domain failed; served from www")
        return retry, alternate
    return result, primary


def _validates(model: type, **kwargs: object) -> bool:
    """True if ``model(**kwargs)`` passes Pydantic validation."""
    try:
        model(**kwargs)
    except ValidationError:
        return False
    return True


def _verify_against_source(values: list[str], source: str) -> list[str]:
    """Keep only strings that genuinely appear in the crawled content.

    This is the anti-hallucination gate: the LLM may only surface contact data
    that a substring search can confirm we actually saw.
    """
    haystack = source.lower()
    return [v for v in values if v and v.lower() in haystack]


def _build_intel(
    raw: LlmCompanyIntel,
    *,
    source_blob: str,
    known_emails: list[str],
    known_linkedins: list[str],
    contact_page: str | None,
) -> CompanyIntel:
    """Convert the LLM's loose payload into the validated output model."""
    verified_emails = _verify_against_source(raw.emails, source_blob)
    emails = list(dict.fromkeys(known_emails + verified_emails))

    known_lower = {u.lower() for u in known_linkedins}
    members: list[TeamMember] = []
    for person in raw.leadership:
        name = (person.name or "").strip()
        if not name:
            continue
        url = (person.linkedin_url or "").strip().rstrip("/")
        # Only trust a LinkedIn URL the deterministic scan also saw.
        if url and url.lower() not in known_lower:
            url = ""
        try:
            members.append(
                TeamMember(
                    name=name,
                    role=(person.role or None),
                    linkedin_url=url or None,
                    source="website" if url else "not_searched",
                )
            )
        except ValidationError:
            members.append(TeamMember(name=name, role=(person.role or None), source="not_searched"))

    # Validate each address on its own so one malformed string cannot discard
    # the whole contact list.
    valid_emails = [email for email in emails if _validates(ContactPoints, emails=[email])]
    contact_url = (
        contact_page
        if contact_page and _validates(ContactPoints, contact_page_url=contact_page)
        else None
    )
    contacts = ContactPoints(emails=valid_emails, contact_page_url=contact_url)

    return CompanyIntel(
        company_overview=raw.company_overview.strip(),
        target_audience=raw.target_audience.strip(),
        contact_points=contacts,
        leadership=members,
        data_confidence_score=raw.data_confidence_score,
        llm_self_score=round(max(0.0, min(1.0, raw.data_confidence_score)), 3),
    )


async def _extract_round(
    client: GeminiClient,
    domain: str,
    good: list[FetchResult],
    settings: Settings,
    usage: UsageTracker,
    say: Callable[[str], None],
    round_number: int,
) -> tuple[CompanyIntel | None, str]:
    """One extraction pass over everything retrieved so far.

    Re-extracting over the whole corpus each round, rather than extracting the
    new pages alone and merging, costs one extra call on the domains that need
    a second round. It buys correctness: the model sees a team page in the same
    context as the homepage, so it can attribute titles and tell staff from
    customers. Merging two independent extractions cannot do that, and a
    follow-up round only happens when the first one came up short anyway.
    """
    corpus = build_corpus(
        [(p.url, p.text) for p in good],
        settings.max_chars_per_page,
        settings.max_chars_total,
    )
    source_blob = "\n".join([p.html for p in good] + [p.text for p in good])
    emails = find_emails(source_blob)
    linkedins = find_linkedin_urls(source_blob)
    contact_page = next((p.url for p in good if "contact" in p.url.lower()), None)

    say("extracting with Gemini" if round_number == 1 else "re-extracting with new pages")
    raw_intel = await client.structured(
        prompt=EXTRACTION_USER.format(
            domain=domain,
            emails=", ".join(emails) or "none",
            linkedins=", ".join(linkedins["profiles"]) or "none",
            corpus=corpus,
        ),
        schema=LlmCompanyIntel,
        system=EXTRACTION_SYSTEM.format(domain=domain),
        usage=usage,
    )
    if raw_intel is None:
        return None, source_blob

    return (
        _build_intel(
            raw_intel,
            source_blob=source_blob,
            known_emails=emails,
            known_linkedins=linkedins["profiles"],
            contact_page=contact_page,
        ),
        source_blob,
    )


async def enrich_domain(
    session: BrowserSession,
    client: GeminiClient | None,
    raw_domain: str,
    settings: Settings,
    options: RunOptions,
    progress: ProgressFn | None = None,
    finder: LinkedInFinder | None = None,
) -> DomainResult:
    """Run the full pipeline for one domain. Never raises."""
    domain = normalise_domain(raw_domain)
    started = time.perf_counter()
    usage = UsageTracker(model=settings.gemini_model)
    result = DomainResult(domain=domain, url=f"https://{domain}", status="failed")
    rounds_run = 0  # survives the `finally` block, which rebuilds result.metrics

    def say(message: str) -> None:
        if progress:
            progress(domain, message)

    context = None
    try:
        context = await session.new_context()

        say("fetching homepage")
        home, home_url = await _fetch_homepage(session, context, domain)
        result.url = home_url
        result.pages.append(
            PageRecord(
                url=home_url,
                ok=home.ok,
                chars=len(home.text),
                note="; ".join(home.notes) or None,
            )
        )
        if not home.ok:
            result.errors.append(f"homepage unreachable: {home.error}")
            return result

        say("scoring links")
        candidates = extract_candidates(
            home.html,
            home_url,
            limit=settings.max_link_candidates,
            exclude={normalise(home_url)},
        )

        selected, reason = await select_pages(
            client if options.use_llm_navigation else None,
            domain=domain,
            homepage_url=home_url,
            homepage_text=home.text,
            candidates=candidates,
            max_pages=settings.max_pages_per_domain,
            usage=usage,
        )
        say(f"selected {len(selected)}/{len(candidates)} links - {reason}")

        fetched = (
            await session.fetch_many(context, selected, settings.page_concurrency)
            if selected
            else []
        )
        for page in fetched:
            result.pages.append(
                PageRecord(
                    url=page.url,
                    ok=page.ok,
                    chars=len(page.text),
                    note=page.error or ("; ".join(page.notes) or None),
                )
            )

        good = [home, *[p for p in fetched if p.ok and p.text]]
        say(f"retrieved {len(good)} page(s)")

        if client is None:
            result.status = "partial"
            result.errors.append("LLM disabled; only deterministic extraction ran")
            return result

        # --- the agent loop ------------------------------------------------ #
        # Extract, look at what is missing, and if anything important is, go
        # back for pages that would fill exactly those gaps. Most domains
        # answer everything on the first round and never enter the second.
        visited = {normalise(home_url), *(normalise(p.url) for p in fetched)}
        intel: CompanyIntel | None = None

        for round_number in range(1, settings.max_agent_rounds + 1):
            rounds_run = round_number
            intel, source_blob = await _extract_round(
                client, domain, good, settings, usage, say, round_number
            )
            if intel is None:
                result.status = "partial"
                result.errors.append("LLM extraction returned no usable structured output")
                return result

            gaps = identify_gaps(intel)
            if not gaps:
                break
            if round_number == settings.max_agent_rounds:
                result.errors.append(f"still missing after {round_number} round(s): "
                                     f"{summarise_gaps(gaps)}")
                break
            if len(result.pages) >= settings.max_pages_total:
                result.errors.append("page budget reached; stopped looking")
                break

            # Only pages the heuristic does not actively reject. The prompt
            # asks the model to return nothing rather than fetch a page that
            # cannot hold the answer, but on a site with no team page it will
            # reach for whatever is left -- vapi.ai's run picked /blog and
            # /community, scored -8 and -4. Enforce it in code, not in prose.
            unread = [
                c for c in candidates if c.score > 0 and normalise(c.url) not in visited
            ]
            if not unread:
                result.errors.append(f"no unread pages left to fill: {summarise_gaps(gaps)}")
                break

            budget = min(
                settings.max_followup_pages, settings.max_pages_total - len(result.pages)
            )
            more, why = await select_followup_pages(
                client,
                domain=domain,
                gaps=gaps,
                visited=[p.url for p in good],
                candidates=unread,
                max_pages=budget,
                usage=usage,
            )
            if not more:
                result.errors.append(f"{summarise_gaps(gaps)} not found; {why}")
                break

            say(f"round {round_number + 1}: {summarise_gaps(gaps)} missing - {why}")
            extra = await session.fetch_many(context, more, settings.page_concurrency)
            visited.update(normalise(p.url) for p in extra)
            for page in extra:
                result.pages.append(
                    PageRecord(
                        url=page.url,
                        ok=page.ok,
                        chars=len(page.text),
                        note=page.error or ("; ".join(page.notes) or None),
                    )
                )
            usable = [p for p in extra if p.ok and p.text]
            if not usable:
                result.errors.append("follow-up pages could not be read")
                break
            good.extend(usable)

        missing = [m for m in intel.leadership if not m.linkedin_url]
        if options.use_search and missing and finder is not None:
            finder.bind(context)
            say(f"searching LinkedIn for {len(missing)} person(s)")
            await _fill_linkedin(missing, domain, finder)

        pages_ok = sum(1 for p in result.pages if p.ok)
        blended, objective = blended_confidence(intel, pages_ok, len(result.pages))
        intel.data_confidence_score = blended
        intel.completeness_score = objective

        result.intel = intel
        result.status = "ok" if pages_ok > 1 else "partial"
        if pages_ok <= 1:
            result.errors.append("only the homepage could be retrieved")

    except Exception as exc:  # the error boundary: one domain can never kill the run
        log.exception("unhandled error while enriching %s", domain)
        result.errors.append(f"unhandled error: {type(exc).__name__}: {str(exc)[:200]}")
        result.status = "failed"
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the real result
                pass

        result.errors.extend(usage.errors)
        result.metrics = RunMetrics(
            pages_fetched=sum(1 for p in result.pages if p.ok),
            pages_failed=sum(1 for p in result.pages if not p.ok),
            agent_rounds=rounds_run,
            llm_calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=round(usage.cost_usd, 6),
            duration_seconds=round(time.perf_counter() - started, 2),
        )
        _write_partial(result, options.output_dir)

    return result


async def _fill_linkedin(
    members: list[TeamMember], domain: str, finder: LinkedInFinder
) -> None:
    """Look up profile URLs for team members missing one, in sequence.

    Sequential on purpose: parallel search requests are the fastest way to get
    rate-limited, and this runs at most a handful of times per domain. The
    finder is shared across the run, so a person looked up once -- or a
    provider found to be blocked -- is not paid for again.
    """
    company = domain.split(".")[0]
    for member in members:
        lookup = await finder.find(member.name, company)
        if lookup.status != "found" or not lookup.url:
            member.source = (
                "search_unavailable" if lookup.status == "unavailable" else "searched_not_found"
            )
            continue
        try:
            member.linkedin_url = lookup.url  # type: ignore[assignment]
            TeamMember.model_validate(member.model_dump())
            member.source = "search"
        except ValidationError:
            # The provider handed us something that is not a usable URL.
            member.linkedin_url = None
            member.source = "searched_not_found"


def _write_partial(result: DomainResult, output_dir: Path) -> None:
    """Persist a single domain immediately, so progress survives a crash."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{result.domain.replace('/', '_')}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("could not write partial result for %s: %s", result.domain, exc)


async def run_domains(
    domains: list[str],
    settings: Settings,
    options: RunOptions,
    *,
    headless: bool = True,
    client: GeminiClient | None = None,
    progress: ProgressFn | None = None,
) -> RunReport:
    """Enrich every domain, with bounded concurrency, and aggregate the totals."""
    semaphore = asyncio.Semaphore(max(1, settings.domain_concurrency))
    # One finder for the whole run: its cache and its record of which providers
    # are blocked are only worth keeping if they outlive a single domain.
    finder = LinkedInFinder.build(settings) if options.use_search else None

    async with BrowserSession(settings, headless=headless) as session:

        async def worker(domain: str) -> DomainResult:
            async with semaphore:
                return await enrich_domain(
                    session, client, domain, settings, options, progress, finder
                )

        results = list(await asyncio.gather(*(worker(d) for d in domains)))

    totals = RunMetrics(
        pages_fetched=sum(r.metrics.pages_fetched for r in results),
        pages_failed=sum(r.metrics.pages_failed for r in results),
        agent_rounds=sum(r.metrics.agent_rounds for r in results),
        llm_calls=sum(r.metrics.llm_calls for r in results),
        prompt_tokens=sum(r.metrics.prompt_tokens for r in results),
        completion_tokens=sum(r.metrics.completion_tokens for r in results),
        total_tokens=sum(r.metrics.total_tokens for r in results),
        estimated_cost_usd=round(sum(r.metrics.estimated_cost_usd for r in results), 6),
        duration_seconds=round(sum(r.metrics.duration_seconds for r in results), 2),
    )
    return RunReport(model=settings.gemini_model, results=results, totals=totals)


def write_report(report: RunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def write_csv(report: RunReport, path: Path) -> None:
    """Flat one-row-per-company view, for dropping straight into a CRM."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "domain", "status", "company_overview", "target_audience", "emails",
                "leadership", "linkedin_urls", "confidence", "pages_fetched",
                "total_tokens", "estimated_cost_usd",
            ]
        )
        for r in report.results:
            intel = r.intel
            writer.writerow(
                [
                    r.domain,
                    r.status,
                    intel.company_overview if intel else "",
                    intel.target_audience if intel else "",
                    "; ".join(str(e) for e in intel.contact_points.emails) if intel else "",
                    "; ".join(
                        f"{m.name} ({m.role})" if m.role else m.name for m in intel.leadership
                    )
                    if intel
                    else "",
                    "; ".join(str(m.linkedin_url) for m in intel.leadership if m.linkedin_url)
                    if intel
                    else "",
                    intel.data_confidence_score if intel else "",
                    r.metrics.pages_fetched,
                    r.metrics.total_tokens,
                    r.metrics.estimated_cost_usd,
                ]
            )


def load_domains(path: Path) -> list[str]:
    """Read domains from a file, one per line, ``#`` for comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

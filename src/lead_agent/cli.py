"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import Settings, load_settings
from .llm import GeminiClient, LlmUnavailableError
from .models import RunReport
from .pipeline import RunOptions, load_domains, run_domains, write_csv, write_report

console = Console()

STATUS_STYLE = {"ok": "green", "partial": "yellow", "failed": "red"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lead-agent",
        description="Crawl company domains and extract structured lead intelligence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  lead-agent --domains postman.com supabase.com vapi.ai\n"
            "  lead-agent --input domains.txt --out output.json\n"
            "  lead-agent --domains vapi.ai --headful --verbose\n"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--domains", nargs="+", metavar="DOMAIN", help="Domains to enrich.")
    source.add_argument("--input", type=Path, metavar="FILE", help="File with one domain per line.")

    parser.add_argument("--out", type=Path, default=Path("output.json"), help="JSON output path.")
    parser.add_argument("--csv", type=Path, help="Also write a flattened CSV here.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for per-domain results, written as each domain finishes.",
    )
    parser.add_argument("--concurrency", type=int, help="Domains to process in parallel.")
    parser.add_argument("--max-pages", type=int, help="Max subpages to crawl per domain.")
    parser.add_argument(
        "--max-rounds",
        type=int,
        help="Assess-and-continue rounds per domain (1 disables the follow-up loop).",
    )
    parser.add_argument(
        "--headful", action="store_true", help="Show the browser window (useful for demos)."
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip the external LinkedIn lookup for people missing a profile URL.",
    )
    parser.add_argument(
        "--no-llm-nav",
        action="store_true",
        help="Use heuristic link ranking instead of LLM page selection.",
    )
    parser.add_argument(
        "--no-robots", action="store_true", help="Do not consult robots.txt before crawling."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # These are chatty at DEBUG and drown out our own output.
    for noisy in ("httpx", "httpcore", "urllib3", "trafilatura"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # google_genai warns about automatic function calling on every single call.
    # We pass no tools, so the warning does not apply to us; real API failures
    # surface as exceptions and are reported by the pipeline instead.
    logging.getLogger("google_genai").setLevel(logging.ERROR)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.concurrency:
        settings.domain_concurrency = args.concurrency
    if args.max_pages:
        settings.max_pages_per_domain = args.max_pages
    if args.max_rounds:
        settings.max_agent_rounds = args.max_rounds
    if args.no_robots:
        settings.respect_robots = False
    return settings


def render_summary(report: RunReport) -> None:
    table = Table(title="Lead enrichment run", header_style="bold", title_style="bold")
    table.add_column("Domain")
    table.add_column("Status")
    table.add_column("Pages", justify="right")
    table.add_column("Rnds", justify="right")
    table.add_column("People", justify="right")
    table.add_column("Emails", justify="right")
    table.add_column("Conf.", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost $", justify="right")
    table.add_column("Time", justify="right")

    for result in report.results:
        intel = result.intel
        status = result.status
        m = result.metrics
        table.add_row(
            result.domain,
            f"[{STATUS_STYLE.get(status, 'white')}]{status}[/]",
            f"{m.pages_fetched}/{m.pages_fetched + m.pages_failed}",
            str(m.agent_rounds),
            str(len(intel.leadership)) if intel else "-",
            str(len(intel.contact_points.emails)) if intel else "-",
            f"{intel.data_confidence_score:.2f}" if intel else "-",
            f"{m.total_tokens:,}",
            f"{m.estimated_cost_usd:.5f}",
            f"{m.duration_seconds:.1f}s",
        )

    totals = report.totals
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]",
        "",
        f"{totals.pages_fetched}/{totals.pages_fetched + totals.pages_failed}",
        "",
        "",
        "",
        "",
        f"[bold]{totals.total_tokens:,}[/]",
        f"[bold]{totals.estimated_cost_usd:.5f}[/]",
        "",
    )
    console.print(table)
    console.print(
        f"[dim]Model {report.model} - {totals.llm_calls} LLM call(s), "
        f"{totals.prompt_tokens:,} in / {totals.completion_tokens:,} out. "
        f"Cost is estimated at published paid-tier rates.[/dim]"
    )

    for result in report.results:
        if result.errors:
            console.print(f"[yellow]{result.domain}[/]: " + "; ".join(result.errors[:3]))


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252 and will crash on the first bullet or
    # arrow scraped from a website.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    domains = args.domains or load_domains(args.input)
    if not domains:
        console.print("[red]No domains supplied.[/]")
        return 2

    settings = apply_overrides(load_settings(), args)

    try:
        client = GeminiClient(settings)
    except LlmUnavailableError as exc:
        console.print(f"[red]{exc}[/]")
        return 2

    options = RunOptions(
        use_llm_navigation=not args.no_llm_nav,
        use_search=not args.no_search,
        output_dir=args.output_dir,
    )

    console.print(
        f"[bold]Enriching {len(domains)} domain(s)[/] with {settings.gemini_model} "
        f"({'headful' if args.headful else 'headless'} Chromium, "
        f"{settings.domain_concurrency} at a time)\n"
    )

    def progress(domain: str, message: str) -> None:
        console.print(f"  [cyan]{domain:<20}[/] {message}")

    report = asyncio.run(
        run_domains(
            domains,
            settings,
            options,
            headless=not args.headful,
            client=client,
            progress=progress,
        )
    )

    write_report(report, args.out)
    if args.csv:
        write_csv(report, args.csv)

    console.print()
    render_summary(report)
    console.print(f"\n[green]Wrote[/] {args.out}" + (f" and {args.csv}" if args.csv else ""))

    # Non-zero only if nothing at all worked -- a partial run is still useful.
    return 0 if any(r.status != "failed" for r in report.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

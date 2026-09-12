# Autonomous Lead Enrichment Agent

Give it a list of company domains. It drives a headless browser over each one,
decides which pages are worth reading, and returns structured lead intelligence
— company overview, ideal customer profile, public contact e-mails, named
leadership with LinkedIn URLs, and a confidence score — as validated JSON.

```
$ lead-agent --input domains.txt --out output.json --csv output.csv

  postman.com          fetching homepage
  postman.com          selected 4/25 links - Selected key company overview, contact, product
                       and pricing pages to learn about Postman's offerings, target audience,
                       leadership, and contact details.
  postman.com          retrieved 5 page(s)
  postman.com          extracting with Gemini
  postman.com          searching LinkedIn for 3 person(s)
  ...
  vapi.ai: no unread pages left to fill: leadership

                                     Lead enrichment run
  ┏━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
  ┃ Domain       ┃ Status ┃ Pages ┃ Rnds ┃ People ┃ Emails ┃ Conf. ┃ Tokens ┃  Cost $ ┃  Time ┃
  ┡━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
  │ postman.com  │ ok     │   5/5 │    1 │      3 │      2 │  0.98 │  5,858 │ 0.00544 │ 27.5s │
  │ supabase.com │ ok     │   5/5 │    1 │      3 │      5 │  0.93 │  5,651 │ 0.00523 │ 22.6s │
  │ vapi.ai      │ ok     │   6/6 │    1 │      0 │      2 │  0.75 │  4,591 │ 0.00415 │ 18.0s │
  ├──────────────┼────────┼───────┼──────┼────────┼────────┼───────┼────────┼─────────┼───────┤
  │ TOTAL        │        │ 16/16 │      │        │        │       │ 16,100 │ 0.01482 │       │
  └──────────────┴────────┴───────┴──────┴────────┴────────┴───────┴────────┴─────────┴───────┘
```

---

## Setup

Requires **Python 3.11+**.

```bash
git clone <this-repo>
cd lead-enrichment-agent

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                  # installs the `lead-agent` command
playwright install chromium       # one-time browser download (~130 MB)
```

### Environment variables

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | **yes** | Free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | no | Defaults to `gemini-3.6-flash` |
| `TAVILY_API_KEY` | recommended | What makes LinkedIn discovery actually work. Free tier at [tavily.com](https://tavily.com). Without it the agent falls back to scraping Brave/DuckDuckGo, which get rate-limited quickly |

The free Gemini tier is enough to run all three test domains many times over.

---

## Running it

```bash
# the three assignment targets
lead-agent --input domains.txt --out output.json --csv output.csv

# ad-hoc domains
lead-agent --domains stripe.com linear.app

# watch the browser work — this is the one to use for a demo
lead-agent --domains vapi.ai --headful --verbose
```

Or without installing the entry point: `python -m lead_agent --input domains.txt`.

### Flags

| Flag | Effect |
|---|---|
| `--domains A B C` / `--input FILE` | Where the domains come from (one is required) |
| `--out`, `--csv` | Output paths (JSON always; CSV optional) |
| `--output-dir` | Per-domain JSON, written the instant each domain finishes (default `outputs/`) |
| `--concurrency` | Domains processed in parallel (default 2) |
| `--max-pages` | Subpages crawled per domain, per round (default 5) |
| `--max-rounds` | Assess-and-continue rounds per domain (default 3; `1` disables the loop) |
| `--headful` | Show the Chromium window |
| `--no-llm-nav` | Use heuristic link ranking instead of LLM page selection |
| `--no-search` | Skip the external LinkedIn lookup |
| `--no-robots` | Do not consult robots.txt |
| `--verbose` | Debug logging |

---

## How it works

```
domain
  │
  ├─ 1. browser.py      Playwright fetches the homepage. Images, fonts, media and
  │                     CSS are aborted at the network layer; only the DOM matters.
  │
  ├─ 2. discovery.py    Every same-site link is normalised, deduped and scored by
  │                     slug/anchor keywords. /about scores +10, /docs scores −10.
  │
  ├─ 3. navigator.py    The LLM picks which of those real links to crawl. Its
  │                     choices are intersected with the candidate set, so an
  │                     invented URL is discarded rather than fetched.
  │
  ├─ 4. browser.py      Selected pages fetched with bounded concurrency.
  │
  ├─ 5. extract.py      HTML → clean prose (trafilatura), per-page and per-corpus
  │                     character budgets. Regex pass harvests e-mails + LinkedIn URLs.
  │
  ├─ 6. llm.py          One schema-constrained Gemini call returns the full record.
  │
  ├─ 7. pipeline.py     LLM output is validated against the source text; anything
  │                     unverifiable is dropped.
  │
  ├─ 8. search.py       People still missing a LinkedIn URL get looked up:
  │                     Tavily's API first, browser search as fallback, accepted
  │                     only on a name match. Cached across the whole run.
  │
  ├─ 9. gaps.py         What the extraction failed to find. If anything important
  │                     is missing and there are unread links worth trying, the
  │                     agent goes back to step 3 for pages targeting those gaps.
  │
  └─ 10. scoring.py     Confidence = 0.35 × LLM self-score + 0.65 × measured completeness.
```

Steps 3-9 form a bounded loop: the second round is not planned in advance, it
happens because the first round's *output* came up short. Rounds, follow-up
pages and total pages are all capped, so it always terminates.

### Design decisions worth explaining

**The LLM never sees raw HTML.** Pages go through trafilatura (with a manual
strip-and-markdown fallback for marketing homepages, which trafilatura tends to
discard as boilerplate). Text is then capped per page and per corpus, so prompt
size is bounded no matter how large the site is. A typical domain costs well
under 10k tokens.

**Deterministic extraction and LLM extraction do different jobs.** E-mails and
LinkedIn URLs are exact strings that exist in the source, so they are found with
regex, not asked for. Those results are handed to the model as verified facts,
and after extraction every contact detail the model returned is checked back
against the source text — anything that is not literally there is dropped. The
LLM is left with the jobs it is actually good at: summarising, classifying the
ICP, and attributing titles to names.

**Dynamic navigation beats a hardcoded path list.** Guessing `/about`, `/team`,
`/company` wastes requests on 404s — vapi.ai has no `/about` page. Showing the
model the links that exist and letting it choose means every fetch is a real
URL, and the validation step means a hallucinated path costs nothing.

**Search providers are ordered by how much they can be trusted.** An
unauthenticated scrape of a search engine is unreliable by design — the engine
is actively trying to stop you, and cleverness is just the rate limiter's next
target. So Tavily's API goes first when a key is configured and scraping is the
fallback, a provider that fails is retired for the whole run instead of being
re-probed per person, and misses are cached as firmly as hits because failure is
the common case. The name-match guard sits in the finder rather than in any
provider, so every path is checked by the same code.

**Confidence is not self-reported.** Models rate their own output optimistically
and with little variance, so the final score blends the model's estimate (35%)
with a completeness measure computed from observable facts (65%): pages actually
retrieved, whether contact e-mails were found, whether people were named, and
whether those people have verifiable LinkedIn URLs. `llm_self_score` and
`completeness_score` are both kept in the output so the blend can be audited.

**Cost is reported even though the free tier bills nothing.** Token counts come
from Gemini's `usage_metadata`; cost is computed at published paid-tier rates
($0.75 / $3.75 per 1M tokens in/out for `gemini-3.6-flash`). That is the number
that matters when this runs over ten thousand domains instead of three.
Deliberation is turned down to the floor the model family allows — these are
extraction tasks over text we already supply, and thinking tokens bill at the
output rate. Gemini 2.x takes a numeric `thinking_budget` and Gemini 3 replaced
it with `thinking_level`, rejecting the old field with a 400, so `llm.py` picks
per family and degrades to no thinking config at all if the model refuses.

---

## Resilience

The script is built so that no single site can end a run.

| Failure | Handling |
|---|---|
| Nav timeout / connection reset | Retried with exponential backoff (`tenacity`) |
| JS-heavy SPA never settles | `networkidle` wait is bounded; the DOM is captured as-is |
| Bot wall / empty render | Detected by title pattern and text length, then retried over plain HTTP |
| 404 or 5xx subpage | Recorded on the page record; the domain continues |
| Bare domain does not resolve | Automatically retried as `www.` |
| Unparseable LLM output | Re-parsed from raw text, then degraded to a `partial` result |
| Gemini 429 / 503 | Retried with exponential backoff, capped at 45s so it can outwait a per-minute free-tier quota |
| Search provider blocked or rate-limited | Providers are tried in order of reliability (Tavily API, then Brave, then DuckDuckGo); one that fails is retired for the rest of the run rather than re-queried per person, and misses are cached so a failure is never paid for twice |
| Anything unforeseen | Per-domain error boundary → `status: "failed"` with the reason recorded |

Results are written to `outputs/<domain>.json` the moment each domain finishes,
so an interrupted run keeps everything already completed.

`robots.txt` is honoured by default, requests are spaced out, and concurrency is
bounded.

---

## Output

`output.json`:

```jsonc
{
  "generated_at": "2026-09-12T15:21:58.775229",
  "model": "gemini-3.6-flash",
  "results": [
    {
      "domain": "postman.com",
      "url": "https://postman.com",
      "status": "ok",
      "intel": {
        "company_overview": "Postman is a unified API platform for designing, testing, distributing, documenting, and monitoring APIs. ...",
        "target_audience": "Developers and enterprise engineering teams building, testing, and managing APIs.",
        "contact_points": {
          "emails": ["info@postman.com", "info-jp@postman.com"],
          "contact_page_url": "https://postman.com/company/contact-us"
        },
        "leadership": [
          { "name": "Abhinav Asthana", "role": "CEO and co-founder",
            "linkedin_url": "https://www.linkedin.com/in/abhinavasthana", "source": "search" },
          { "name": "Ankit Sobti", "role": "co-founder",
            "linkedin_url": "https://www.linkedin.com/in/ankit-sobti", "source": "search" },
          { "name": "Abhijit Kane", "role": "co-founder",
            "linkedin_url": "https://in.linkedin.com/in/abhijitkane", "source": "search" }
        ],
        "data_confidence_score": 0.982,
        "llm_self_score": 0.95,
        "completeness_score": 1.0
      },
      "pages": [
        { "url": "https://postman.com", "ok": true, "chars": 770,
          "note": "network never went idle; captured DOM as-is" }
        // ... 4 more
      ],
      "errors": [],
      "metrics": {
        "pages_fetched": 5, "pages_failed": 0, "agent_rounds": 1, "llm_calls": 2,
        "prompt_tokens": 5509, "completion_tokens": 349,
        "total_tokens": 5858, "estimated_cost_usd": 0.005441,
        "duration_seconds": 27.53
      },
      "scraped_at": "2026-09-12T15:21:18.037788"
    }
    // ... supabase.com, vapi.ai
  ],
  "totals": { "pages_fetched": 16, "total_tokens": 16100, "estimated_cost_usd": 0.014817 }
}
```

`source` on each person records provenance: `website` when the URL was on the
company's own site, `search` when it came from the external lookup,
`searched_not_found` when we looked and no candidate passed the name match, and
`search_unavailable` when every provider was blocked.

The committed `output.json` and `output.csv` are the real artefacts of the run
in the table above, on the default model with a `TAVILY_API_KEY` configured.

---

## Tests

```bash
pytest -v
```

The suite covers the pure functions — HTML cleaning, e-mail and LinkedIn
harvesting, token budgeting, link scoring and normalisation, the LinkedIn
name-match guard, and confidence scoring — using saved HTML fixtures. It needs
no network access and no API key.

---

## Project layout

```
src/lead_agent/
  cli.py          argument parsing, progress output, summary table
  config.py       settings, timeouts, budgets, model pricing
  models.py       Pydantic schemas (LLM wire schema + strict output schema)
  browser.py      Playwright lifecycle, fetching, retries, fallbacks
  robots.py       robots.txt cache
  discovery.py    link extraction, normalisation and scoring
  navigator.py    LLM page selection
  extract.py      HTML → text, regex harvesting, token budgeting
  llm.py          Gemini client, structured output, token/cost accounting
  prompts.py      prompt text
  search.py       external LinkedIn lookup
  scoring.py      confidence blending
  pipeline.py     per-domain orchestration and error boundary
```

## Known limitations

- LinkedIn blocks automated profile access, so profile URLs come from search
  results and are validated by name match rather than by opening the profile. A
  slug match is strong evidence, not proof. Without a `TAVILY_API_KEY` the agent
  falls back to scraping Brave and DuckDuckGo, which rate-limit quickly; when no
  provider can confirm a profile the field stays `null` and `source` records
  whether we looked and failed (`searched_not_found`) or could not look at all
  (`search_unavailable`), rather than filling in a guess.

- Sites behind an aggressive WAF may yield only the homepage. The run reports
  this as `partial` with the reason recorded rather than inventing data.
- Only English-language content is prompted for.
- Personal e-mail addresses are deliberately not targeted; the extraction asks
  only for generic/public inboxes.

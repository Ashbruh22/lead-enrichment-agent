# Autonomous Lead Enrichment Agent

Give it a list of company domains. It drives a headless browser over each one,
decides which pages are worth reading, and returns structured lead intelligence
— company overview, ideal customer profile, public contact e-mails, named
leadership with LinkedIn URLs, and a confidence score — as validated JSON.

```
$ lead-agent --domains postman.com supabase.com vapi.ai

  postman.com          fetching homepage
  postman.com          selected 4/25 links - About, contact and careers pages cover the
                       company, its audience and its contact details.
  postman.com          retrieved 5 page(s)
  postman.com          extracting with Gemini
  ...

                              Lead enrichment run
  ┏━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
  ┃ Domain       ┃ Status ┃ Pages ┃ People ┃ Emails ┃ Conf. ┃ Tokens ┃  Cost $ ┃  Time ┃
  ┡━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
  │ postman.com  │ ok     │   5/0 │      4 │      2 │  0.87 │  9,214 │ 0.00305 │ 31.2s │
  └──────────────┴────────┴───────┴────────┴────────┴───────┴────────┴─────────┴───────┘
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
| `GEMINI_MODEL` | no | Defaults to `gemini-2.5-flash` |
| `TAVILY_API_KEY` | no | Fallback search provider if the built-in browser search is rate-limited |

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
| `--max-pages` | Subpages crawled per domain (default 5) |
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
  ├─ 8. search.py       People still missing a LinkedIn URL get looked up via a
  │                     browser search, accepted only on a name match.
  │
  └─ 9. scoring.py      Confidence = 0.35 × LLM self-score + 0.65 × measured completeness.
```

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

**Confidence is not self-reported.** Models rate their own output optimistically
and with little variance, so the final score blends the model's estimate (35%)
with a completeness measure computed from observable facts (65%): pages actually
retrieved, whether contact e-mails were found, whether people were named, and
whether those people have verifiable LinkedIn URLs. `llm_self_score` and
`completeness_score` are both kept in the output so the blend can be audited.

**Cost is reported even though the free tier bills nothing.** Token counts come
from Gemini's `usage_metadata`; cost is computed at published paid-tier rates
($0.30 / $2.50 per 1M tokens in/out for `gemini-2.5-flash`). That is the number
that matters when this runs over ten thousand domains instead of three. Gemini's
"thinking" mode is disabled — these are extraction tasks over supplied text, and
thinking tokens bill as output.

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
| Gemini 429 / 503 | Retried with exponential backoff |
| Search engine rate-limits | LinkedIn lookup disables itself for the rest of the run |
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
  "generated_at": "...",
  "model": "gemini-2.5-flash",
  "results": [
    {
      "domain": "supabase.com",
      "url": "https://supabase.com",
      "status": "ok",
      "intel": {
        "company_overview": "Supabase is an open-source backend-as-a-service …",
        "target_audience": "Developers and teams building applications …",
        "contact_points": {
          "emails": ["support@supabase.io"],
          "contact_page_url": "https://supabase.com/contact-us"
        },
        "leadership": [
          {
            "name": "Paul Copplestone",
            "role": "CEO & Co-Founder",
            "linkedin_url": "https://www.linkedin.com/in/paulcopplestone",
            "source": "search"
          }
        ],
        "data_confidence_score": 0.85,
        "llm_self_score": 0.9,
        "completeness_score": 0.83
      },
      "pages": [ { "url": "…", "ok": true, "chars": 2433, "note": null } ],
      "errors": [],
      "metrics": {
        "pages_fetched": 5, "pages_failed": 0, "llm_calls": 2,
        "prompt_tokens": 8134, "completion_tokens": 1080,
        "total_tokens": 9214, "estimated_cost_usd": 0.00514,
        "duration_seconds": 28.4
      },
      "scraped_at": "..."
    }
  ],
  "totals": { }
}
```

`source` on each team member records provenance: `website` when the LinkedIn URL
was on the company's own site, `search` when it came from the external lookup,
`unknown` when no profile was found.

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

- LinkedIn blocks automated profile access, so profile URLs are discovered
  through search results and validated by name match rather than by opening the
  profile. A slug match is strong evidence but not proof.
- Sites behind an aggressive WAF may yield only the homepage. The run reports
  this as `partial` with the reason recorded rather than inventing data.
- Only English-language content is prompted for.
- Personal e-mail addresses are deliberately not targeted; the extraction asks
  only for generic/public inboxes.

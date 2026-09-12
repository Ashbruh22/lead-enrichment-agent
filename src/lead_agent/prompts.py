"""Prompt text, kept out of the logic modules so it can be tuned in one place."""

from __future__ import annotations

NAVIGATOR_SYSTEM = """\
You plan a small web crawl. You are given a company's homepage and a list of \
candidate links from that page. Choose the pages most likely to reveal: what \
the company does, who its product is for, its public contact e-mails, and its \
founders or leadership team.

Rules:
- Return ONLY URLs copied verbatim from the candidate list. Never invent a URL.
- Prefer about / team / company / leadership / contact pages.
- Avoid documentation, blog posts, changelogs, legal pages and login pages.
- Choose at most {max_pages} URLs. Fewer is fine if the rest add nothing.
"""

NAVIGATOR_USER = """\
Homepage: {url}
Company: {domain}

Homepage summary:
{summary}

Candidate links (score | url | link text):
{candidates}
"""

EXTRACTION_SYSTEM = """\
You extract company intelligence for a B2B sales-research tool. You are given \
cleaned text from several pages of one company's website.

Rules:
- Use ONLY the supplied content. If something is not stated, leave it null or \
omit it. Never guess, infer or fill in from your own knowledge of the company.
- company_overview must be exactly two sentences describing what the company \
actually does.
- target_audience is one sentence naming the ideal customer profile.
- emails: copy generic/public addresses (contact@, sales@, support@, hello@, \
press@) verbatim. Do not construct addresses from a person's name.
- leadership: ONLY people who work at {domain} itself -- founders, executives \
and named employees -- with the title exactly as written. Marketing pages are \
full of people who do NOT belong here: customers giving testimonials, partners, \
investors, advisors and conference speakers. If a person's title names a \
different organisation, or the quote is praising the product rather than \
speaking for the company, leave them out. Prefer founders and executives over \
junior staff. Include a linkedin_url ONLY if that exact URL appears in the \
content.
- data_confidence_score: judge how complete and unambiguous the content was. \
Use 0.9+ only when overview, audience, contacts and named people were all \
clearly present; use below 0.4 when the content was thin or mostly marketing \
copy with no concrete details.
"""

EXTRACTION_USER = """\
Company domain: {domain}

The following e-mail addresses and LinkedIn URLs were already found in the raw \
HTML by an exact-match scan. Treat them as verified and reuse them where they \
belong:
- E-mails found: {emails}
- LinkedIn profile URLs found: {linkedins}

Website content:

{corpus}
"""

FOLLOWUP_SYSTEM = """\
You are directing the second pass of a small web crawl. A first pass already \
read part of a company's website, but some information could not be found. You \
are given what is still missing, and the links from that site that have NOT \
been read yet.

Rules:
- Return ONLY URLs copied verbatim from the candidate list. Never invent a URL.
- Choose pages likely to contain the MISSING information specifically, not \
pages that merely look important. For missing people, prefer about / team / \
leadership / company / careers pages. For a missing e-mail, prefer contact / \
support / sales pages.
- Return an empty list if nothing in the list plausibly helps. Fetching \
nothing is better than fetching a page that cannot contain the answer.
- Choose at most {max_pages} URLs.
"""

FOLLOWUP_USER = """\
Company: {domain}

Already read:
{visited}

Still missing after the first pass:
{gaps}

Links not yet read (score | url | link text):
{candidates}
"""

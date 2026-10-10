# Slipstream Web-Search & Retrieval Sources — Providers, Architecture, and Sequencing

**Date:** 2026-10-09 · **Scope:** the agent harness's online-research surface
(`server/agent/tools.py` `WebResearcher` + the `web_search` / `fetch_url` tools)
and the external search/fetch/deep-research providers it could be wired to.
**Method:** source read of the current implementation; a provider survey using an
independent agentic-search benchmark (8 APIs, 100 queries, 4,000 results, LLM
judge); a direct read of Parallel's published API/pricing page; and a Gravity
Index service search. All claims carry evidence or a source note; vendor pricing
is quoted as published on 2026-10-09 and should be re-checked before purchase.
**Relationship to existing docs:** this is the *research-tooling* companion to
[`agent-harness-review.md`](agent-harness-review.md) (F2 and F6 are the two
blockers that gate every recommendation here). Design intent for the harness
lives in
[`agent-intelligence-and-harness-survey.md`](agent-intelligence-and-harness-survey.md)
(which already proposes an MCP adapter at §5.3), and engine-level priorities stay
in [`architecture-roadmap.md`](architecture-roadmap.md). Actionable items are
tracked in [`.agents/tasks.md`](../.agents/tasks.md).

## 0. The one-paragraph thesis

The harness does **not** need a cleverer search vendor; it needs its existing
search path to actually work, then a thin provider-agnostic adapter behind it.
Today `web_search` works but `fetch_url` — the tool that opens the links
`web_search` returns — is **not registered**, so the research loop dead-ends at
snippets. Once that is fixed, depth comes from the *loop* (expand → search in
parallel → fetch → dedup → rank by source credibility → synthesize), not from the
vendor. Build one adapter selected by a single env var; keep DuckDuckGo as the
zero-config default; add a fast provider (Brave/Parallel) for in-loop search, a
semantic provider (Exa) for technical queries, and an async deep-research API
(Exa /research, Parallel Task) for the multi-minute tier. MCP is the extension
surface — Slipstream as a client — but only after the loop above is real.

## 1. Current state (verified)

`WebResearcher` lives in `server/agent/tools.py:157`.

| Capability | Implementation | Evidence |
|---|---|---|
| Search | `duckduckgo_search.DDGS().text(query, max_results)`, formatted as `[title](href)` + body | `tools.py:161` |
| Fetch | `urllib.request` (Mozilla UA) → `trafilatura.extract(html, output_format="txt", include_links=True)`; regex tag-strip fallback | `tools.py:192`, `:224` |
| Cache | SQLite at `~/.slipstream/cache/web.db`, `page_cache` table, 24 h TTL, 6,000-char default cap | `tools.py:180–214` |

**Two blockers gate everything below** (from the harness review):

- **F2 — `fetch_url` is unreachable.** `execute_tool` handles it
  (`agent.py:649–652`) and `SYSTEM_PROMPT` advertises it (`agent.py:45`), but it
  is **absent from `TOOL_DEFINITIONS`** (`tools.py:304`; the 7 registered tools
  are `run_command`, `view_file`, `replace_file_content`, `write_file`, `search`,
  `web_search`, `manage_todo`). The model can never emit it, so links returned by
  `web_search` are never opened — the single most important fix for research.
- **F6 — packaging drift.** `duckduckgo_search` (`tools.py:164`), `trafilatura`
  (`tools.py:224`), and `rapidfuzz` (`tools.py:94`) are imported at runtime but
  are **not** in `install/requirements.txt` (41 lines, none of the three). A
  fresh install loses `web_search` and markdown extraction.

**MCP is named but not built.** No MCP client or server code exists in `server/`
or `install/` (grep for `mcp.server` / `mcp.client` / `ClientSession` /
`stdio_client` → zero). The design docs *propose* it —
`agent-intelligence-and-harness-survey.md:246–248` (an in-process `mcp` Python
SDK adapter) and `:383` ("Native MCP Adapter") — but nothing is implemented
today. This is the greenfield the recommendation in §6 fills.

Also present but unused by the agent: `server/agent/repo_indexer.py`
`build_repo_token_index` (MAGIC `0x534C5452`). It is a *local* retrieval index, not
a web source, but it is the natural place to store fetched-and-deduped pages
alongside the SQLite cache.

## 2. Provider landscape (2026-10-09)

Ranked by an independent agentic-search benchmark (source:
`aimultiple.com/agentic-search`, updated 2026-05-13; 8 APIs, 100 real queries,
4,000 results, LLM judge; **Agent Score = Mean Relevant × Quality**).

| Provider | Agent Score | Median latency | Best for | Entry price (as published) |
|---|---|---|---|---|
| **Brave** | **14.89** (winner) | **669 ms** (fastest) | Fast, cheap, quality-tier in-loop search | $5 / 1k requests (+$5 free credits/mo, 50 qps) |
| Firecrawl | 14.58 | 1.335 s | JS-heavy fetch/extract | Pages-based; free 500 pages; Hobby €14/mo (3k pages) |
| **Exa** | 14.39 | ~1.2 s | Technical/coding-doc semantic search, highlights, citations | $5–15 / 1k requests/pages; free $20 initial |
| Parallel Search Pro | 14.21 | **13.6 s** (slowest) | Deep/pro async research | See §3 |
| Tavily | 13.67 | 998 ms | Balanced agent search | Free 1k credits/mo |
| Parallel Search Base | 13.50 | ~2.9 s | Mid-depth async | See §3 |
| Perplexity / SerpAPI | below | — | — | — |

**The differentiator is latency, not quality.** The top four are statistically
tied on quality; only Brave is meaningfully ahead of Tavily (~1 point). But
latency varies **20×** — Brave 669 ms vs Parallel Pro 13.6 s. Because research is
multi-step, this compounds: five searches ≈ **3 s (Brave) vs ~68 s (Parallel
Pro)**. For an *in-loop* tool the model waits on, this is decisive. Exa is called
out as best on technical-documentation queries, which is exactly the coding-agent
case.

## 3. Parallel's API tiers (direct read of `parallel.ai/pricing`)

Parallel exposes several APIs, not one. Published on 2026-10-09:

| API | Price | Latency | Use |
|---|---|---|---|
| **Search API** | $0.001–$0.005 / request (10 results), free up to 5k req/mo | 200 ms–3 s, sync | In-loop fast search |
| **Extract API** | $0.001 | < 3 s | Page fetch/extract |
| **Responses API** | $0.01–$0.25 | 5–60 s, OpenAI-compatible | Cited answers |
| **Task API** | $0.005–$2.4 / request | 5 s–30 min, async | Deep research / structured |
| Monitor API | $0.003–$0.01 | — | Change tracking |
| Find All API | $0.03–$1 / match | — | Exhaustive entity search |

Pay-as-you-go; up to 5,000 requests/month free, plus "$5 in free credits per
month" and "up to $80 at signup". All SOC2. Search/Extract rate limit
600 req/min. (Parallel's own articles quote Turbo/Fast ≈ $1/1k req with 10
results at 200/700 ms median, and Basic/Advanced $5/1k.)

## 4. Gravity Index recommendation: Exa

A Gravity service search for "web-search / deep-research API for an agent harness"
recommends **Exa** (`slug: exa`, Web Search category):

- Semantic search with highlights and citations, purpose-built for AI agents;
  strongest on technical-documentation queries.
- Free **$20 initial credits** (~2,800 searches) + $10/mo.
- Env var `EXA_API_KEY`; Python SDK `exa-py` (`pip install exa-py`); docs at
  `https://exa.ai/docs/reference/search-api-guide`.
- Browsing the "Web Search" category returns Exa as the only listed service
  (quality_score 50); "Search" is a separate category.

The install guidance Gravity returned sketches exactly the shape recommended
below: a `ResearchEngine` using `exa.search(query, num_results,
use_autoprompt=True, highlights=True)` and `exa.get_contents([url])`.

## 5. Proposed architecture — three tiers behind one adapter

**Do not pick one vendor.** Put a provider-agnostic adapter behind
`WebResearcher`, selected by a single env var
(`SLIPSTREAM_SEARCH_PROVIDER=brave|exa|parallel|tavily|searxng|ddg`), normalizing
all results to the existing `[title](href)` + body shape. Keep DuckDuckGo as the
zero-config default so nothing regresses when no key is set.

| Tier | In-loop? | Preferred provider | Fallback |
|---|---|---|---|
| **1 — Search** | yes, fast/cheap | **Brave** (fastest + cheapest + quality-tier); **Parallel Search Fast** ($1/1k, 200 ms) | DDG (default) |
| **2 — Fetch / Extract** | yes | keep local `trafilatura`; add Exa `/contents`, Parallel Extract ($0.001), Firecrawl for JS-heavy | regex strip (exists) |
| **3 — Deep research** | **no** — async, multi-minute tool | **Exa `/research`**, **Parallel Task API**, Perplexity Sonar | — |

Route **Exa** for technical/coding queries, **Brave** when a query is generic and
speed matters. Tier 3 must be an *out-of-loop* tool (the model dispatches it and
polls), because 13.6 s+ inside the turn is unusable.

**Depth comes from the loop, not the vendor.** Implement, on top of the existing
SQLite cache: query expansion → parallel search fan-out → fetch → dedup →
source-credibility ranking → synthesis. A tier-1 provider with a good loop beats a
better provider with a naive loop.

**MCP is the extension surface.** Two directions, both cheap once the adapter
exists:
- Slipstream as MCP **client** — plug in Brave/Exa/Tavily/Firecrawl/Parallel MCP
  servers with no code change. This is the "enable services/tools/MCPs" path and
  reuses the adapter's normalization.
- Slipstream as MCP **server** — expose `view_file` / `search` / shell to other
  agents.

## 6. Recommendation (ranked) and regression risk

Each is labeled against the "no quality or speed regression" constraint.

| # | Action | Risk |
|---|---|---|
| R1 | **Fix F2 (register `fetch_url`)** and **F6 (pin the 3 deps)** first | F2 changes the request payload → **A/B required** on the 145-item scorecard; F6 is a **pure win** |
| R2 | Wire **Exa** as the first real provider (best for a coding agent; Gravity-listed; free credits), Brave as the fast/cheap default, Parallel for deep research | Provider swaps don't touch sampling but **change tool output content** → validate against the scorecard |
| R3 | Build the tiered, provider-agnostic adapter behind `WebResearcher` | Behavior-preserving when the default stays DDG |
| R4 | Implement the research loop (expand → fan-out → dedup → rank) over the existing cache | Pure win at fixed provider |
| R5 | Add a thin **MCP client** adapter (as designed in the harness survey §5) | New surface; no change to existing tools |
| R6 | Route **technical queries** to Exa, **generic/fast** to Brave | Content-dependent → scorecard |

**Avoid direct Google SERP scraping** — ToS-risky and fragile. The sanctioned
Google Programmable Search JSON API is capped and low-quality for agents; Brave's
independent index is the better fast option and Exa is better for depth.

## 7. Non-goals

- **Not** replacing the local `trafilatura` fetch path — it is cheap, private, and
  already cached; external Extract is an *addition* for JS-heavy or blocked pages.
- **Not** vendoring a search index — build on providers, not a crawler.
- **Not** putting Tier-3 deep research in the interactive loop.

## 8. Status

| # | Item | Gate | Status |
|---|---|---|---|
| R1 | Register `fetch_url`, pin deps (F2/F6) | 145-item scorecard for F2 | ☐ Open |
| R2 | First real provider (Exa + Brave + Parallel) | scorecard (tool content) | ☐ Open |
| R3 | Provider-agnostic adapter | none (DDG default preserved) | ☐ Open |
| R4 | Research loop over SQLite cache | none | ☐ Open |
| R5 | MCP client adapter | new surface review | ☐ Open |
| R6 | Query routing (Exa vs Brave) | scorecard (tool content) | ☐ Open |

**Constraint (non-negotiable):** registering `fetch_url` (R1) changes the request
payload sent to the engine, and any provider swap (R2/R6) changes the *content* of
tool results — both require an A/B on the 145-item scorecard before shipping.
Adding a provider adapter that leaves DDG as the default (R3/R4) is
behavior-preserving and cannot trade quality or speed.

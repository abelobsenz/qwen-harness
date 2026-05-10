# Scientific agent skills

Reference for the research-flavoured tools the qwen agent ships with. Each
"skill" is a thin function in `scripts/agent_tools.py` plus its OpenAI-shape
schema in `TOOLS`. They're called from chat (`qwen_ui.py`) and the CLI agent
(`agent.py`) through a single `CachedDispatcher` so cross-turn caching and
intra-turn dedup are identical for both runtimes.

## Skill inventory

### Web

| tool | what it returns | when to reach for it |
| - | - | - |
| `web_search` | DDG snippets. Built-in near-duplicate dedup (cosine ≥ 0.97 or high lexical overlap against prior queries) + `site:` / `filetype:` filters | First-pass discovery |
| `web_outline` | Headings (h1–h4) + a few words per heading | Decide *whether* to fetch a page; pinpoint the right section before paying full-text cost |
| `web_fetch` | Article-aware extraction with **smart truncation** (keeps both ends so abstract + conclusion both survive) | Once you know a page is worth reading. `head_only=True` for cheap |

The near-duplicate detector lives in `_check_search_duplicate`; query strings
are L2-normalized BGE-small vectors compared against the in-process memo of
recent queries. It also has a lexical-overlap fallback so obvious
reformulations that cosine misses still return a near-duplicate notice instead
of burning another live search.

### Papers / citations

| tool | source | notes |
| - | - | - |
| `arxiv_search` | `export.arxiv.org` Atom API | Sort by relevance / date; structured metadata |
| `arxiv_fetch` | `export.arxiv.org` (abstract), `arxiv.org/html/<id>` (HTML5), `arxiv.org/pdf/<id>` (PDF) | Always start with `what="abstract"`; only step up when you need the body |
| `doi_resolve` | `doi.org` content negotiation (CSL-JSON) | Bypasses publisher landing pages; emits CSL-style citation block |
| `pdf_extract` | local file or HTTPS URL via `pypdf` | `pages="1-3"` is the typical "abstract + intro" slice |

### Repos / data

| tool | what | typical use |
| - | - | - |
| `github_repo` | repo info / dir listing / file read / readme via api.github.com | Always prefer over `web_fetch` on github.com URLs |
| `csv_summary` | shape / col types / preview / numeric describe (no pandas) | Quick "what's in this file?" without firing up `python_run` |
| `now` | `zoneinfo`-aware datetime | Whenever the user says "today / now / this week" |

### Memory (cross-session, semantic)

| tool | what |
| - | - |
| `memory_save` / `memory_get` / `memory_search` / `memory_list` / `memory_delete` | SQLite + BGE-small embeddings; hybrid semantic + lexical scoring with a small bonus for repeated keyword hits |

## Cache & dedup semantics

Both runtimes share `agent_tools.CachedDispatcher`, which tracks two stores:

- `fs_cache`: keyed by `(fn, args_json)`. Invalidated by *any* write tool
  (`write_file`, `edit_file`, `bash`, `apply_patch`, `append_finding`,
  `write_file_verified`, `notebook_edit`).
- `web_cache`: keyed identically; never invalidated within a session.

Identical (`fn`, `args`) pairs that arrive in one turn are deduped to a single
dispatch. Distinct calls run in a `ThreadPoolExecutor` with `max_workers=6`.
That's how the agent gets parallel tool execution for free.

The **chat UI** holds one `CachedDispatcher` per `session_id` (LRU bound at
32 sessions) so two browser tabs never poison each other's caches. The
**CLI agent** holds one for the lifetime of the Python process.

## LangGraph and Pydantic AI integration

`scripts/agent_adapters.py` re-exports the registry through three converters:

```python
from agent_adapters import (
    to_pydantic_ai_tools,   # for pydantic_ai.Agent(tools=…)
    to_langgraph_tools,     # langchain_core.StructuredTool list → ToolNode
    to_openai_tools,        # passthrough — for any OpenAI-compatible runtime
)
```

Both adapters thread a single `CachedDispatcher` through the closures, so
external runtimes get the exact same caching the in-house agent does. Use
`names=[...]` on either to subset the registry; pass an explicit
`cdisp=…` to share a cache across multiple Agent invocations.

### When to use each runtime

- Stay on `agent.py` when you want the existing harness (compaction, exit
  nudges, write-tracking gate). It's the most opinionated and the one we
  test under load.
- Reach for **Pydantic AI** when you want validated structured outputs and
  the per-tool retry semantics it gives you for free. Especially good for
  scientific workflows where each tool's return type matters.
- Reach for **LangGraph** when you actually need an explicit graph: a
  multi-agent supervisor / worker setup, or branching tool execution where
  some tools should run conditionally on a state field.

## What's deliberately *not* here

- No web crawling. We have one-page `web_fetch` and that's intentional —
  multi-page crawls bloat context for very little marginal information.
  Reach for `subagent_implement` / `explore` if you need bulk reading.
- No "agentic search" wrapper that does search→fetch→summarize in one tool.
  We tried it; the agent gets better outcomes when it sees the search
  results and decides what to fetch, especially with the dedup + outline
  primitives.
- No Wolfram Alpha / scholarly databases beyond arxiv + DOI. Add them if
  you have API keys; the pattern is in `arxiv_search`.

## Hot path decisions a model should know

1. **Don't `web_fetch` until you've outlined.** `web_outline` is ~30× faster
   and tells you whether the body has what you want.
2. **For arxiv, use `arxiv_*`, not `web_*`.** Cleaner, structured, faster.
3. **For GitHub, use `github_repo`, not `web_*`.** Direct API, no HTML noise.
4. **For PDFs, use `pdf_extract` with a page range.** Page 1–3 is usually
   the abstract + intro you want.
5. **`csv_summary` before `python_run` + pandas** for the common "what's in
   this CSV?" question.
6. **`now()` whenever the user references time.** The system prompt has the
   date but it gets stale across midnight in long-running sessions.

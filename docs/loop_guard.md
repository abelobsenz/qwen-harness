# loop_guard — proxy-level repetition detector

## Why it exists

`dflash_mlx` runs **pure greedy argmax** decoding (no temperature, top_p,
repetition_penalty). Greedy decoding is mathematically guaranteed to fall
into pathological loops once it enters a low-loss n-gram cycle. The
canonical example was a Qwen3.6 user prompt where the model produced 14 KB
of `"I will use make_table now. Then the Mermaid code. I will not use any
other tools. I will just output the text. Wait, I need to use…"` repeated
verbatim.

We can't add full sampling without re-engineering the speculative decoding
acceptance protocol (draft and target need consistent sampling). The
proxy is the right layer for a generic safety net: it sees every chat
completion, it's stateless, and the cost (≤ 5 ms) is negligible vs real
model generation (500-15000 ms).

## Architecture

```
client (agent.py / chat.py / qwen_ui.py)
  ↓ stream=False or stream=True, OpenAI shape
qwen_proxy:8000  ←  loop_guard runs here
  ↓ stream=True (always — even for stream=False clients)
dflash-serve:8002  ←  the actual model
```

For `stream=False` requests, the proxy now drives upstream in **streaming
mode under the hood**, so the guard can early-abort upstream generation
when a loop is detected. Without that, a 14 KB loop would burn ~5 KB of
decoded tokens before we could even see the response.

## Detection algorithm

Two complementary detectors over a sliding 600-char window:

1. **Suffix-repeat**: a literal byte chunk of length ≥ 24 chars repeats
   ≥ 4× consecutively at the tail. Tries successive lengths from longest
   down to the minimum; first match wins.

2. **N-gram churn**: ratio of distinct 6-grams to total 6-grams over the
   trailing 600 chars. Below 0.45 = the model is recycling phrases.

### Trigger logic (combined)

| suffix-repeat | low-churn | trigger? | reason       |
|---------------|-----------|----------|--------------|
| yes           | yes       | YES      | combined     |
| yes (dominant¹)| no       | YES      | suffix-dominant |
| yes (extreme²) | no       | YES      | suffix-extreme |
| no            | yes       | no       | (legitimate prose) |
| no            | no        | no       |              |

¹ "Dominant" = the literal repeats account for ≥ 80% of the text (so the
  model produced almost no novel content).

² "Extreme" = a chunk ≥ 30 chars repeats ≥ 8× consecutively (handles
  short loops where the churn detector hasn't filled its window yet).

The combined-AND requirement is what eliminates false positives on
*structurally* repetitive but content-varied outputs (XML tool-call
fanouts, markdown sections, JSON arrays). The escape conditions catch
unambiguous loops that one detector alone would miss.

## Streaming model

`StreamingLoopGuard` accumulates a sliding window (`ngram_window + 1024`
chars), runs the detector every `check_every` chars (default 128), and
exposes `finalize()` for the end-of-stream final check.

The proxy uses both:

- Mid-stream: `guard.observe(chunk)` after each SSE token. On trigger it
  closes the upstream HTTP socket (which makes dflash-serve's generator
  raise GeneratorExit and stop generation), then emits a synthetic
  `finish_reason=stop` frame plus a `[loop-guard: …]` content marker.

- End-of-stream: `guard.finalize()` runs a final check on the full
  buffered tail. Catches loops that develop in the trailing < check_every
  chars and loops shorter than check_every total.

## Tunables (all env-overridable)

| env var                       | default | meaning                          |
|-------------------------------|---------|----------------------------------|
| `LOOP_GUARD_DISABLE`          | 0       | global kill-switch (=1 disables) |
| `LOOP_GUARD_SUFFIX_MIN_LEN`   | 24      | min chunk length for suffix-rep  |
| `LOOP_GUARD_SUFFIX_REPEATS`   | 4       | min consecutive reps for suffix  |
| `LOOP_GUARD_NGRAM_WINDOW`     | 600     | n-gram detector window (chars)   |
| `LOOP_GUARD_NGRAM_N`          | 6       | n-gram size (chars)              |
| `LOOP_GUARD_NGRAM_FLOOR`      | 0.45    | distinct/total threshold         |
| `LOOP_GUARD_MIN_TEXT`         | 200     | skip checks below this length    |
| `QWEN_PROXY_COMPACT_SCHEMA`   | 1       | compact tool blurb (=0 = verbose)|
| `QWEN_PROXY_MAX_BODY_MB`      | 50      | max request body MB (=0 disables)|

## Test coverage

Detector / proxy:

| suite                                    | what it covers                       |
|------------------------------------------|--------------------------------------|
| `loop_guard.py` self-tests (`-m`)        | unit: 11 hand-picked cases           |
| `test_loop_guard_adversarial.py`         | 16 hand-picked false-positive traps  |
| `test_loop_guard_fuzz.py`                | 700 random samples (loops/clean/mix) |
| `test_loop_guard_proxy.py`               | end-to-end SSE + non-stream proxy    |
| `test_loop_guard_disabled.py`            | `LOOP_GUARD_DISABLE=1` byte-equiv    |
| `test_proxy_perf.py`                     | < 5 ms absolute overhead             |
| `test_proxy_tool_call.py`                | tool_call/<think> parsing preserved  |
| `test_proxy_concurrent.py`               | no races/cross-talk under 20× load   |
| `test_proxy_memory.py`                   | 1000-request RSS bound               |
| `test_proxy_long_stream_memory.py`       | 30-cancel RSS bound                  |
| `test_proxy_backpressure.py`             | client-disconnect propagates upstream|
| `test_proxy_routing_audit.py`            | every LLM call goes via proxy        |

Marker contract + client surfaces:

| suite                                    | what it covers                       |
|------------------------------------------|--------------------------------------|
| `test_loop_guard_marker.py`              | shared helpers (19 cases)            |
| `test_agent_loop_guard_nudge.py`         | agent CLI integration (7 cases)      |
| `test_agent_graph_loop_guard.py`         | graph integration (4 cases)          |
| `test_qwen_ui_loop_guard.py`             | JS↔Python cross-language consistency |

Runtime patch + bench:

| suite                                    | what it covers                       |
|------------------------------------------|--------------------------------------|
| `test_runtime_patch.py`                  | `apply_repetition_penalty` (14)      |
| `test_bench_rep_penalty.py`              | bench harness smoke + streaming      |
| `test_agent_compaction.py`               | `maybe_compact` token + RSS shrink   |

Run all (recommended — uses the tiered runner):
```sh
scripts/run_tests.py            # fast tier (default, ~12s)
scripts/run_tests.py --slow     # slow tier (~100s — sustained-load)
scripts/run_tests.py --all      # everything
scripts/run_tests.py --list     # show classification
```

The runner classifies tests by filename pattern; `test_proxy_long_*` is
the slow tier today (sustained-duration memory tests). Add new patterns
to `SLOW_PATTERNS` at the top of `scripts/run_tests.py` when long-running
tests get added.

Inline equivalent (no runner):
`for t in scripts/test_*.py; do venv/bin/python "$t" || break; done`

## Tool schema compaction (related win)

Pre-fix the proxy injected 27 KB / 6,728 tokens of verbose JSON tool
schemas into every system prompt. With `_compact_schema` +
`_compact_param_descriptions`, schemas are rendered as Python-style
signatures with per-parameter descriptions on indented lines:

```
- web_search(query: str, max_results?: int, site?: str): Search the web…
    query=Search query — keywords, NOT a question.
    max_results=Max results to return (default 5, max 25).
    site=Restrict to a domain, e.g. 'arxiv.org', 'github.com'.
```

Net: **26.9 KB → 22.1 KB (-17%, -1,210 tokens per request)**. Round 30
recovered the per-parameter descriptions that an earlier sig-only form
had stripped — the model needs that guidance for tools where parameter
semantics matter (memory_*, web_search, …). Disable with
`QWEN_PROXY_COMPACT_SCHEMA=0` to fall back to verbose JSON.

## Detection contract (the marker)

The proxy emits one of two specific marker shapes when it aborts:

  - non-stream: `[loop-guard: <reason> (<detail>) — output stopped early]`
  - streaming:  `[loop-guard: aborted (<reason>) — the model fell into a repetition loop. Try rephrasing or asking a more specific question.]`

The marker itself is plaintext that ends up in the assistant's
`content` (intentionally — it's both human-readable in the UI and
model-readable for the next turn's nudge). Downstream detectors
identify a real abort by requiring **both**:

  1. The literal substring `[loop-guard:`
  2. One of the proxy's specific suffix phrases:
     `output stopped early` OR `fell into a repetition loop`

A bare-substring check would false-positive whenever the model
legitimately mentions the marker — e.g. answering "how does the loop
guard work?" or echoing a grep result that quotes the codebase. Round
15 surfaced this as a real bug in `agent.py`; Rounds 16-18 propagated
the fix to `agent_graph.py` and the JS in `qwen_ui_static/app.js`.

The contract is centralized in `scripts/loop_guard_marker.py`:

```python
from loop_guard_marker import (
    is_proxy_abort_marker,   # bool — substring + suffix
    extract_reason,          # captures hyphenated reason names
    harness_nudge_message,   # builds the [HARNESS] user msg
)
```

If the proxy's suffix wording ever changes, update both
`qwen_proxy.py`'s emit sites AND the JS regex in `app.js`. The cross-
language consistency test (`scripts/test_qwen_ui_loop_guard.py`)
catches drift between the two at test time.

## Detection-point inventory

| call site                          | language | role                                |
|------------------------------------|----------|-------------------------------------|
| `qwen_proxy.py`                    | Python   | emits the marker on abort           |
| `loop_guard_marker.py`             | Python   | canonical detector + nudge factory  |
| `agent.py:_check_and_handle_loop_guard` | Python | CLI surfacing, injects HARNESS    |
| `agent_graph.py:_run_node`         | Python   | per-node graph surfacing            |
| `qwen_ui.py` (chat loop)           | Python   | web-UI server-side nudge injection  |
| `qwen_ui_static/app.js`            | JS       | bubble warning + toast              |

Four client surfaces, one shared detector, one cross-language
contract. Adding a fifth surface (e.g. an evaluation harness or
dashboard) means importing the existing helpers — no detection logic
re-implementation.

## Agent-side surfacing (CLI loop)

`scripts/agent.py:_check_and_handle_loop_guard` is called at the top
of every `step()` so it sees every assistant message regardless of
whether the response also carried tool_calls. When `is_proxy_abort_marker`
returns true, the agent prints a yellow `[loop-guard fired: <reason>
— injecting course-correction nudge]` notice and appends a `[HARNESS]`
user message:

> Your previous response was cut off by the proxy's loop guard …
> Do NOT resume that line of reasoning. Step back, take a different
> angle …

Without this nudge the model has no signal that its last turn was
truncated, and often resumes the loop. Single-fire per top-level user
query; resets in `run_query`.

`scripts/agent_graph.py:_run_node` does the analogous thing per node
(single-fire per node, since each node is a separate sub-conversation).

`scripts/qwen_ui_static/app.js:appendMessageDelta` adds a `.has-loop-
guard-abort` class to the bubble (CSS shows a warning callout) and
fires a "warn" toast.

Tests:
- `test_loop_guard_marker.py` — direct helpers (19 cases)
- `test_agent_loop_guard_nudge.py` — agent CLI integration (7 cases)
- `test_agent_graph_loop_guard.py` — graph integration (4 cases)
- `test_qwen_ui_loop_guard.py` — JS↔Python cross-language consistency

## Memory bounds (empirically verified)

Three pathologies, all proven bounded via subprocess RSS measurement
(`ps -o rss=`):

| test                               | pathology                                    | RSS growth |
|------------------------------------|----------------------------------------------|------------|
| `test_proxy_memory.py`             | 1000 short requests                          | +0.1 MB    |
| `test_agent_compaction.py`         | 50 agent compaction cycles                   | +7.5 MB ¹  |
| `test_proxy_long_stream_memory.py` | 30 long-stream cancels                       | +0.1 MB    |
| `test_proxy_long_idle_memory.py`   | 90 reqs over 90s @ 1 r/s                     | +0.1 MB    |
| `test_proxy_long_midrate.py`       | 100k+ reqs over 30s @ 3,351 r/s × 4 workers  | +0.4 MB    |

¹ The +7.5 MB on agent compaction is allocator fragmentation from
per-cycle session regrowth, not a leak — sub-linear and well under
gate. The proxy paths are essentially silent.

The five tests cover orthogonal patterns: burst-load (test_proxy_memory),
heavy-cycle (test_agent_compaction), partial-stream (test_proxy_long_
stream_memory), low-rate-long-duration (test_proxy_long_idle_memory),
and **sustained moderate concurrency** (test_proxy_long_midrate — the
multi-tab interactive user pattern). A leak that survives all five
would have to be specific to some shape NONE of them exercises —
extremely unlikely.

Headline number from the most recent test: the proxy sustained
**3,351 req/sec across 4 concurrent workers for 30 seconds**
(100,538 total requests) with +0.4 MB RSS growth and peak 6 threads.
The framework's per-connection threading and the proxy's per-request
state both scale cleanly.

## Bench harness (real-model A/B)

`scripts/bench_rep_penalty.py` runs the canonical "make_table" loop
prompt N times with vs without the rep-penalty patch installed,
records TPS / TTFT / loop-presence, and emits a side-by-side diff:

```sh
python scripts/bench_rep_penalty.py --label baseline --stream
DFLASH_REP_PENALTY=0.05 python scripts/bench_rep_penalty.py --label rep05 --stream
python scripts/bench_rep_penalty.py --diff baseline rep05
```

`--stream` adds time-to-first-token measurement (matters for perceived
interactive speed; can diverge from steady-state TPS).

## Open follow-ups

- **Z-algorithm or suffix-array** for guaranteed O(n) longest-repeat. The
  current scan is O(n²) worst-case (~2.6 M ops at the 1624-char buffer).
  Acceptable today; revisit if we raise the window.
- **Token-level (BPE) detection** for non-English. Per-codepoint already
  works for CJK + Arabic + emoji, but Tibetan/Myanmar combining marks may
  benefit from explicit token-boundary awareness.
- **Runtime-level repetition penalty** — research artifact landed at
  `scripts/runtime_patch.py`. Pure-function library + `RepetitionContext`
  that integrators can install in `dflash_mlx/runtime.py:generate_*_once`.
  Off by default (`DFLASH_REP_PENALTY=0.0`). Not yet wired into the
  daemon — the autonomous environment can't validate end-to-end against
  a real model. Tests in `scripts/test_runtime_patch.py` (14 cases on
  synthetic logits). Apply during a future real-model session and A/B
  TPS / quality before promoting from "research artifact" to "default
  on."

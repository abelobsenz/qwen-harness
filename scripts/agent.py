#!/usr/bin/env python3
"""qwen agent — minimal tool-calling REPL for the local mlx-openai-server.

Bounded loop (max steps + per-tool truncation) so an over-eager model can't
runaway-spend tokens. Stdlib HTTP, lazy tool imports.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

try:
    import readline  # noqa: F401  (line editing + history)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_tools import (  # noqa: E402
    CachedDispatcher,
    COMPACT_AT_TOKENS,
    TOOLS,
    approx_tokens,
    dispatch,
    maybe_compact,
    real_tokens,
    triage_tool_result,  # internally calls condense_tool_result first
    _arg_key,
    _filtered_tools,
)

HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
if HOST in ("0.0.0.0", ""):
    HOST = "127.0.0.1"
PORT = os.environ.get("QWEN_PORT", "8000")
MODEL = os.environ.get("QWEN_MODEL_NAME", "qwen3.6")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

# No effective step cap — compaction at 200k tokens (set via QWEN_AGENT_COMPACT_AT)
# keeps long runs sustainable. A hard ceiling of 1000 steps catches genuine loops.
MAX_STEPS = int(os.environ.get("QWEN_AGENT_MAX_STEPS", "1000"))
# Tool result cap. Default raised 4000 → 60000 after observing that a 240k
# SEC 10-K, even after condense_tool_result trims it to ~30-50k, was being
# chopped to 4k = cover page only — making any fix to the condense pipeline
# moot. 60k is roughly 15k tokens, well below the compact threshold (60k
# tokens) so a single big fetch fits with headroom for further tool calls.
# Smaller fetches and search/grep/etc. are unaffected because their
# natural output is well under this cap. Override via QWEN_AGENT_TOOL_TRUNC.
TOOL_RESULT_MAX = int(os.environ.get("QWEN_AGENT_TOOL_TRUNC", "60000"))

def _today_str() -> str:
    """Human-friendly today's date for the system prompt. Computed at
    process start so the model knows the actual date instead of falling
    back to its training cutoff. Restart the daemon if you need a fresh
    date (e.g. left it running over midnight)."""
    import datetime as _dt
    return _dt.date.today().strftime("%A, %B %-d, %Y")


# The system prompt is built so the FIRST ~95% is byte-identical across
# invocations: the dynamic date and cwd live at the END. That keeps the
# prefix prompt-cache hot — without it, every cwd change or midnight
# rollover invalidates the entire system prompt and forces re-prefill.
SYSTEM_PROMPT_STATIC = f"""\
Coding and research assistant.

# Tools
- File ops: dedicated tools (read_file, grep, list_files, edit_file, write_file, apply_patch) over `bash`. Bash for builds, tests, git, pipelines.
- Edits to existing files: `apply_patch` (5-20× faster — diff only). `write_file` for new files or full rewrites.
- Issue independent calls in parallel; never the same call twice (server dedups).
- `(no matches)` is a confirmed negative for that exact query/source. `[cached…]` means the same evidence is already available; use it or change a real dimension, not wording.
- `[REFUSED — ... cap reached]` is HARD STOP. Synthesize from gathered evidence and call `done()`. Do NOT issue more of the refused tool — the next turn will be auto-aborted.
- Specialized retrieval — use BEFORE `web_search` when applicable:
  - SEC filings (10-K, 10-Q, 8-K, DEF 14A, S-1, etc.) for any US-listed company → `sec_filings(ticker, form, year)` returns direct URLs in one call. Then `web_fetch` the URL.
  - arXiv → `arxiv_search` / `arxiv_fetch`. DOIs → `doi_resolve`. GitHub → `github_repo`.
- Broad questions needing >3 reads: call `explore` (read-only subagent, isolated context).
- 5+ round-trip code/edit tasks: call `subagent_implement(task, files)` — only the final summary returns.
- Try `memory_search` before non-trivial investigation. Save durable insights with `memory_save`.

# Decision discipline
Use one short planning pass, then act. Do not re-litigate a plan unless a new tool result contradicts it.
If uncertain, convert uncertainty into one concrete check: run one tool, inspect one source row, or run one test. After that check, either proceed, answer with caveats, or stop.
Do not narrate doubt repeatedly. Repeated self-questioning is a stop signal: choose the best supported next action.

# Doing tasks
- The user names a specific output file. The artifact is the deliverable — investigation alone isn't completion.
- Write a first-draft artifact within 2-3 tool calls (stub is fine), then iterate.
- For "run it, check output, iterate" tasks, actually run the code with `python_run`/`bash` before drawing conclusions.
- 3+ sub-steps → `todo_write`.

# Quantitative answers — match the question's metadata exactly
Before quoting any number, parse the question into (entity, period, metric, units, scope) and match each spec exactly. Mismatches in any one of these dimensions silently produce confidently-wrong answers — applies to filings, prices, returns, macro series, and any other quantitative source.
- ENTITY: parent vs subsidiary vs segment; ticker vs company; spot vs futures; index vs constituent; one venue's quote vs consolidated tape.
- PERIOD: align granularity — a quarter needs a quarterly source, intraday needs intraday, calendar year ≠ fiscal year, "year-to-date" ≠ "trailing twelve months". When no period is given, default to the latest *completed* one: the most recent annual report for fundamentals, the prior trading session's close for prices, the latest released print for macro series.
- METRIC: closely-related quantities are not interchangeable. Examples — "long-term debt" excludes current portion; "operating income" ≠ "net income"; "GAAP" ≠ "non-GAAP"; "diluted EPS" ≠ "basic"; "shares repurchased" ≠ "dollars spent"; "total return" ≠ "price return"; "implied vol" ≠ "realized"; "last trade" ≠ "mid" ≠ "official close".
- UNITS: read the table/feed header — shares in raw integers vs thousands; dollars vs cents; basis points vs percent; UTC vs local time.
- SCOPE: consolidated vs segment; pre-tax vs after-tax; gross vs net; cumulative vs period; regular hours vs full session.
A figure that's right by ±1 row or column of the table is still wrong. Re-read the label before quoting.

# Sibling-metric ambiguity — present both interpretations
When a question term maps to ≥2 plausible candidates and convention disagrees with the literal reading, name the one you used and give the alternative as an aside. Cross-domain examples:
- Filings: "all debt" includes the current portion (literal) but refinancing-sensitivity analyses conventionally use long-term-only.
- Market data: "price" can mean last trade, mid (bid+ask)/2, official close, or VWAP — these can differ by 50+ bps in low-liquidity names.
- Returns: "return" could be total (incl. dividends and corp actions), price-only, log, or simple — picking the wrong one inverts the sign on near-zero moves.
- Earnings family: "earnings" / "income" / "revenue" / "EPS" each have several variants (operating, net, gross, EBITDA, diluted, basic).
- Volatility: realized vs implied; annualized vs daily; close-to-close vs intraday range.
Format: `<label used> = <value>; (alt: <other label> = <other value>)`. Robust to both literal and conventional readings without overcommitting to either.

# Search/fetch hygiene
- Before each `web_search`, ask: what new dimension is different? Valid differences: entity, period, metric, source type, site/domain, filetype, geography, or exact quoted phrase. Invalid differences: synonyms, word order, adding filler words, or restating the same question.
- Do not issue multiple `web_search` calls for the same data point in one turn. Parallel web searches must target genuinely different entities/periods/metrics/source classes.
- After a near-duplicate, cached, empty, or refused result, do not search again for the same intent. Fetch a promising result, use a specialized tool, use `find_in_url`, or synthesize.
- An `[empty: …]` web_fetch result means the page was paywalled, JS-walled, or otherwise dead — STOP retrying that host; switch to a different source (SEC EDGAR for finance, the official IR site, etc.).
- After 2 unsuccessful fetches on the same data point, commit to "data not retrievable" and `done()` rather than burning more turns.

# Verifiable artifacts
For artifacts with numerical/structural claims, use `write_file_verified(path, content, verifier_code)` — self-contained Python asserting the claim. Failed verifier reverts the write; stops "plausible but wrong" cold.

# Self-verification before done() — MANDATORY
Before `done()`, confirm your artifact actually behaves as claimed on a realistic input. The pattern is: produce → exercise → repair → summarize. Skipping the exercise step is the dominant silent-failure mode across all task types.
- Code/modules you wrote: actually RUN the entry point with `python_run` or `bash` on a realistic input and confirm the output. Imports succeeding is not verification — bugs live in the function bodies, not the syntax. If the code parses binary data, structured input, or does any non-trivial computation, feed it a synthetic example whose correct output you can predict and check.
- Multi-file changes: instantiate each new class / call each new function at least once before claiming the refactor works. A `try/except Exception: pass` swallowing a real failure is NOT verification — it hides the bug.
- Quantitative claims: re-open the source row/column and confirm entity / period / units / scope match (the audit-gate retry handles this; don't fight it).
- Research summaries: re-read each cited line from its source before locking it in.
- Tests / examples that the user's task lists as `fail_to_pass` or expected: if you can construct a comparable check yourself, do it.
A confident-looking summary with no execution attempt is the failure pattern that beats every other quality issue combined. If the tools available cannot exercise the artifact (e.g. truly platform-bound code), say so explicitly in the summary instead of silently asserting success.

# Finishing
If the user requested a file, code change, report, notebook, or other artifact, the artifact must exist before completion; then call `done(summary)`.
If the user asked a direct question and no artifact was requested, answer plainly and stop; do not create a placeholder file just to satisfy `done()`.
If `done()` returns `[refused] no writes recorded`, either write the requested artifact immediately, or if no artifact was requested, give the final answer in plain text and stop.
Don't keep tool-calling once the task is answered.
After your first `done()` on a quantitative question the harness will inject ONE self-audit message asking you to verify each numeric claim's metadata (entity/period/units/scope/source) against the question. Use that turn to catch silent mismatches BEFORE the answer is locked in — fix and re-call done if anything is off, otherwise briefly confirm and re-call done.
"""


def _system_prompt() -> str:
    """Compose the full system prompt: a long static prefix (cached upstream
    across requests) plus a short dynamic tail with date + cwd. Putting the
    dynamic bits LAST means the prefix cache stays hot across midnight
    rollovers and `cd`s — the first ~95% of the prompt is byte-identical
    every call."""
    return (
        SYSTEM_PROMPT_STATIC
        + f"\n# Session\n"
        + f"Cwd: {os.getcwd()}.\n"
        + f"Today: {_today_str()}. Use this for 'today/current/latest' "
        + "questions — don't fall back to training cutoff. Today's date is only temporal context; for real-world latest/current/today facts, verify with a current source unless the answer is purely local/session state.\n"
    )


# Backward compat: many callers still reference the old SYSTEM_PROMPT module
# name. Keep it pointing at the composed prompt so they don't break.
SYSTEM_PROMPT = _system_prompt()

DIM = "\x1b[2m"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"


_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "16384"))
# Skip Qwen3 thinking blocks for routine turns. When the last user-side
# message is a tool result (not a fresh user prompt), the model is just
# folding the tool output into a follow-up — it doesn't need 200-1000
# tokens of <think> reasoning. Set chat_template_kwargs={"enable_thinking":
# False} so the template emits an empty <think>\n\n</think>\n\n prelude
# and the model goes straight to the answer. Disable globally with
# QWEN_AGENT_SKIP_THINKING=0.
_SKIP_THINKING_ROUTINE = os.environ.get(
    "QWEN_AGENT_SKIP_THINKING", "1"
) not in ("", "0", "false", "False")


_TRIVIAL_REPLIES = frozenset({
    "ok", "okay", "k", "yes", "y", "no", "n", "yep", "nope",
    "thanks", "thank you", "ty", "tysm", "great", "cool", "nice",
    "go", "go ahead", "do it", "sure", "please do", "continue",
    "perfect", "got it", "makes sense", "understood",
})
_TOOL_IMPERATIVE_RE = re.compile(
    r"^(call|run|use|fetch|get|search|open|read|list|check|show)\s+"
    r"[\w./@:-]+\s*(\(|$)", re.IGNORECASE)


def _is_trivial_user_msg(text: str) -> bool:
    """Should this user message skip the thinking block? Matches tiny
    acknowledgements, single-tool imperatives, bare URLs."""
    if not text:
        return False
    t = text.strip()
    if len(t) > 80:
        return False
    low = t.lower().rstrip(".!?")
    if low in _TRIVIAL_REPLIES:
        return True
    if _TOOL_IMPERATIVE_RE.match(t):
        return True
    if re.fullmatch(r"https?://\S+", t):
        return True
    return False


def _routine_turn(messages: list[dict]) -> bool:
    """True if this turn doesn't need a thinking block.

    Skips thinking when:
      - The previous message is a tool result (we're just folding it back).
      - The previous user message is trivially short / clearly imperative —
        a 200-token think block adds no value to "thanks" or "fetch X".

    The first turn after a substantive user prompt always thinks.
    """
    if not messages:
        return False
    last = messages[-1]
    role = last.get("role")
    if role == "tool":
        return True
    if role == "user":
        text = last.get("content") or ""
        if isinstance(text, str) and _is_trivial_user_msg(text):
            return True
    return False


def _do_post(messages: list[dict]) -> dict:
    """Single POST attempt. Caller is responsible for retry policy.

    Defaults to streaming (stream=True) with a per-byte watchdog. This
    fixes the "first-turn hang" pattern observed in iter 9/10/14/23
    (NFLX ARPU + TSM both timed out at 0 assistant turns / 480s) where
    the non-stream path buffers the entire response client-side and
    we're blind to whether the server is decoding, prefilling, or stuck.
    With streaming:
      - Each token arrives in its own SSE chunk, so a watchdog can fire
        on first-byte silence (server stuck in prefill) OR mid-stream
        silence (decode stalled mid-token).
      - The proxy's loop-guard, which runs mid-decode, actually aborts
        runaway thinking blocks instead of decorating the final response
        post-hoc.
      - max_tokens runaway becomes visible: a 16k thinking block now
        streams in real time and the watchdog can cut it off.
    Watchdog aborts surface as URLError, so the existing post_chat()
    retry-with-backoff path kicks in transparently. Set
    QWEN_AGENT_STREAM=0 to use the legacy non-streaming path."""
    if os.environ.get("QWEN_AGENT_STREAM", "1") in ("", "0", "false", "False"):
        return _do_post_blocking(messages)
    return _do_post_streaming(messages)


# Watchdog timeouts — separately tunable so a slow first prefill doesn't
# get aborted as if it were a stuck decode.
#   FIRST_BYTE_TIMEOUT covers cold-cache prefill on a 32k-token prompt.
#     dflash typically prefills ~1k tokens/s, so 60s = 60k tokens of
#     headroom — well above any real prompt.
#   IDLE_TIMEOUT covers between-token silence after decoding has started.
#     At ~24 TPS, a normal token arrives every ~40ms; 90s of silence is
#     a guaranteed stall (or a loop-guard mid-decode abort that already
#     fired but didn't close the stream cleanly).
_STREAM_FIRST_BYTE_TIMEOUT = float(
    os.environ.get("QWEN_AGENT_FIRST_BYTE_TIMEOUT", "60"))
_STREAM_IDLE_TIMEOUT = float(
    os.environ.get("QWEN_AGENT_IDLE_TIMEOUT", "90"))


def _do_post_blocking(messages: list[dict]) -> dict:
    """Legacy non-streaming POST. Kept as a fallback when QWEN_AGENT_STREAM=0
    or for tests that pre-date the streaming refactor."""
    body_dict: dict = {
        "model": MODEL,
        "messages": messages,
        "tools": _filtered_tools(),
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": _MAX_TOKENS,
    }
    if _SKIP_THINKING_ROUTINE and _routine_turn(messages):
        body_dict["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(body_dict).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _do_post_streaming(messages: list[dict]) -> dict:
    """Streaming POST with socket-level read timeouts. Returns the same
    dict shape as `_do_post_blocking` so callers don't change.

    Why socket-level (not a watchdog thread): `resp.close()` on a urllib
    HTTPResponse doesn't reliably interrupt an in-flight `readline()`
    call — the OS keeps the syscall blocked until something else wakes
    it. Setting `socket.settimeout(N)` on the underlying socket makes
    every read raise `socket.timeout` after N seconds, which is
    deterministic and propagates cleanly through the iterator.

    We use a SINGLE socket timeout = `_STREAM_IDLE_TIMEOUT` (the more
    conservative of the two thresholds). For the first-byte case, we
    track elapsed time in software and re-classify the timeout as a
    "first-byte" abort if it fires before any chunk arrived. This is
    simpler than running two timeouts and gives the same behaviour —
    a stalled prefill manifests as zero chunks within the timeout
    window, which is exactly what we want to detect.
    """
    import socket
    import time as _time
    body_dict: dict = {
        "model": MODEL,
        "messages": messages,
        "tools": _filtered_tools(),
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": _MAX_TOKENS,
    }
    if _SKIP_THINKING_ROUTINE and _routine_turn(messages):
        body_dict["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(body_dict).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Initial connect+headers timeout — should be sub-second on a healthy
    # daemon. If the daemon is down, fail fast so post_chat() retries.
    # We use the larger of the two stream timeouts to cover slow-startup
    # cases (first-prefill on a 32k-token cold prompt cache).
    initial_timeout = max(_STREAM_FIRST_BYTE_TIMEOUT, 30.0)
    resp = urllib.request.urlopen(req, timeout=initial_timeout)

    # Set the per-read socket timeout. Each subsequent line read from
    # the SSE stream will block for at most this long before raising
    # socket.timeout.
    try:
        # urllib's HTTPResponse keeps the socket on `.fp.raw._sock` (or
        # `.fp._sock` on some Python builds). Try both before giving up.
        sock = None
        fp = getattr(resp, "fp", None)
        if fp is not None:
            raw = getattr(fp, "raw", None)
            sock = getattr(raw, "_sock", None) if raw is not None else None
            if sock is None:
                sock = getattr(fp, "_sock", None)
        if sock is not None:
            sock.settimeout(_STREAM_IDLE_TIMEOUT)
    except Exception:  # noqa: BLE001
        # If we can't reach the socket on this Python build, the stream
        # still works — just without the per-read timeout. The driver's
        # subprocess timeout caps total wallclock either way.
        pass

    started = _time.monotonic()
    got_first_byte = False
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    structured_tool_calls: list[dict] = []
    finish_reason: str | None = None

    try:
        for raw in resp:
            got_first_byte = True
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
            delta = choices[0].get("delta") or {}

            # Reasoning_content / reasoning fields come through in deltas
            # alongside content when the proxy is emitting Qwen's split
            # reasoning. Accumulate separately so the dict we return
            # mirrors the non-stream shape.
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                reasoning_parts.append(r)

            c = delta.get("content")
            if c:
                content_parts.append(c)

            # Streamed tool_calls: collect any delta entry that carries a
            # function name (Qwen3.6 emits a complete tool_call in one
            # delta, so partial-arg accumulation isn't needed at this
            # layer — the existing _parse_xml_tool_calls fallback in
            # step() handles the rare cases where the model emits XML
            # tool_calls inside `content` instead).
            tcs = delta.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if (isinstance(tc, dict)
                            and (tc.get("function") or {}).get("name")):
                        structured_tool_calls.append(tc)
    except (socket.timeout, TimeoutError) as e:
        # The socket-level read timed out. Reclassify as first-byte vs
        # idle based on whether any chunk arrived. Both surface as
        # URLError so post_chat()'s existing retry-with-backoff handles
        # them transparently.
        elapsed = _time.monotonic() - started
        if not got_first_byte:
            reason = (f"first-byte timeout ({elapsed:.0f}s of silence "
                      f"before any token arrived — server stuck in prefill "
                      f"or queue)")
        else:
            reason = (f"idle timeout ({_STREAM_IDLE_TIMEOUT:.0f}s between "
                      f"tokens after streaming had started — decode stalled)")
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
        raise urllib.error.URLError(
            f"streaming watchdog aborted: {reason}. "
            f"Run `bin/qwen restart` to clear stuck server state, or "
            f"set QWEN_AGENT_STREAM=0 to fall back to non-streaming."
        ) from e
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass

    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)
    msg: dict = {"role": "assistant", "content": full_content}
    if structured_tool_calls:
        msg["tool_calls"] = structured_tool_calls
    if full_reasoning:
        msg["reasoning_content"] = full_reasoning
    return {
        "choices": [{
            "message": msg,
            "finish_reason": finish_reason or "stop",
        }],
    }


def post_chat(messages: list[dict], retries: int = 3) -> dict:
    """POST with retry-with-backoff for transient 5xx + URL errors.

    Model servers can transiently return 500 (asyncio queue timeouts, IPC
    hiccups, GC pauses) — retrying after a short wait usually succeeds. On
    the final retry we additionally try a forced compaction so a too-large
    prompt is shrunk before the last attempt. This is a general resilience
    layer; nothing about it is specific to any particular prompt.
    """
    import time
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _do_post(messages)
        except urllib.error.HTTPError as e:
            last_err = e
            transient = e.code in (500, 502, 503, 504)
            if not transient or attempt == retries:
                # Last-chance compaction: if the server keeps 5xx-ing, the
                # prompt may simply be too large. Try to compact and retry once.
                if transient and attempt == retries:
                    try:
                        compacted = maybe_compact(messages, threshold=0)
                    except TypeError:
                        compacted = maybe_compact(messages)
                    if compacted is not None:
                        before = real_tokens(messages, _filtered_tools())
                        after = real_tokens(compacted, _filtered_tools())
                        print(f"{YELLOW}[server kept returning {e.code}; force-compacted "
                              f"{before // 1000}k → {after // 1000}k tokens and retrying once]{RESET}")
                        messages.clear()
                        messages.extend(compacted)
                        try:
                            return _do_post(messages)
                        except Exception:  # noqa: BLE001
                            pass
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"{YELLOW}[server {e.code}; retry {attempt + 1}/{retries} in {wait}s]{RESET}")
            time.sleep(wait)
        except urllib.error.URLError as e:
            last_err = e
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"{YELLOW}[connection error {e}; retry {attempt + 1}/{retries} in {wait}s]{RESET}")
            time.sleep(wait)
    if last_err is not None:
        raise last_err
    raise RuntimeError("post_chat: exhausted retries with no specific error")


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"\n…[truncated {len(s) - n} chars]"


def short(s: str, n: int = 100) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


# ---------- tool-call dedup + caching --------------------------------------
# Two failure modes we mitigate here:
# 1. Model emits multiple identical tool_calls in one turn (parallel) → dispatch
#    once, share the result. Saves time and avoids polluting context.
# 2. Model re-issues the same call across turns (e.g. searching with minor
#    variations of a pattern that returns no matches) → return a cached
#    "[same call returned the same result earlier]" marker so the model
#    sees identical input and either updates its hypothesis or stops.
#
# The actual cache+dispatch logic lives in agent_tools.CachedDispatcher so
# the chat UI can share identical semantics without duplication. We hold one
# instance for the lifetime of this CLI process. Seed the URL guard's seen-
# set with anything in the system prompt (so e.g. a system prompt that
# mentions docs.python.org pre-authorizes that URL).
_CACHED = CachedDispatcher()
_CACHED.note_text(SYSTEM_PROMPT)


def cached_dispatch(fn: str, args: dict) -> tuple[str, bool]:
    """Module-level wrapper around the shared CachedDispatcher instance —
    kept for backward compatibility with code that imports this name."""
    return _CACHED.dispatch(fn, args)


# Loop-detection state: count consecutive turns where every tool call was a
# cache hit. After N such turns the model is provably stuck retrying the
# exact same calls — inject a hard commit nudge to break the loop.
_consecutive_all_cached_turns = 0
_LOOP_BREAK_THRESHOLD = 3
_consecutive_missing_arg_turns = 0
_MISSING_ARG_THRESHOLD = int(os.environ.get("QWEN_MISSING_ARG_THRESHOLD", "3"))
# Detect cap-exhaustion loops: model keeps trying tools that all return
# `[REFUSED — ...]` because session caps are hit, URLs aren't whitelisted,
# or earlier calls already errored. Without intervention the model narrates
# "I've hit the search cap. Let me try..." indefinitely instead of calling
# done() with a partial answer.
_consecutive_all_refused_turns = 0
_total_all_refused_turns = 0   # cumulative count, doesn't reset on nudge — drives the backstop
# Lowered 2 → 1 after observing the model burning 3-4 turns issuing more
# capped tool calls before the previous-threshold nudge fired. Single-turn
# trigger means the FIRST turn where every tool call was refused
# immediately gets the "synthesize-now" nudge — no further wasted calls.
_REFUSED_THRESHOLD = int(os.environ.get("QWEN_REFUSED_THRESHOLD", "1"))

# Loop-guard surfacing: when the proxy emits its `[loop-guard: …]` marker,
# the model's previous response was cut short because of a repetition loop.
# We print a colored notice to the user (so it doesn't look like a normal
# trailing sentence) and inject a nudge so the NEXT turn course-corrects
# instead of resuming the loop. Without the nudge, the model often
# regenerates the same loop because nothing in its context tells it the
# previous attempt was a runaway. Single-fire per top-level user query so
# we never spam.
# Loop-guard marker detection lives in scripts/loop_guard_marker.py so
# agent.py and agent_graph.py share the same false-positive-resistant
# detector. See that module's docstring for the design rationale.
from loop_guard_marker import (  # noqa: E402
    LOOP_GUARD_RE as _LOOP_GUARD_RE,
    is_proxy_abort_marker as _is_proxy_abort_marker,
    harness_nudge_message as _shared_loop_guard_nudge,
)
_loop_guard_nudge_fired = False
# Iter 29: hard cap on loop-guard aborts per query. After this many in
# one user query, step() force-terminates instead of letting the model
# keep retrying on every turn. The course-correction nudge already
# single-shoots; this counter is the safety net.
_loop_guard_abort_count = 0
_loop_guard_force_terminate = False
_LOOP_GUARD_HARD_LIMIT = int(os.environ.get("QWEN_LOOP_GUARD_HARD_LIMIT", "3"))


def _loop_guard_nudge_message(reason: str) -> dict:
    """Course-correction nudge after a loop-guard abort. Now a thin
    wrapper around the shared helper so the same prose lives in one
    place. Kept under this name for backward-compat with the test
    suite and any other call sites."""
    return _shared_loop_guard_nudge(reason)


def _loop_break_message() -> dict:
    return {
        "role": "user",
        "content": (
            "[LOOP DETECTED: your last "
            f"{_LOOP_BREAK_THRESHOLD} turns issued tool calls that were ALL "
            "served from cache (you've already asked these exact questions). "
            "Stop investigating. Synthesize the best answer you can from the "
            "evidence already gathered above and return it as plain text "
            "with NO further tool calls. If genuinely missing critical "
            "information, say so explicitly and stop.]"
        ),
    }


_XML_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)
_XML_FUNC_RE = re.compile(r"<function=([\w_.-]+)>(.*?)(?:</function>|\Z)", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([\w_.-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_xml_tool_calls(content: str) -> list[dict]:
    """Fallback parser for Qwen's XML-style tool-call syntax.

    The model occasionally emits tool calls as text inside `content` instead
    of populating the `tool_calls` array, especially under harness-nudge
    pressure. Format observed:

        <tool_call>
        <function=NAME>
        <parameter=key>value</parameter>
        ...
        </function>
        </tool_call>

    Returns a list of synthetic tool_call dicts compatible with the OpenAI
    tool-calling shape. Returns [] if no parseable XML calls are found.
    Generalizable: works for any function name + parameter set.
    """
    if "<tool_call>" not in content and "<function=" not in content:
        return []
    out: list[dict] = []
    blocks = _XML_TOOLCALL_RE.findall(content) or ([content] if "<function=" in content else [])
    for block in blocks:
        for fn_match in _XML_FUNC_RE.finditer(block):
            name = fn_match.group(1)
            body = fn_match.group(2)
            args: dict = {}
            for pm in _XML_PARAM_RE.finditer(body):
                args[pm.group(1)] = pm.group(2)
            out.append({
                "id": f"xmlcall_{len(out)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
    return out


def _check_and_handle_loop_guard(msg: dict, messages: list[dict]) -> None:
    """If the assistant message's content contains the proxy's loop-guard
    marker, print a colored notice and inject a course-correction nudge
    for the next turn.

    Iter 29 efficiency: tracks consecutive loop-guard aborts in a global
    counter. The course-correction nudge fires once (single-shot like
    iter 28) but the COUNTER ticks every time, even after the nudge is
    spent. After `_LOOP_GUARD_HARD_LIMIT` aborts in a single query the
    next call to step() will force-terminate via the
    `_loop_guard_force_terminate` flag — without that, a model that
    loops on every retry keeps the eval pinned for the full 600s
    timeout (iter 29: a session hit 4 consecutive aborts before kill).

    Called BEFORE the tool_calls branch in step() so that even a response
    with both tool_calls + truncating loop marker gets surfaced. The
    proxy's loop-guard runs AFTER tool_call XML has been generated, so
    a partial-tool-call-then-loop response is a valid (rare) shape: the
    tool_call should still execute, but the user / model should know the
    response was loop-aborted before any further reasoning.
    """
    global _loop_guard_nudge_fired, _loop_guard_abort_count
    global _loop_guard_force_terminate
    content = msg.get("content") or ""
    # Tightened: require the proxy's specific abort suffix to avoid
    # false-firing on benign mentions of the substring (e.g. user
    # asking "how does the loop guard work?", or a tool result that
    # echoes config/log text containing "[loop-guard:").
    if not _is_proxy_abort_marker(content):
        return
    _loop_guard_abort_count += 1
    m = _LOOP_GUARD_RE.search(content)
    reason = m.group(1).strip() if m else "repetition loop"
    if _loop_guard_abort_count >= _LOOP_GUARD_HARD_LIMIT:
        _loop_guard_force_terminate = True
        print(f"\n{YELLOW}[loop-guard fired #{_loop_guard_abort_count}: "
              f"{reason} — HARD LIMIT reached, terminating session]{RESET}")
        # Append a final synthesize-and-stop message so the next turn
        # has zero ambiguity about what to do.
        messages.append({
            "role": "user",
            "content": (
                f"[HARNESS] {_loop_guard_abort_count} loop-guard aborts in "
                "this query. STOP. Issue ONE final assistant message: "
                "either call done(summary=<best answer from gathered "
                "evidence>) if you have any answer, or write a brief "
                "plain-text 'unable to complete due to repeated "
                "decode loops' note. No more tool calls."),
        })
        return
    if _loop_guard_nudge_fired:
        return  # nudge spent; let the counter still tick
    _loop_guard_nudge_fired = True
    print(f"\n{YELLOW}[loop-guard fired: {reason} — injecting "
          f"course-correction nudge for next turn]{RESET}")
    messages.append(_loop_guard_nudge_message(reason))


def step(messages: list[dict], step_num: int = 0) -> bool:
    """Run one agent step. Returns True if a final answer was produced."""
    # Iter 29: hard-terminate if too many consecutive loop-guard aborts
    # in this query. Without this, a model that loops every retry pins
    # the eval for the full subprocess timeout (iter 29: a session
    # hit 4 aborts before manual kill).
    if _loop_guard_force_terminate:
        print(f"{YELLOW}[step {step_num}: loop-guard hard limit "
              f"reached — terminating session]{RESET}")
        return True
    compacted = maybe_compact(messages)
    if compacted is not None:
        before = real_tokens(messages, _filtered_tools())
        after = real_tokens(compacted, _filtered_tools())
        print(f"{YELLOW}[compacted context: {before // 1000}k → {after // 1000}k tokens]{RESET}")
        messages.clear()
        messages.extend(compacted)
    # Lightweight progress banner: step number + ~context size + warning at 75%
    # of compaction threshold. Helps users gauge wall-clock progress on long
    # multi-step runs without changing any tool-loop semantics.
    tokens = real_tokens(messages, _filtered_tools())
    pct = (tokens * 100) // max(COMPACT_AT_TOKENS, 1)
    if pct >= 75:
        ctx_color = YELLOW
    else:
        ctx_color = DIM
    print(f"{ctx_color}[step {step_num} | ~{tokens // 1000}k tokens ({pct}% of compact threshold)]{RESET}")
    resp = post_chat(messages)
    msg = resp["choices"][0]["message"]
    messages.append(msg)

    # Loop-guard surfacing happens early: even responses that DO have
    # tool_calls can carry the marker if the model emitted a tool_call
    # before slipping into a loop. Without this early check, the marker
    # in `content` would be invisible because the no-tool-calls branch
    # wouldn't run.
    _check_and_handle_loop_guard(msg, messages)

    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        # FALLBACK: Qwen sometimes emits XML-style tool calls as text in
        # `content` instead of populating `tool_calls`. Parse and recover so
        # the call isn't silently dropped. Generalizable across scenarios.
        raw_content = msg.get("content") or ""
        xml_calls = _parse_xml_tool_calls(raw_content) if raw_content else []
        if xml_calls:
            print(f"{YELLOW}[harness: parsed {len(xml_calls)} XML tool_call(s) from content]{RESET}")
            tool_calls = xml_calls
            msg["tool_calls"] = xml_calls
            msg["content"] = ""  # avoid double-display + double-parse on retries
    if not tool_calls:
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            print(f"{DIM}thinking… {short(reasoning, 200)}{RESET}")
        content = (msg.get("content") or "").strip()
        if content:
            print(content)
        # Loop-guard surfacing already happened earlier in this step()
        # call (before the tool_calls branch). If the marker was present
        # AND there were no tool_calls, the nudge has been appended; we
        # need to bail out without exiting so the next step can act on
        # it. Use the tightened detector so a benign mention of the
        # substring doesn't change exit semantics.
        if _loop_guard_nudge_fired and content and _is_proxy_abort_marker(content):
            return False  # don't exit; let the next step course-correct
        # Empty-turn nudge: assistant emitted NO content, NO tool_calls,
        # and NO reasoning. Iter 28 netflix bug: streaming returned an
        # empty SSE (model still warming up after hard-restart) and the
        # loop exited silently after a single tool call, scoring 0. This
        # also catches the rare case where dflash-serve produces a
        # zero-token completion for any reason. Fire a SINGLE nudge per
        # session before giving up — the next step usually recovers.
        # Generalizable: any future empty-stream state is now a soft
        # error rather than a silent task failure.
        global _empty_turn_nudged
        if (not content and not tool_calls and not reasoning
                and not _empty_turn_nudged):
            _empty_turn_nudged = True
            print(f"{YELLOW}[exit blocked: empty assistant turn — nudging]{RESET}")
            messages.append({
                "role": "user",
                "content": (
                    "[HARNESS] Your previous turn returned no content, no "
                    "tool calls, and no reasoning. This usually means the "
                    "model returned an empty completion. Resume the task: "
                    "either continue with the next tool call, emit a final "
                    "answer in plain text, or call `done(summary=...)` if "
                    "you're finished. Do NOT return another empty turn."),
            })
            return False  # don't exit; let the next step act on the nudge
        # Block premature exits ONLY when the user named specific output
        # paths that haven't been written yet. The previous version also
        # nudged on "n_writes == 0 and first nudge", which fired
        # spuriously on conversational prompts ("how's your day?") that
        # don't ask for any file output — forcing the model to write a
        # useless placeholder file just to satisfy the gate. Now we only
        # nudge when there's a concrete missing deliverable to point at.
        global _exit_nudge_count
        if _exit_nudge_count < _MAX_EXIT_NUDGES:
            try:
                from agent_tools import _session_writes
                writes_now = _session_writes()
            except Exception:
                writes_now = {"_fail_open": 1}  # fail-open: don't block if tracking broken
            missing = _named_paths_unwritten(messages, writes_now)
            if missing:
                _exit_nudge_count += 1
                paths_clause = (f"User-named paths NOT YET WRITTEN: {missing}. "
                                f"Each one must exist on disk before completion. ")
                print(f"{YELLOW}[exit blocked nudge #{_exit_nudge_count}: "
                      f"missing={missing}]{RESET}")
                messages.append({
                    "role": "user",
                    "content": (f"[HARNESS] STOP. {paths_clause}"
                                f"Your VERY NEXT tool call MUST be `write_file(<path>, <content>)` "
                                f"or `append_finding(<path>, <heading>, <content>)` for the missing path(s). "
                                f"Do NOT call python_run, bash, read_file, or anything else. "
                                f"Just write the file. Even partial content is fine. Then call done."),
                })
                return False  # do not exit; continue loop
        return True

    # Parse + announce all calls up front. Then dispatch UNIQUE calls in
    # parallel (intra-turn dedup) using cached_dispatch (cross-turn cache).
    prepared = []
    for tc in tool_calls:
        fn = tc["function"]["name"]
        raw_args = tc["function"].get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
        prepared.append((tc, fn, args))
        arg_preview = short(json.dumps(args, ensure_ascii=False), 120)
        print(f"{CYAN}→ {fn}({arg_preview}){RESET}")

    # Intra-turn dedup + parallel dispatch + cross-turn cache: all handled
    # by the shared CachedDispatcher.dispatch_batch in agent_tools.
    results = _CACHED.dispatch_batch([(fn, args) for _tc, fn, args in prepared])

    global _consecutive_all_cached_turns, _consecutive_missing_arg_turns
    global _consecutive_all_refused_turns, _total_all_refused_turns
    cache_count = sum(1 for _r, was_cached in results if was_cached)
    if results and cache_count == len(results):
        _consecutive_all_cached_turns += 1
    else:
        _consecutive_all_cached_turns = 0

    # Detect generation-truncation: when ALL tool calls in this turn raised
    # "missing N required positional argument", the model is likely emitting
    # tool_calls with truncated JSON args (output budget overflow on big
    # content). Counts consecutively across turns; nudge fires once at
    # threshold. Generalizable across any tool with required args.
    all_missing_arg = (
        bool(results)
        and all(
            isinstance(r, str) and "missing" in r and "required positional argument" in r
            for r, _cached in results
        )
    )
    if all_missing_arg:
        _consecutive_missing_arg_turns += 1
    else:
        _consecutive_missing_arg_turns = 0

    # Detect cap-exhaustion loops: every tool call in this turn returned a
    # `[REFUSED — ...]` marker (search/fetch cap hit, unseen URL, dead URL,
    # duplicate). Without intervention the model narrates "I'll try X" prose
    # without ever calling done(). Threshold low (default 2) because each
    # turn after the first is wasted compute.
    all_refused = (
        bool(results)
        and all(
            isinstance(r, str) and r.lstrip().startswith("[REFUSED")
            for r, _cached in results
        )
    )
    if all_refused:
        _consecutive_all_refused_turns += 1
        _total_all_refused_turns += 1   # cumulative: never resets within a query
    else:
        _consecutive_all_refused_turns = 0

    done_signaled = False
    # Last user message is the relevance signal for triage. Walk back from
    # the end to find it (skipping any system/assistant/tool messages added
    # since the user spoke).
    last_user_task = next(
        (m.get("content", "") for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        "",
    )
    for (tc, fn, _args), (raw_result, was_cached) in zip(prepared, results):
        # Run condense FIRST on the raw result, THEN cap with TOOL_RESULT_MAX
        # as a safety net. Reversing this order matters: a 240k SEC 10-K
        # would otherwise be head-truncated to 60k BEFORE condense, throwing
        # away the data tables in the middle. With condense first, the
        # full document gets chunk-ranked (with table-detection +
        # section-continuity boosts), so the right sections survive
        # whatever the document size.
        result_str = str(raw_result)
        result, triage_info = triage_tool_result(last_user_task, fn, result_str)
        result = truncate(result, TOOL_RESULT_MAX)
        _CACHED.record_reduction(triage_info)
        first_line = next((l for l in result.splitlines() if l.strip()), "")
        cache_tag = " [cached]" if was_cached else ""
        triage_tag = ""
        if triage_info["verdict"] == "low_relevance":
            triage_tag = (f" [triage: pruned "
                          f"{triage_info['chars_in']}→{triage_info['chars_out']} "
                          f"score={triage_info['score']}]")
        elif triage_info["verdict"] == "condensed":
            triage_tag = (f" [condensed: "
                          f"{triage_info['chars_in']}→{triage_info['chars_out']} "
                          f"chars, {triage_info.get('chunks_kept')}/"
                          f"{triage_info.get('chunks_in')} chunks]")
        print(f"{DIM}  {short(first_line, 120)}{cache_tag}{triage_tag}{RESET}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": fn,
                "content": result,
            }
        )
        # If the agent called `done` and the result was accepted (not refused),
        # treat this as an explicit "session complete" signal: break out of
        # run_query immediately, equivalent to /exit. Avoids burning more LLM
        # rounds after the task is finished.
        #
        # Exception — pre-done() audit gate (general single-shot fix): for
        # quantitative questions, the FIRST done() defers to a self-audit
        # turn so the model can catch period / units / scope mismatches
        # before the answer is locked in. Subsequent done() calls (after
        # audit) terminate normally.
        if fn == "done" and isinstance(raw_result, str) and raw_result.startswith("DONE accepted"):
            global _audit_fired
            if (_AUDIT_GATE_ENABLED and not _audit_fired
                    and _question_is_quantitative(messages)):
                _audit_fired = True
                # Iter 33: extract done() summary, check against evidence.
                # `_args` is the current tool call's parsed arguments
                # from the for-loop above (the unique-args binding).
                try:
                    done_summary = _args.get("summary", "") or ""
                except Exception:  # noqa: BLE001
                    done_summary = ""
                evidence = _evidence_text_from_messages(messages)
                unsupported = _unsupported_numbers_in_summary(done_summary, evidence)
                if unsupported:
                    print(f"{YELLOW}  [audit gate: deferring done; one self-audit turn "
                          f"+ {len(unsupported)} unsupported numbers flagged]{RESET}")
                else:
                    print(f"{YELLOW}  [audit gate: deferring done; one self-audit turn]{RESET}")
                messages.append(_audit_message(unsupported_numbers=unsupported or None))
                # Don't set done_signaled — let the loop run one more turn
                # so the model can confirm or correct.
            else:
                done_signaled = True
                print(f"{CYAN}  [done signal received — closing session]{RESET}")

    # `done` tool was just called and accepted — escalate to caller via the
    # same return path the model normally uses ("no tool calls means stop").
    # We surface this by setting a sentinel attribute on the messages list
    # and returning True (mirroring the no-tool-calls path).
    if done_signaled:
        return True

    # ---- Nudge prioritization (audit issue A fix) -----------------------
    # Multiple nudges could fire in the same turn — e.g. the read-loop
    # nudge and the cache-loop nudge both want to speak when the model has
    # been investigating fruitlessly for several steps. Pre-fix, we'd
    # append all of them, which gave the model a confusing pile of
    # contradictory `[HARNESS]` messages ("synthesize now" vs "stop
    # investigating" vs "write a draft").
    #
    # Now we pick ONE nudge per turn, the highest-priority one. The
    # priority order (high → low) reflects how specific each signal is:
    #   1. Cap-exhaustion (every tool call REFUSED) — most concrete,
    #      tool budget is literally spent.
    #   2. Truncation (every call missing required args) — broken
    #      output, fix it before doing anything else.
    #   3. Cache-loop (every call cache hit) — model is repeating
    #      itself, pivot or commit.
    # Each nudge still has its own single-fire latch so it can re-trigger
    # on a later turn if the priority winner changes. We also still let
    # the existing read-loop / stale nudges fire — but only when none of
    # these higher-priority tool-result nudges fired this turn.
    nudge_fired_this_turn = False

    # Priority 1 — cap-exhaustion. Every tool call REFUSED for N turns.
    if _consecutive_all_refused_turns >= _REFUSED_THRESHOLD:
        print(f"{YELLOW}[cap-exhaustion detected: {_consecutive_all_refused_turns} "
              "consecutive all-refused turns — injecting commit-now nudge]{RESET}")
        messages.append({
            "role": "user",
            "content": (
                "[HARNESS] Every tool call in your last "
                f"{_consecutive_all_refused_turns} turns has been REFUSED (search/fetch cap "
                "hit, URLs not whitelisted, or earlier calls already errored). The tool "
                "budget for this query is exhausted. Do NOT keep narrating retries. Right now: "
                "(1) write your best-effort answer using ONLY the data you've already gathered; "
                "(2) call write_file or write_file_verified for any required artifact; "
                "(3) call done(summary) — the summary should explicitly state which specific "
                "datum is missing and why it couldn't be retrieved."
            ),
        })
        _consecutive_all_refused_turns = 0
        nudge_fired_this_turn = True

    # Priority 2 — truncation. Generation overflowed and broke JSON args.
    elif _consecutive_missing_arg_turns >= _MISSING_ARG_THRESHOLD:
        print(f"{YELLOW}[truncation detected: {_consecutive_missing_arg_turns} "
              "consecutive missing-arg errors — injecting smaller-write nudge]{RESET}")
        messages.append({
            "role": "user",
            "content": (
                "[HARNESS] Your recent tool calls failed with `missing required positional argument`. "
                "This is a generation-truncation symptom: the JSON arguments overflowed your output budget. "
                "Fix: (1) write a SHORTER initial file (under 1000 chars) using write_file, "
                "then (2) call append_finding(<path>, <heading>, <next_chunk>) for each additional chunk. "
                "Do NOT retry the same large write_file call."
            ),
        })
        _consecutive_missing_arg_turns = 0
        nudge_fired_this_turn = True

    # Priority 3 — cache-loop. Model is repeating identical calls.
    elif _consecutive_all_cached_turns >= _LOOP_BREAK_THRESHOLD:
        print(f"{YELLOW}[loop detected: {_consecutive_all_cached_turns} consecutive "
              "all-cached turns — injecting commit-now nudge]{RESET}")
        messages.append(_loop_break_message())
        _consecutive_all_cached_turns = 0
        nudge_fired_this_turn = True

    # Auto-nudges. Two failure modes to catch:
    #  (A) step >= 4 with 0 writes  -> read-loop trap (s9/s10/h7-style)
    #  (B) >= 3 turns since last write -> stale-after-write (s2-style timeout)
    # Both surface as a single short user-message injection. Once each.
    global _read_loop_nudged, _stale_post_write_nudged, _last_write_step
    try:
        from agent_tools import _session_writes
        writes_now = _session_writes()
        n_writes = sum(writes_now.values())
    except Exception:
        n_writes = -1
        writes_now = {}

    # Track when the last write happened
    if n_writes > 0:
        # Compute current write count vs prior. If grew, mark this step.
        prior = getattr(step, "_prior_write_count", 0)
        if n_writes > prior:
            _last_write_step = step_num
            step._prior_write_count = n_writes  # type: ignore[attr-defined]

    # (A) read-loop nudge — never wrote anything. Escalates: first nudge
    # at _NUDGE_AFTER_STEPS, second harder nudge at 2× that. Without the
    # second escalation, a deeply-exploring model can ignore the first
    # nudge and burn 15+ more turns before timing out (observed in iter
    # 20: a session ran 17 turns with 0 writes despite nudge at turn 4).
    #
    # Audit-issue-F gate: investigation-heavy tasks ("audit X",
    # "summarize Y", "what's in Z") legitimately spend many turns reading
    # without writing — that IS the work. The 4-step nudge derails them
    # by demanding an artifact when none was requested. We now defer the
    # read-loop nudge to a much later step when the task neither names
    # an output path nor uses a creation verb. The eventual safety net
    # still fires (so a genuinely stuck investigation doesn't run forever)
    # but normal investigation gets ~16 turns to breathe instead of 4.
    expects_artifact = _task_expects_artifact(messages)
    nudge_threshold = (_NUDGE_AFTER_STEPS if expects_artifact
                       else _NUDGE_AFTER_STEPS_INVESTIGATIVE)
    if (step_num >= nudge_threshold and n_writes == 0
            and not _read_loop_nudged
            and not nudge_fired_this_turn):
        _read_loop_nudged = True
        paths = _extract_paths_from_messages(messages)
        paths_clause = (f"User-named output path(s): {paths}. Write a draft of the FIRST one NOW.") \
                       if paths else "Write a stub of your deliverable NOW."
        print(f"{YELLOW}[no writes after {step_num} steps "
              f"(threshold={nudge_threshold}, expects_artifact={expects_artifact}) "
              f"— injecting commit nudge]{RESET}")
        messages.append({
            "role": "user",
            "content": (f"[HARNESS] {step_num} tool calls, 0 writes — read-loop trap. "
                        f"Stop investigating, start writing. {paths_clause} Use write_file or append_finding."),
        })
        return False
    # Second escalation: after nudge fired and STILL no writes by 2× the
    # threshold, force commit-now. Use the existing _read_loop_nudged
    # latch to detect "post-first-nudge"; track second-fire on the
    # function-attribute pattern for cheap idempotency.
    second_threshold = 2 * nudge_threshold
    if (_read_loop_nudged and n_writes == 0
            and step_num >= second_threshold
            and not getattr(step, "_read_loop_2nd_fired", False)
            and not nudge_fired_this_turn):
        step._read_loop_2nd_fired = True  # type: ignore[attr-defined]
        print(f"{YELLOW}[STILL no writes after {step_num} steps — injecting hard commit nudge]{RESET}")
        messages.append({
            "role": "user",
            "content": (f"[HARNESS] You've now made {step_num} tool calls with ZERO writes. "
                        "The first nudge was ignored. STOP all reading. Your VERY NEXT call must be "
                        "write_file (or append_finding) with whatever you have. Even partial. "
                        "After the write, call done(summary) — do not investigate further."),
        })
        return False

    # (B) stale-after-write nudge — wrote at least once, but >=3 steps ago
    # and still tool-calling. This is the s2 pattern: artifact was written
    # but agent keeps making redundant calls until timeout.
    if (n_writes > 0 and not _stale_post_write_nudged
            and _last_write_step > 0
            and (step_num - _last_write_step) >= 3
            and not nudge_fired_this_turn):
        _stale_post_write_nudged = True
        print(f"{YELLOW}[stale {step_num - _last_write_step} steps post-write — injecting done-now nudge]{RESET}")
        messages.append({
            "role": "user",
            "content": ("[HARNESS] Your artifact is on disk and you've made several more tool calls "
                        "without writing anything new. If the task is complete, call done(summary) NOW. "
                        "If you need to refine, write the next concrete update — don't keep reading."),
        })
    return False


_NUDGE_AFTER_STEPS = int(os.environ.get("QWEN_NUDGE_AFTER_STEPS", "4"))
# Audit-issue-F: investigation tasks ("audit", "summarize", "what's in")
# legitimately spend many turns reading. The standard 4-step nudge derails
# them. When a task neither names an output path nor uses a creation verb
# we use this much higher threshold instead — still a safety net for
# genuinely stuck investigations, but doesn't fire on normal exploration.
_NUDGE_AFTER_STEPS_INVESTIGATIVE = int(os.environ.get(
    "QWEN_NUDGE_AFTER_STEPS_INVESTIGATIVE", "16"))
_MAX_EXIT_NUDGES = int(os.environ.get("QWEN_MAX_EXIT_NUDGES", "2"))
_read_loop_nudged = False
_stale_post_write_nudged = False
_last_write_step = 0
_exit_nudge_count = 0  # supersedes _exit_without_write_blocked
# Iter 28: latch for the empty-assistant-turn nudge. Fires once per
# session if the model emits a turn with no content, no tool_calls, and
# no reasoning. Catches first-turn warmup races and rare zero-token
# completions instead of silently exiting at score 0.
_empty_turn_nudged = False


# Verbs that signal the user expects an artifact written to disk. Kept
# tight on purpose — false positives here force investigation tasks ("what
# is in this file") to fire the read-loop nudge at step 4 unnecessarily.
# Explicit creation verbs only; common prose nouns like "file" or "make"
# excluded because they appear in investigation prompts too. Filename
# extensions stay because a `.py` reference is a strong artifact signal.
_ARTIFACT_VERBS = re.compile(
    r"\b(?:write|create|build|produce|draft|save|generate|"
    r"implement|patch|refactor|"
    r"deliverable|artifact)\b"
    r"|\b(?:fix|edit|modify)\s+(?:the|a|an)?\s*\w+\.\w+"  # fix bug.py, edit foo.md
    r"|\.(?:md|py|ipynb|csv|json|html|txt|sh|yaml|yml|toml)\b",
    re.IGNORECASE,
)


def _task_expects_artifact(messages: list[dict]) -> bool:
    """Heuristic: does the user's most recent task expect a file written
    to disk as the deliverable?

    True when:
      - The user named an explicit output path (e.g. /tmp/foo.md), OR
      - The user message uses a creation/writing verb (write/create/build…)

    False for investigation-only tasks ("audit X", "summarize Y", "what's
    in Z") where reading IS the work and forcing a write derails the task.

    Used to choose between the standard 4-step read-loop nudge and the
    much higher 16-step investigative threshold."""
    paths = _extract_paths_from_messages(messages)
    if paths:
        return True
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.lstrip().startswith("[HARNESS"):
            continue
        return bool(_ARTIFACT_VERBS.search(content))
    return False


# ---- Pre-done() self-audit gate -----------------------------------------
# General fix for the "single-shot variance" failure mode (40-50% per run vs
# 93% best-of-attempts) observed at iter 20/21. Across the failing prompts
# (Q4-vs-FY period mismatch; total-debt vs long-term-debt scope mismatch;
# magnitude / units / scope errors) the model fetched correct data but
# quoted a number whose metadata didn't match the question's exact specs.
#
# Mitigation: when the model first calls done() on a quantitative question,
# the harness defers termination by ONE turn and injects a self-audit user
# message. The model can either:
#   (a) confirm — call done() again, accepted as terminal
#   (b) fix — re-fetch / re-extract / re-compute, then call done() again
# Single-fire per query (post-audit done is always terminal). Skipped for
# purely qualitative questions where the audit's numeric framing doesn't
# apply.
_audit_fired = False  # reset per query
_AUDIT_GATE_ENABLED = os.environ.get("QWEN_AUDIT_GATE", "1") not in (
    "", "0", "false", "False",
)
# Heuristic for "this question expects a numeric answer". Catches money,
# percentages, basis points, comma-formatted numbers (definite quantitative
# signals), quarter / fiscal-year tokens, and common quantitative verbs.
# Misses are fine — the audit only changes single-shot reliability for
# numeric questions; qualitative answers are unaffected.
#
# Iter 27: dropped the bare-4-digit-number trigger (`\b\d{4,}\b`) after
# audit issue E investigation — it was the sole false-positive source on
# qualitative "who was nominated to serve in YYYY"-style questions where
# the year alone fired the audit gate on a purely qualitative question.
# A bare year is too weak a signal on its own. We keep
# `\b\d{1,3}(?:,\d{3})+\b` (comma-formatted numbers like 1,234,567 are
# unambiguously quantitative) and the explicit period tokens (Q1-Q4,
# fiscal, FY20XX). Years still trigger when paired with another cue.
_QUANTITATIVE_CUE = re.compile(
    r"\$|%|\bbps\b|basis\s*points?|"
    r"\b(?:q[1-4]|quarter|fiscal|fy\s?20\d{2}|year[\s-]?(?:ended|over[\s-]year))\b|"
    r"\bhow\s+much\b|\bby\s+how\s+much\b|\bcalculate\b|\bcompute\b|\bchange\b|"
    r"\bgrowth\b|\bmargin\b|\brevenue\b|\bearnings\b|\bebitda\b|\bdebt\b|"
    r"\bdividend\b|\brepurchas|\boutstanding\b|\bshares?\b|"
    r"\b\d{1,3}(?:,\d{3})+\b",
    re.IGNORECASE,
)


def _question_is_quantitative(messages: list[dict]) -> bool:
    """Most recent non-harness user message looks numeric? The audit gate
    only fires for quantitative questions because its prompt is calibrated
    for numeric metadata-mismatch (period / units / scope) — the most
    common silent-error class. Conversational queries get the normal
    one-shot done() path."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.lstrip().startswith("[HARNESS"):
            continue
        return bool(_QUANTITATIVE_CUE.search(content))
    return False


def _audit_message(unsupported_numbers: list[str] | None = None) -> dict:
    """One-shot pre-done() self-audit. Generic, task-agnostic — names the
    four specs every quantitative answer must match (entity / period /
    units / scope) and the most common silent-error patterns.

    Iter 33: when the model's `done()` summary contains numeric claims
    that do NOT appear verbatim in any tool result, we surface them
    explicitly. Catches the NFLX ARPU iter 32 hallucination where the
    model wrote "$6.47 / $7.20 / $9.56" without ever fetching ARPU
    from a real source — pure made-up numbers. Generalizes any future
    "model invents data when retrieval failed" pattern.
    """
    base = (
        "[HARNESS] Before I accept completion, run a 30-second self-audit on your answer.\n"
    )
    if unsupported_numbers:
        # Show up to 6 unsupported numbers; cap to keep token cost low.
        shown = unsupported_numbers[:6]
        more = "" if len(unsupported_numbers) <= 6 else f" (+{len(unsupported_numbers)-6} more)"
        base += (
            f"\n⚠ EVIDENCE CHECK: these numbers in your summary don't appear "
            f"verbatim in any tool result this session: {shown}{more}. "
            f"Either (a) the value was derived (fine — show the formula and "
            f"source values), (b) you read it from a doc whose exact "
            f"phrasing differs (re-quote the row label + value), or "
            f"(c) the number was hallucinated (fix or remove it).\n\n"
        )
    base += (
        "For EACH numeric claim, verify against the original question:\n"
        "  1. ENTITY — exact company / subsidiary / segment? (parent vs subsidiary, "
        "consolidated vs segment)\n"
        "  2. PERIOD — exact time window? (FY vs a quarter; fiscal vs calendar; "
        "year-ended vs three-months-ended; annual total vs Q4-only)\n"
        "  3. UNITS — shares vs dollars vs %; raw vs thousands vs millions vs billions; "
        "gross vs net; GAAP vs non-GAAP; diluted vs basic.\n"
        "  4. SCOPE — which slice of the line item? (total debt vs long-term-only; "
        "operating income vs net income; including vs excluding items)\n"
        "Common silent errors: full-year totals quoted for a quarterly question; "
        "'all debt' quoted when 'long-term debt' was asked; raw thousands quoted "
        "as millions; subsidiary figure quoted for the parent.\n"
        "Use ONLY the documents already in your context — re-fetch only if a "
        "specific row is genuinely missing. If any figure's metadata doesn't match "
        "the question's specs, fix it and call done() again. If everything checks "
        "out, briefly state in 1-2 lines what you verified (which row/column/period "
        "for each number), then call done() again."
    )
    return {"role": "user", "content": base}


# Iter 33: numeric tokens in `done()` summaries. Matches the same
# patterns as the rubric scorer so detection is symmetric.
_DONE_NUM_RE = re.compile(
    r"(?:\$|USD\s*)?"
    r"(?:-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)"
    r"\s*(?:%|bps|basis\s*points?|bn|billion|m|million|k|thousand)?",
    re.IGNORECASE,
)


def _evidence_text_from_messages(messages: list[dict]) -> str:
    """Concatenate all tool_result content blobs from this session.
    Used by `_unsupported_numbers_in_summary` to anchor numeric claims
    against actual retrieved evidence."""
    parts: list[str] = []
    for m in messages:
        # Tool results land as role='tool' with content being the
        # stringified payload. Also include user 'content' since the
        # original question may quote numbers the model is allowed to
        # echo back.
        role = m.get("role")
        if role == "tool":
            c = m.get("content") or ""
            if isinstance(c, str):
                parts.append(c)
        elif role == "user":
            c = m.get("content") or ""
            if isinstance(c, str) and not c.startswith("[HARNESS"):
                parts.append(c)
    return "\n".join(parts)


def _unsupported_numbers_in_summary(summary: str, evidence: str) -> list[str]:
    """Find numeric tokens in `summary` that don't appear verbatim in
    `evidence`. Loose comparison: also tries the comma-stripped and
    dollar-stripped form to avoid false-positives on "1,234,567" vs
    "1234567" or "$10.50" vs "10.50". Skips trivial small ints (≤ 4
    digits, no separators) since they often appear as years, counts,
    or rubric-irrelevant noise.
    """
    if not summary or not evidence:
        return []
    matches = []
    for m in _DONE_NUM_RE.finditer(summary):
        tok = m.group(0).strip()
        # Skip tokens that are just years/small ints with no
        # decimal/comma/percent suffix — too noisy for evidence check.
        if re.fullmatch(r"\d{1,4}", tok):
            continue
        # Try: exact match, comma-stripped, dollar-stripped, both.
        candidates = {tok}
        candidates.add(tok.replace("$", "").strip())
        candidates.add(tok.replace(",", ""))
        candidates.add(tok.replace("$", "").replace(",", "").strip())
        candidates = {c for c in candidates if c}
        if not any(c in evidence for c in candidates):
            matches.append(tok)
    # Dedup while preserving order.
    seen, out = set(), []
    for m in matches:
        if m not in seen:
            seen.add(m); out.append(m)
    return out



def _named_paths_unwritten(messages: list[dict], writes_map: dict) -> list[str]:
    """Return the subset of user-named output paths that haven't been written
    this session. Generalizable across all scenarios — works whenever the
    user's prompt contains explicit /path/to/file.ext references.

    `writes_map` is the dict from agent_tools._session_writes() — mapping
    real-path → write count. We compare via os.path.realpath so symlinks
    and normalized paths match.
    """
    named = _extract_paths_from_messages(messages)
    if not named:
        return []
    written = set()
    for p in writes_map.keys():
        try:
            written.add(os.path.realpath(p))
        except OSError:
            pass
    out = []
    for p in named:
        try:
            rp = os.path.realpath(p)
        except OSError:
            rp = p
        if rp not in written:
            out.append(p)
    return out


def _extract_paths_from_messages(messages: list[dict]) -> list[str]:
    """Pull out file paths the user mentioned. Used by the read-loop nudge
    to surface the deliverable target."""
    import re as _re
    paths: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content") or ""
        if not isinstance(content, str):
            continue
        # /tmp/... /Users/... or any /xxx/yyy.ext pattern
        for match in _re.findall(r"(/[\w./_-]+\.(?:md|py|ipynb|txt|json|csv|html))", content):
            if match not in paths:
                paths.append(match)
    return paths[:3]


def run_query(messages: list[dict]) -> None:
    global _consecutive_all_cached_turns, _read_loop_nudged
    global _stale_post_write_nudged, _last_write_step, _exit_nudge_count
    global _consecutive_missing_arg_turns, _loop_guard_nudge_fired
    global _consecutive_all_refused_turns, _total_all_refused_turns
    global _audit_fired, _empty_turn_nudged
    global _loop_guard_abort_count, _loop_guard_force_terminate
    _consecutive_all_cached_turns = 0  # reset per top-level user query
    _read_loop_nudged = False  # reset nudge per query
    _stale_post_write_nudged = False
    _last_write_step = 0
    _exit_nudge_count = 0
    _consecutive_missing_arg_turns = 0
    _consecutive_all_refused_turns = 0
    _total_all_refused_turns = 0  # reset cumulative counter per top-level query
    _loop_guard_nudge_fired = False  # reset per top-level user query
    _audit_fired = False  # reset per top-level user query
    _empty_turn_nudged = False  # reset per top-level user query
    _loop_guard_abort_count = 0
    _loop_guard_force_terminate = False
    if hasattr(step, "_prior_write_count"):
        delattr(step, "_prior_write_count")

    # Iter 36: register messages list with SIGTERM handler so eval-driver
    # timeouts can still emit an emergency stub artifact instead of
    # losing all partial findings.
    _ref = getattr(sys.modules[__name__], "_term_messages_ref", None)
    if isinstance(_ref, list):
        _ref.clear()
        _ref.append(messages)
    # Hard backstop: if the CUMULATIVE all-refused counter reaches BACKSTOP_LIMIT
    # within a single user query, the model has ignored the nudge and is still
    # spending turns on all-refused calls. Force-terminate to prevent endless
    # loops. Uses _total_all_refused_turns (which doesn't reset on nudge), not
    # _consecutive (which does reset on nudge).
    # Lowered 4 → 2 so a model that ignores the synthesize-now nudge and
    # keeps calling refused tools is force-terminated after just one more
    # all-refused turn. Each ignored-nudge turn adds ~30s of decode +
    # context bloat for zero gain.
    _BACKSTOP_LIMIT = int(os.environ.get("QWEN_REFUSED_BACKSTOP", "2"))
    for i in range(MAX_STEPS):
        if step(messages, step_num=i + 1):
            _maybe_emergency_stub(messages, reason="model called done()")
            return
        if _total_all_refused_turns >= _BACKSTOP_LIMIT:
            print(f"{RED}[BACKSTOP: {_total_all_refused_turns} all-refused turns this query "
                  f"despite nudge — force-terminating. Tools are exhausted; the model is "
                  f"not respecting the cap-exhaustion rule. Returning what was gathered.]{RESET}")
            _maybe_emergency_stub(messages, reason="all-refused backstop")
            return
    print(f"{YELLOW}[hit hard ceiling of {MAX_STEPS} steps — something is genuinely stuck]{RESET}")
    _maybe_emergency_stub(messages, reason="MAX_STEPS hit")


def _maybe_emergency_stub(messages: list[dict], reason: str) -> None:
    """Iter 30 failsafe: when run_query is about to exit and the session
    has produced ZERO writes, harvest the assistant's accumulated
    findings (last few non-empty content blocks + done() summaries)
    and write them to a stub artifact. This converts FAIL → PARTIAL on
    the rare pattern where the model gathered partial data but couldn't
    synthesize cleanly before hitting a guard.

    Generic, response-shape-keyed: only fires on the (no-writes) edge
    case. Sessions that produced any artifact via write_file /
    append_finding / write_file_verified are unaffected.
    """
    try:
        from agent_tools import _session_writes
        if _session_writes():
            return  # already have an artifact
    except Exception:
        return
    # Harvest the user's question (for context) + assistant's
    # accumulated content/done summaries.
    user_q = next(
        (m.get("content","") for m in reversed(messages)
         if m.get("role") == "user"
         and isinstance(m.get("content"), str)
         and not m.get("content","").startswith("[HARNESS")),
        ""
    )
    findings = []
    for m in messages[-30:]:
        if m.get("role") != "assistant":
            continue
        c = (m.get("content") or "").strip()
        if c and not c.startswith("[loop-guard:"):
            findings.append(c)
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name","")
            if fn == "done":
                try:
                    args = json.loads((tc.get("function") or {}).get("arguments","{}"))
                    if args.get("summary"):
                        findings.append(f"[done summary] {args['summary']}")
                except Exception:
                    pass
    if not findings:
        return
    body = (
        f"# Emergency stub artifact\n\n"
        f"Session terminated before the agent wrote a final artifact "
        f"(reason: {reason}). The harness has harvested the assistant's "
        f"accumulated findings below for downstream scoring / review.\n\n"
        f"## Question\n{user_q[:500]}\n\n"
        f"## Findings (most recent first)\n\n"
    )
    for f in reversed(findings[-10:]):
        body += f"---\n{f[:1500]}\n\n"
    try:
        from agent_tools import write_file
        import tempfile
        path = f"{tempfile.gettempdir()}/qwen_emergency_stub_{os.getpid()}.md"
        write_file(path, body)
        print(f"{YELLOW}[emergency stub written: {path} ({len(body)} chars)]{RESET}")
    except Exception as e:  # noqa: BLE001
        print(f"{YELLOW}[emergency stub failed: {e}]{RESET}")


HELP = """\
Commands:
  /exit, /quit       leave the agent (server keeps running)
  /clear             reset conversation history
  /system <text>     replace the system prompt
  /tools             list registered tools
  /help              this help
"""


def _open_session_log() -> "SessionLog | None":
    """Best-effort: open a session JSONL log file. Returns None if disabled
    or unwritable — the agent never fails because logging didn't work.

    Disable with QWEN_SESSION_LOG=off. Override directory with
    QWEN_SESSION_LOG_DIR (default: <project_root>/logs/sessions).
    """
    if os.environ.get("QWEN_SESSION_LOG", "on").lower() == "off":
        return None
    # Resolve sessions dir: env override or sibling of scripts/.
    log_dir = os.environ.get("QWEN_SESSION_LOG_DIR")
    if not log_dir:
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(proj_root, "logs", "sessions")
    try:
        os.makedirs(log_dir, exist_ok=True)
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(log_dir, f"agent-{ts}-{os.getpid()}.jsonl")
        # line-buffered so a crash leaves a usable file
        fh = open(path, "a", buffering=1, encoding="utf-8")
        return SessionLog(path=path, fh=fh)
    except OSError as e:  # noqa: BLE001
        print(f"{DIM}[session log disabled: {e}]{RESET}")
        return None


class SessionLog:
    """Append-only JSONL session transcript. One JSON object per line.

    Each entry: {"ts": iso8601, "kind": "user|assistant|tool_call|tool_result|info", ...}
    Crash-safe (line-buffered, atomic line writes).
    """

    def __init__(self, path: str, fh) -> None:
        self.path = path
        self._fh = fh
        self._closed = False

    def write(self, kind: str, **fields) -> None:
        if self._closed:
            return
        import datetime as _dt
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "kind": kind, **fields}
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


# Module-level so step() can record tool activity without plumbing it through.
_session_log: SessionLog | None = None


def _run_headless(prompt: str, system_prompt: str | None = None) -> int:
    """One-shot mode: feed `prompt` as a single user turn, run the tool
    loop until the agent stops, persist the session log, exit.

    Used by the agent supervisor (qwen_agent_supervisor.py) to drive
    a saved agent on schedule. Set QWEN_SESSION_LOG_DIR to direct the
    JSONL log somewhere the supervisor can find it.
    """
    global _session_log
    _session_log = _open_session_log()
    if _session_log is not None:
        names = [t["function"]["name"] for t in TOOLS]
        _session_log.write(
            "session_start",
            model=MODEL, url=URL, tools=names,
            cwd=os.getcwd(),
            system_prompt=system_prompt or SYSTEM_PROMPT,
            mode="headless",
        )
    try:
        sys_text = system_prompt or SYSTEM_PROMPT
        messages: list[dict] = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": prompt},
        ]
        # Fresh per-task budgets + URL guard seeding (seed both system and
        # user content so URLs in either are pre-authorized for fetching).
        _CACHED.start_new_task()
        _CACHED.note_text(sys_text)
        _CACHED.note_text(prompt)
        if _session_log:
            _session_log.write("user", content=prompt)
        prev_len = len(messages)
        try:
            run_query(messages)
        except KeyboardInterrupt:
            if _session_log:
                _session_log.write("interrupted")
            return 130
        except urllib.error.URLError as e:
            if _session_log:
                _session_log.write("error", kind_detail="url", message=str(e))
            return 2
        except Exception as e:  # noqa: BLE001
            if _session_log:
                _session_log.write("error", kind_detail=type(e).__name__, message=str(e))
            return 3
        if _session_log:
            for m in messages[prev_len:]:
                role = m.get("role", "")
                if role == "assistant":
                    _session_log.write(
                        "assistant",
                        content=m.get("content"),
                        tool_calls=m.get("tool_calls"),
                        reasoning=m.get("reasoning_content")
                        or m.get("reasoning"),
                    )
                elif role == "tool":
                    _session_log.write(
                        "tool_result",
                        name=m.get("name"),
                        tool_call_id=m.get("tool_call_id"),
                        content=m.get("content"),
                    )
                elif role == "user":
                    content = m.get("content") or ""
                    kind = "harness_nudge" if "[HARNESS" in content else "user"
                    _session_log.write(kind, content=content)
        return 0
    finally:
        if _session_log:
            _session_log.write("session_end")
            _session_log.close()


def main() -> int:
    global _session_log
    # Clear any stale per-PID session-write tracking from a prior process
    # that crashed or was killed.
    for stale in (f"/tmp/qwen_session_writes_{os.getpid()}.json",
                  f"/tmp/qwen_session_done_{os.getpid()}.txt"):
        try:
            os.unlink(stale)
        except OSError:
            pass

    # Iter 36: SIGTERM handler — convert eval-driver timeouts (subprocess
    # killed at 600s) into stub-emit + clean exit so partial findings are
    # captured. Without this, the iter-35 TSM TIMEOUT case writes nothing
    # and the rubric scores 0 on a session that had real partial data.
    import signal as _signal
    _last_messages_ref: list = []
    def _term_handler(signum, frame):
        try:
            if _last_messages_ref:
                _maybe_emergency_stub(_last_messages_ref[0],
                                      reason=f"SIGTERM signum={signum}")
        finally:
            os._exit(143)  # standard SIGTERM exit code
    try:
        _signal.signal(_signal.SIGTERM, _term_handler)
    except (ValueError, OSError):
        pass  # signal handler can't always be installed (e.g., not main thread)
    # Stash the signal handler's reference holder on the module so run_query
    # can populate it as the messages list is built.
    _module = sys.modules[__name__]
    setattr(_module, "_term_messages_ref", _last_messages_ref)

    # --- headless mode: one-shot prompt for the agent supervisor ---
    if "--headless" in sys.argv:
        argv = list(sys.argv)
        argv.remove("--headless")
        prompt = None
        sys_prompt = None
        i = 1
        while i < len(argv):
            a = argv[i]
            if a == "--prompt" and i + 1 < len(argv):
                prompt = argv[i + 1]; i += 2; continue
            if a == "--prompt-file" and i + 1 < len(argv):
                with open(argv[i + 1], encoding="utf-8") as f:
                    prompt = f.read()
                i += 2; continue
            if a == "--system" and i + 1 < len(argv):
                sys_prompt = argv[i + 1]; i += 2; continue
            i += 1
        if not prompt:
            print("[--headless] --prompt or --prompt-file is required",
                  file=sys.stderr)
            return 2
        return _run_headless(prompt, sys_prompt)

    names = [t["function"]["name"] for t in TOOLS]
    print(f"{BOLD}qwen agent{RESET}  model={MODEL}  tools={names}")
    print("type /help for commands, Ctrl-D or /exit to leave\n")

    _session_log = _open_session_log()
    if _session_log is not None:
        print(f"{DIM}[session log: {_session_log.path}]{RESET}\n")
        _session_log.write(
            "session_start",
            model=MODEL, url=URL, tools=names,
            cwd=os.getcwd(), system_prompt=SYSTEM_PROMPT,
        )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            try:
                user = input(f"{BOLD}›{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not user:
                continue
            if user in ("/exit", "/quit"):
                return 0
            if user == "/help":
                print(HELP)
                continue
            if user == "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if _session_log:
                    _session_log.write("clear")
                print("[history cleared]")
                continue
            if user == "/tools":
                for t in TOOLS:
                    f = t["function"]
                    print(f"  {f['name']}: {short(f['description'], 80)}")
                continue
            if user.startswith("/system"):
                text = user[len("/system"):].strip() or SYSTEM_PROMPT
                messages = [{"role": "system", "content": text}]
                if _session_log:
                    _session_log.write("system_replace", system_prompt=text)
                print("[system prompt set]")
                continue

            messages.append({"role": "user", "content": user})
            # Per-task hooks. Each new user prompt gets:
            #   - fresh web_search/web_fetch budgets (cumulative caps would
            #     starve later turns of a long session)
            #   - URL guard seeded with any URLs the user pasted in
            _CACHED.start_new_task()
            _CACHED.note_text(user)
            if _session_log:
                _session_log.write("user", content=user)
            prev_len = len(messages)
            try:
                run_query(messages)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}[interrupted]{RESET}")
                while messages and messages[-1].get("role") in ("assistant", "tool"):
                    last = messages.pop()
                    if last.get("role") == "assistant" and not last.get("tool_calls"):
                        break
                if _session_log:
                    _session_log.write("interrupted")
            except urllib.error.URLError as e:
                print(f"{RED}[error] cannot reach {URL}: {e}{RESET}", file=sys.stderr)
                if _session_log:
                    _session_log.write("error", kind_detail="url", message=str(e))
            except Exception as e:  # noqa: BLE001
                print(f"{RED}[error] {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                if _session_log:
                    _session_log.write("error", kind_detail=type(e).__name__, message=str(e))
            else:
                # Persist any new messages this turn (assistant, tool, AND
                # harness-injected user nudges so we can audit which fired).
                if _session_log:
                    for m in messages[prev_len:]:
                        role = m.get("role", "")
                        if role == "assistant":
                            _session_log.write(
                                "assistant",
                                content=m.get("content"),
                                tool_calls=m.get("tool_calls"),
                                reasoning=m.get("reasoning_content")
                                or m.get("reasoning"),
                            )
                        elif role == "tool":
                            _session_log.write(
                                "tool_result",
                                name=m.get("name"),
                                tool_call_id=m.get("tool_call_id"),
                                content=m.get("content"),
                            )
                        elif role == "user":
                            # Harness-injected nudges (e.g. read-loop, exit-block)
                            content = m.get("content") or ""
                            kind = "harness_nudge" if "[HARNESS" in content else "user"
                            _session_log.write(kind, content=content)
    finally:
        if _session_log:
            _session_log.write("session_end")
            _session_log.close()


if __name__ == "__main__":
    sys.exit(main())

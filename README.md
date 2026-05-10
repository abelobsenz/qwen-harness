# qwen-harness

A local tool-calling agent stack built on **Qwen3.6-35B-A3B** with **DFlash speculative decoding** on Apple Silicon. Runs the model behind a small chat UI; the agent can search the web, fetch SEC filings / arXiv / GitHub, run shell commands, write files, and self-audit numeric claims before finishing.

> **Platform: macOS + Apple Silicon only.** The inference stack uses MLX (Apple's Metal-backed array framework) and `dflash-mlx`. There is no CUDA / Linux build path.

---

## What's in here

```
bin/qwen                  CLI wrapper (start, stop, ui, status, restart, bench)
config/qwen.conf          Daemon config — model paths, KV/cache knobs, agent tunables
scripts/
  agent.py                Agent loop, system prompt, audit gate, loop guard, retries
  agent_tools.py          Tool implementations: web, fs, SEC, arXiv, explore, memory…
  qwen_ui.py              Local web UI (chat + agent supervisor) on :8001
  qwen_proxy.py           Parses upstream tool-call output back into structured form
  _qwen_daemon.py         Supervisor: dflash-serve + qwen-proxy as one unit
  dflash_serve_patched.py Drop-in for `dflash-serve` that installs the APC patch
  apc_patch.py            Multi-turn prompt-cache fix for hybrid Qwen3.6 KV layout
  loop_guard.py, …        Streaming loop / repetition / cache-hit detectors
  agent_graph.py          Multi-agent graph orchestration
  test_*.py               34 unit / integration tests (stdlib-only)
eval_data/
  finance_agent_benchmark.csv  Fixture (vals-ai/finance_agent_benchmark sample)
  run_finance_eval.py     Eval driver with --retry-on-fail
examples/                 Pre-built agent graphs (research, code review, etc.)
experiments/              Probes / unmerged work (paged attention, JSON grammar)
docs/                     Design docs (internal iteration logs are .gitignored)
```

---

## Quick start

### 1. Install

Requires:
- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ (3.14 recommended)
- ~30 GB free disk for models + venv

```bash
git clone https://github.com/abelobsenz/qwen-harness
cd qwen-harness
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull model weights

Two models are needed — the target (~20 GB) and the small DFlash draft (~900 MB):

```bash
mkdir -p models
huggingface-cli download bstnxbt/Qwen3.6-35B-A3B-OptiQ-4bit \
    --local-dir models/Qwen3.6-35B-A3B-OptiQ-4bit
huggingface-cli download bstnxbt/Qwen3.6-35B-A3B-DFlash \
    --local-dir models/Qwen3.6-35B-A3B-DFlash
```

(Replace the repo IDs with whatever you actually built or downloaded — these are placeholders matching the paths in `config/qwen.conf`.)

### 3. Configure

Copy the example config and edit if your paths differ:

```bash
cp config/qwen.conf.example config/qwen.conf
$EDITOR config/qwen.conf
```

Defaults work if you used the layout above.

### 4. Run

```bash
bin/qwen ui                # starts daemon (model + proxy) and the chat UI
# open http://127.0.0.1:8001
```

Other commands:

```bash
bin/qwen start -d          # daemon only, no UI
bin/qwen status            # show daemon / proxy / UI state
bin/qwen restart           # full restart (resets KV cache)
bin/qwen stop
bin/qwen bench             # run perf benchmark suite
scripts/run_tests.py       # unit + integration tests
```

---

## Architecture

```
┌──────────────┐   user prompt   ┌──────────────┐   /v1/chat   ┌──────────────────┐
│  qwen_ui.py  │ ──────────────> │  agent.py    │ ───────────> │  qwen_proxy.py   │
│  :8001       │   tool-call     │  (loop +     │   parsed     │  :8000           │
│  (web UI)    │ <────────────── │   audit      │ <─────────── │  (tool-call      │
└──────────────┘                 │   gate)      │              │   re-parser)     │
                                 │              │              └────────┬─────────┘
                                 │  agent_      │                       │
                                 │  tools.py    │                       v
                                 └──────┬───────┘            ┌──────────────────┐
                                        │ web_fetch          │  dflash_serve_   │
                                        │ web_search         │  patched.py      │
                                        │ sec_filings        │  :8002           │
                                        │ explore            │  (Qwen3.6 +      │
                                        │ memory             │   DFlash + APC)  │
                                        │ ...                └──────────────────┘
                                        v
                                 ┌──────────────┐
                                 │  external    │
                                 │  APIs / web  │
                                 └──────────────┘
```

**Agent guards** (these are the part that took the most iteration to get right):
- **Audit gate:** after a `done()` on a quantitative question, the harness injects one self-audit message asking the model to verify each numeric claim's metadata (entity / period / units / scope / source) before locking the answer.
- **Loop guard:** detects in-stream repetition / cache-hit storms / cold-thinking loops and force-aborts the turn before it burns the eval timeout.
- **Refusal caps** on `web_search` (per-session), `web_fetch` (per-session), per-URL refetch, near-duplicate searches, and `bash curl` bypass attempts. Caps return structured `[REFUSED — …]` markers that the model is trained (via system prompt) to treat as hard stops.
- **Empty-turn nudge / emergency stub:** ensures the agent always emits *something* — partial findings get harvested to a stub file even if the session crashes.

---

## Running evals

The `eval_data/run_finance_eval.py` driver runs a 15-prompt sample from
`vals-ai/finance_agent_benchmark` against the local agent and grades each prompt.

```bash
python eval_data/run_finance_eval.py --n 15 --retry-on-fail 2
```

`--retry-on-fail N` re-runs FAIL/PARTIAL/TIMEOUT prompts up to N more times and keeps the best result. The benchmark CSV ships in this repo; rubric scoring matches the upstream rubric.

---

## Performance knobs

The default `config/qwen.conf` is tuned for an M4 Pro / 48 GB unified memory machine. Key knobs:

| Var | Default | Effect |
|---|---|---|
| `DFLASH_QUANTIZE_KV` | `1` | 8-bit target KV cache (≈ -33% memory, no measured quality loss) |
| `DFLASH_KV_4BIT_FROM_LAYER` | `20` | Mixed-precision: layers ≥ 20 drop to 4-bit KV |
| `DFLASH_PROMPT_CACHE_SLOTS` | `48` | APC trie slots (cross-request KV reuse) |
| `DFLASH_PROMPT_CACHE_GB` | `4` | Byte cap on the trie — evicts when exceeded |
| `DFLASH_PREFILL_CHUNK` | `2048` | Larger chunk improves long-prompt prefill |
| `DFLASH_VERIFY_LEN` | `8` | Speculative-decoding verify length |
| `QWEN_AGENT_COMPACT_AT` | `60000` | Auto-compact agent context at this many tokens |
| `QWEN_UI_MAX_TURNS` | `100` | Per-session turn cap |
| `QWEN_UI_MAX_TOOL_CALLS` | `100` | Per-turn tool-call cap |

See `config/qwen.conf.example` for the full list with rationale.

---

## Caveats

- **Model weights are not in this repo.** You need to download / build them separately.
- **Apple Silicon only.** No CUDA path. `mlx` and `dflash-mlx` are mac-only.
- **Personal-research project.** This is not a hardened product — error paths are best-effort, the proxy is an HTTP shim with no auth, and the supervisor assumes one user / one machine. Run on `127.0.0.1` only.
- **`scripts/agent_tools.py` is large** (~6,700 lines). Splitting is on the to-do list.

---

## License

MIT — see `LICENSE`.

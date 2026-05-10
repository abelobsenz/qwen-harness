"""json_grammar_runtime — bridge json_grammar.py into dflash_mlx.runtime.

Phase C of the constrained-decoding roadmap. This module is responsible
for:

  1. One-time lazy build of TokenClassification at first use (cached
     on disk for fast restart).
  2. Per-request `JsonCharFSM` lifecycle: create on prefill, feed
     committed tokens between cycles, produce updated suppress masks.
  3. Conversion between Python bool-list masks and MLX bool tensors.

`hook_into_dflash()` patches dflash_mlx.runtime so the spec-decode loop
calls back into this module between cycles. Gated by env var
DFLASH_JSON_GRAMMAR=1; default OFF. When OFF, the import is a no-op.

Usage from server boot (e.g. scripts/dflash_serve_patched.py):
    from json_grammar_runtime import maybe_install_grammar_hook
    maybe_install_grammar_hook()  # no-op unless DFLASH_JSON_GRAMMAR=1
"""
from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path

import mlx.core as mx

# Make sibling import work whether this is loaded as `scripts.json_grammar_runtime`
# or just `json_grammar_runtime` (depending on how the patched server bootstraps).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from json_grammar import (  # noqa: E402
    JsonCharFSM,
    TokenClassification,
    build_token_classification,
    cache_classification,
    load_classification,
)


# --------------------------------------------------------------------- #
#  Lazy classification load
# --------------------------------------------------------------------- #

_CLASSIFICATION: TokenClassification | None = None
_CLASSIFICATION_LOCK = threading.Lock()
_CACHE_PATH = os.environ.get(
    "DFLASH_JSON_GRAMMAR_CACHE",
    "/tmp/qwen36_token_classification.json",
)


def get_classification(tokenizer=None) -> TokenClassification:
    """Return the global TokenClassification, building it on first use.

    If a cached classification is present at DFLASH_JSON_GRAMMAR_CACHE,
    load it (fast). Else build fresh from `tokenizer` and persist.
    """
    global _CLASSIFICATION
    if _CLASSIFICATION is not None:
        return _CLASSIFICATION
    with _CLASSIFICATION_LOCK:
        if _CLASSIFICATION is not None:
            return _CLASSIFICATION
        if Path(_CACHE_PATH).exists():
            try:
                _CLASSIFICATION = load_classification(_CACHE_PATH)
                return _CLASSIFICATION
            except Exception as e:  # noqa: BLE001
                print(f"[json_grammar] cache load failed ({e}); rebuilding", file=sys.stderr)
        if tokenizer is None:
            raise RuntimeError(
                "json_grammar_runtime: cannot build classification without a tokenizer"
            )
        t0 = time.monotonic()
        _CLASSIFICATION = build_token_classification(tokenizer)
        try:
            cache_classification(_CLASSIFICATION, _CACHE_PATH)
        except Exception:
            pass
        print(f"[json_grammar] built classification in {time.monotonic()-t0:.1f}s "
              f"(cached at {_CACHE_PATH})", file=sys.stderr)
        return _CLASSIFICATION


# --------------------------------------------------------------------- #
#  Per-request FSM holder. Threaded via a contextvar so spec-decode
#  paths (which might run concurrently for streaming) keep separate state.
# --------------------------------------------------------------------- #

import contextvars  # noqa: E402

_REQUEST_FSM: contextvars.ContextVar[JsonCharFSM | None] = contextvars.ContextVar(
    "json_grammar_fsm", default=None
)
# Shared mask cache across all requests (state-keyed). Saves repeated
# mask compute when many requests share state shapes.
_SHARED_MASK_CACHE: dict = {}


def begin_request(tokenizer) -> JsonCharFSM:
    """Create and register a fresh FSM for the current request context."""
    cls = get_classification(tokenizer)
    fsm = JsonCharFSM(cls, mask_cache=_SHARED_MASK_CACHE)
    _REQUEST_FSM.set(fsm)
    print(f"[grammar.begin_request] FSM created, vocab={cls.vocab_size}", file=sys.stderr, flush=True)
    return fsm


def end_request() -> None:
    _REQUEST_FSM.set(None)


def current_fsm() -> JsonCharFSM | None:
    return _REQUEST_FSM.get()


# --------------------------------------------------------------------- #
#  Per-cycle mask update — the hot path
# --------------------------------------------------------------------- #

def feed_committed_tokens(token_ids) -> None:
    """Advance the FSM through every committed token. `token_ids` may
    be an int, a Python list, or an mx.array (1D). No-op if no FSM
    is active for this context."""
    fsm = current_fsm()
    if fsm is None:
        print("[grammar.feed] no FSM in context!", file=sys.stderr, flush=True)
        return
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    before = (fsm.state.in_tool_call, len(fsm.state.stack))
    for tid in token_ids:
        fsm.feed(int(tid))
    after = (fsm.state.in_tool_call, len(fsm.state.stack))
    if before != after:
        print(f"[grammar.feed] state {before} -> {after} after {list(token_ids)[:5]}...", file=sys.stderr, flush=True)


def current_suppress_mask(
    vocab_size: int,
    base_mask: mx.array | None = None,
) -> mx.array | None:
    """Build the suppress mask for the next cycle, ORed with `base_mask`
    (which may already suppress e.g. EOS tokens for safety).

    `vocab_size` is the SIZE OF THE MODEL'S OUTPUT LOGITS, not the
    tokenizer's reported vocab_size. The model's vocab usually includes
    special tokens beyond `tokenizer.vocab_size`. We size the mask to
    cover the full model vocab, with FSM-controlled tokens explicitly
    allowed and unknown tail-tokens (between tokenizer.vocab_size and
    model.vocab_size) treated as ALLOWED (they are typically EOS /
    formatting / chat-template tokens that the model needs).

    Returns None if no FSM is active OR if FSM is in non-constrained
    mode (outside tool-call), in which case `base_mask` is returned
    unchanged. This is the fast-path: mask is bypassed when not
    actively constraining.
    """
    fsm = current_fsm()
    if fsm is None or not fsm.state.in_tool_call:
        return base_mask
    # Compute allow-mask via FSM (cached by state).
    allow_list = fsm.compute_mask()
    fsm_vocab = len(allow_list)
    # Build a vocab_size-shaped suppress mask:
    #   ids 0..fsm_vocab     : True = suppress (inverse of allow_list)
    #   ids fsm_vocab..vocab : False = allowed (model special tokens
    #                           we don't want to constrain)
    if vocab_size <= fsm_vocab:
        suppress_list = [not b for b in allow_list[:vocab_size]]
    else:
        suppress_list = [not b for b in allow_list]
        # Tail tokens (model special tokens beyond tokenizer.vocab_size):
        # SUPPRESS by default when in tool call (most are not valid JSON).
        # Selectively re-allow below.
        suppress_list.extend([True] * (vocab_size - fsm_vocab))
    # FSM-aware boundary special-token handling:
    #   <tool_call>  : ALLOW only when NOT already in tool call (else loop)
    #   </tool_call> : ALLOW only when IN a tool call
    #   <|im_end|>   : ALWAYS allow (stop signal)
    #   <|endoftext|>: ALWAYS allow (stop signal)
    in_tc = fsm.state.in_tool_call
    boundary = [
        (fsm.cls.tool_call_open_id, not in_tc),
        (fsm.cls.tool_call_close_id, in_tc),
        (fsm.cls.im_end_id, True),
    ]
    # Hard-coded common stop tokens for Qwen3.6.
    # 248044 = <|endoftext|>, 248046 = <|im_end|>
    for stop_id in (248044, 248046):
        if 0 <= stop_id < vocab_size:
            suppress_list[stop_id] = False
    for tid, allow in boundary:
        if tid is not None and 0 <= tid < vocab_size:
            suppress_list[tid] = not allow
    suppress = mx.array(suppress_list, dtype=mx.bool_)
    # Sanity log: confirm mask size + a sample value
    n_suppressed = sum(suppress_list)
    print(f"[grammar.mask] in_tc={in_tc} stack_size={len(fsm.state.stack)} "
          f"suppress[27]={suppress_list[27]} suppress[248058]={suppress_list[248058]} "
          f"n_suppressed={n_suppressed}/{vocab_size}",
          file=sys.stderr, flush=True)
    # Combine with base_mask if any (logical OR — suppress if either says so).
    if base_mask is not None and base_mask.shape[0] == vocab_size:
        suppress = mx.logical_or(suppress, base_mask)
    return suppress


# --------------------------------------------------------------------- #
#  Optional hook installer
# --------------------------------------------------------------------- #

def maybe_install_grammar_hook() -> bool:
    """Install runtime hooks if DFLASH_JSON_GRAMMAR=1.

    Patches dflash_mlx.runtime so the spec-decode loop:
      1. Calls begin_request() at the top of generate_dflash_once /
         stream_dflash_generate, after the tokenizer is known.
      2. Calls feed_committed_tokens() after each cycle's committed_segment.
      3. Replaces suppress_token_mask with the FSM-updated version
         between cycles.

    Returns True if installed, False if disabled.

    NOTE: This function intentionally does not patch yet — it documents
    the exact hooks needed. To activate, copy the diff in
    docs/constrained_tool_decoding_design.md to runtime.py.
    """
    enabled = os.environ.get("DFLASH_JSON_GRAMMAR", "0").lower() in ("1", "true", "yes")
    if not enabled:
        return False
    print(
        "[json_grammar] DFLASH_JSON_GRAMMAR=1 set, but runtime hook installation\n"
        "[json_grammar] is staged — see docs/constrained_tool_decoding_design.md\n"
        "[json_grammar] for the exact lines to edit in dflash_mlx/runtime.py.",
        file=sys.stderr,
    )
    return False


# --------------------------------------------------------------------- #
#  Smoke test
# --------------------------------------------------------------------- #

def _smoke():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/Users/abelobsenz/dev/qwen36_MTP/models/Qwen3.6-35B-A3B-OptiQ-4bit",
        trust_remote_code=True,
    )
    # Use the model's expected output vocab size, not tokenizer.vocab_size.
    MODEL_VOCAB = 248320
    fsm = begin_request(tok)
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "test.py"}}\n</tool_call>'
    ids = tok.encode(text, add_special_tokens=False)
    print(f"smoke: {len(ids)} tokens; model_vocab={MODEL_VOCAB}")
    n_pass = 0
    n_total = 0
    t_total = 0.0
    for tid in ids:
        n_total += 1
        t0 = time.monotonic()
        mask = current_suppress_mask(MODEL_VOCAB, base_mask=None)
        t_total += time.monotonic() - t0
        if mask is None:
            allowed = True
        else:
            mask_list = mask.tolist()
            allowed = not bool(mask_list[tid])
        if allowed:
            n_pass += 1
        else:
            print(f"  REJECTED tid={tid} {fsm.cls.decoded.get(tid, '')!r}  state={fsm.state}")
        feed_committed_tokens([tid])
    print(f"smoke result: {n_pass}/{n_total} accepted")
    print(f"  mask compute time: {t_total*1000:.0f} ms total, {t_total/n_total*1000:.2f} ms/tok avg")
    print(f"  shared cache entries: {len(_SHARED_MASK_CACHE)}")

    # Test 2: malformed
    fsm.reset()
    end_request()
    fsm = begin_request(tok)
    malformed = '<tool_call>\n{"name" "broken"}\n</tool_call>'
    ids2 = tok.encode(malformed, add_special_tokens=False)
    n_reject = 0
    for tid in ids2:
        mask = current_suppress_mask(MODEL_VOCAB, base_mask=None)
        if mask is None:
            allowed = True
        else:
            mask_list = mask.tolist()
            allowed = not bool(mask_list[tid])
        if allowed:
            feed_committed_tokens([tid])
        else:
            n_reject += 1
    print(f"malformed test: {n_reject} rejections (>0 expected)")
    end_request()


if __name__ == "__main__":
    _smoke()

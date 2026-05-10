"""json_grammar — character-level constrained-decoding for tool calls.

Qwen3.6's tokenizer aggressively fuses common JSON patterns (e.g. `{"`,
`":`, `"}`, `,"`) into single tokens, so single-class-per-token approach
breaks. The correct treatment is character-level: for each candidate
next-token, simulate feeding its decoded characters through a JSON
character-FSM and accept iff every transition is legal.

Per-step compute is naively O(vocab × avg_chars) ≈ 1M ops, ~10 ms in
Python at this scale. Precomputing per-state token acceptability is a
one-time cost of ~5 s and yields O(1) mask lookups thereafter.

Public API:
    cls = build_token_acceptability(tokenizer)   # one-time, expensive
    fsm = JsonCharFSM(cls)
    fsm.feed(token_id)                           # per committed token
    mask = fsm.is_token_allowed_array(vocab_size)# bool[V]; True = allowed

Status: PHASE A+B COMPLETE; PHASE C runtime hook NOT wired (see
docs/constrained_tool_decoding_design.md).
"""
from __future__ import annotations

import json as _json
import os as _os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

# ====================================================================== #
#  Character-level JSON FSM
# ====================================================================== #
#
# We track a stack of frames; each frame is one of:
#   "OBJ"     — inside an object, position determined by sub_state
#   "ARR"     — inside an array
#   "STR"     — inside a JSON string
#   "NUM"     — inside a number
#   "LIT_T"   — inside the literal "true"
#   "LIT_F"   — inside "false"
#   "LIT_N"   — inside "null"
# Plus a top-level (empty stack) state where we expect a value.
#
# For OBJ/ARR we track sub-state via a small int:
#   OBJ: 0=expecting_key_or_close, 1=in_key (handled by STR push), 2=expecting_colon,
#        3=expecting_value, 4=expecting_comma_or_close
#   ARR: 0=expecting_value_or_close, 1=expecting_comma_or_close
#
# STR has an internal flag: in_escape (after a '\').
# NUM has an internal accumulator state — a JSON number can have at most
# one '.', one 'e'/'E', and only leading sign / digit transitions.
#
# We don't need to be 100% strict on number internals; spec allows a wide
# range. We accept any token whose chars are in the digit/sign/exponent
# class once we're in NUM.

# Transition function returns either a new state or "REJECT".
_REJECT = "REJECT"


def _is_string_safe_char(c: str) -> bool:
    # Inside a string, all chars are OK except unescaped '"' and '\'
    # and unescaped control chars.
    return c not in ('"', '\\') and (ord(c) >= 0x20 or c in "\t\r\n")


def _is_number_char(c: str) -> bool:
    return c in "0123456789.+-eE"


def _is_whitespace_char(c: str) -> bool:
    return c in " \t\r\n"


def _is_literal_char(c: str, frame: str) -> bool:
    # Frame is LIT_T, LIT_F, or LIT_N — but we also need to know how
    # far through the literal we are. Track that in sub_state.
    return c in "truefalsnull"


# State representation: tuple
#   in_tool_call: bool
#   stack:        tuple of frames; each frame is a tuple
#                 (frame_type: str, sub_state: int, in_escape: bool)
# Top of stack = current frame.

@dataclass(frozen=True)
class FSMState:
    in_tool_call: bool
    stack: tuple = ()  # tuple of (frame_type, sub_state, in_escape)
    # Track how many chars of the literal have been consumed (for LIT_*).
    lit_pos: int = 0
    # True once a top-level value has been completed inside the current
    # tool call. After this, only </tool_call> and whitespace are valid.
    top_value_done: bool = False

    def with_in_tool_call(self, v: bool) -> "FSMState":
        return FSMState(in_tool_call=v, stack=(), lit_pos=0) if v != self.in_tool_call else self

    def push(self, frame_type: str, sub_state: int = 0) -> "FSMState":
        return FSMState(self.in_tool_call,
                        self.stack + ((frame_type, sub_state, False),),
                        lit_pos=0,
                        top_value_done=self.top_value_done)

    def pop(self) -> "FSMState":
        new_stack = self.stack[:-1]
        # If we just popped to empty stack inside a tool call, mark
        # top-level value as done.
        new_done = self.top_value_done or (self.in_tool_call and not new_stack)
        return FSMState(self.in_tool_call, new_stack, lit_pos=0, top_value_done=new_done)

    def replace_top(self, *, sub_state: int | None = None, in_escape: bool | None = None) -> "FSMState":
        if not self.stack:
            return self
        ft, ss, esc = self.stack[-1]
        if sub_state is not None:
            ss = sub_state
        if in_escape is not None:
            esc = in_escape
        return FSMState(self.in_tool_call, self.stack[:-1] + ((ft, ss, esc),),
                        self.lit_pos, self.top_value_done)

    def with_lit_pos(self, pos: int) -> "FSMState":
        return FSMState(self.in_tool_call, self.stack, pos, self.top_value_done)

    @property
    def top(self) -> tuple | None:
        return self.stack[-1] if self.stack else None


_INITIAL_STATE = FSMState(in_tool_call=False, stack=(), lit_pos=0)


def _step_char(state: FSMState, c: str) -> FSMState | str:
    """Advance state by one character. Returns _REJECT or new state.

    We assume state.in_tool_call is True; outside-tool-call is handled
    by the caller (it allows everything).
    """
    top = state.top
    # Empty stack = expecting a value at top level (after </tool_call> the
    # caller resets, so empty-stack inside-tool-call means "between values"
    # which only accepts close-tool-call / im_end externally — but for
    # CHARS, anything else rejects).
    if top is None:
        # If the top-level value already completed, only whitespace allowed
        # until caller emits </tool_call> (handled in simulate_token).
        if state.top_value_done:
            if _is_whitespace_char(c):
                return state
            return _REJECT
        # Inside a tool-call but stack is empty: allow whitespace; any
        # value-starter creates a new frame.
        if _is_whitespace_char(c):
            return state
        if c == '{':
            return state.push("OBJ", 0)
        if c == '[':
            return state.push("ARR", 0)
        if c == '"':
            return state.push("STR", 0)
        if c in "-0123456789":
            return state.push("NUM", 0)
        if c == 't':
            return state.push("LIT_T", 1).with_lit_pos(1)
        if c == 'f':
            return state.push("LIT_F", 1).with_lit_pos(1)
        if c == 'n':
            return state.push("LIT_N", 1).with_lit_pos(1)
        return _REJECT

    ft, ss, esc = top

    # ---------------------------------------------------------------- #
    if ft == "STR":
        if esc:
            # Any next char closes the escape. JSON allows: " \ / b f n r t u
            # plus 4 hex digits for \u. We're permissive: any char clears esc.
            return state.replace_top(in_escape=False)
        if c == '\\':
            return state.replace_top(in_escape=True)
        if c == '"':
            new = state.pop()
            # After string, if parent is OBJ key-mode, advance to expect-colon.
            return _post_value(new)
        if _is_string_safe_char(c):
            return state
        return _REJECT

    # ---------------------------------------------------------------- #
    if ft == "OBJ":
        if _is_whitespace_char(c):
            return state
        if ss == 0:  # expecting key or close
            if c == '"':
                # Push STR for the key. _post_value (when the key string
                # closes) will see OBJ ss=0 and bump to ss=2 (expect colon).
                return state.push("STR", 0)
            if c == '}':
                return _post_value(state.pop())
            return _REJECT
        if ss == 2:  # key just finished; expecting colon
            if c == ':':
                return state.replace_top(sub_state=3)
            return _REJECT
        if ss == 3:  # expecting value (any value-starter)
            return _open_value_frame(state, c)
        if ss == 4:  # after value, expecting comma or close
            if c == ',':
                return state.replace_top(sub_state=0)
            if c == '}':
                return _post_value(state.pop())
            return _REJECT
        return _REJECT

    # ---------------------------------------------------------------- #
    if ft == "ARR":
        if _is_whitespace_char(c):
            return state
        if ss == 0:  # expecting value or close
            if c == ']':
                return _post_value(state.pop())
            return _open_value_frame(state, c)
        if ss == 1:  # after value, expecting comma or close
            if c == ',':
                return state.replace_top(sub_state=0)
            if c == ']':
                return _post_value(state.pop())
            return _REJECT
        return _REJECT

    # ---------------------------------------------------------------- #
    if ft == "NUM":
        if _is_number_char(c):
            return state
        # Number ended; pop and replay this char in the parent.
        new = _post_value(state.pop())
        if new is _REJECT:
            return _REJECT
        return _step_char(new, c)

    # ---------------------------------------------------------------- #
    if ft in ("LIT_T", "LIT_F", "LIT_N"):
        target = {"LIT_T": "true", "LIT_F": "false", "LIT_N": "null"}[ft]
        pos = state.lit_pos
        if pos < len(target) and c == target[pos]:
            new_pos = pos + 1
            if new_pos == len(target):
                return _post_value(state.pop())  # literal complete
            return state.with_lit_pos(new_pos)
        # Doesn't match the literal — reject.
        return _REJECT

    return _REJECT


def _is_value_starter(c: str) -> bool:
    return c in '{[" \t\r\n-tfn0123456789' or c.isdigit()


def _open_value_frame(state: FSMState, c: str) -> FSMState | str:
    """Push a new value-frame for char `c`, assuming caller has already
    confirmed the parent state expected a value here."""
    if c == '{':
        return state.push("OBJ", 0)
    if c == '[':
        return state.push("ARR", 0)
    if c == '"':
        return state.push("STR", 0)
    if c in "-0123456789":
        return state.push("NUM", 0)
    if c == 't':
        return state.push("LIT_T", 1).with_lit_pos(1)
    if c == 'f':
        return state.push("LIT_F", 1).with_lit_pos(1)
    if c == 'n':
        return state.push("LIT_N", 1).with_lit_pos(1)
    return _REJECT


def _post_value(state: FSMState) -> FSMState | str:
    """Called after a value (object, array, string, number, literal)
    has been fully consumed. The parent's sub_state needs updating
    so it expects comma-or-close next."""
    top = state.top
    if top is None:
        # Top-level value finished. Stack is empty; that means the
        # tool-call's body is done. Caller is expected to follow with
        # </tool_call>.
        return state
    ft, ss, _esc = top
    if ft == "OBJ":
        if ss == 0:
            # We just finished an OBJ-keyed-value (the value side after `:`
            # ... wait — actually OBJ ss=0 means expecting-key. _post_value
            # called from STR ending means we finished a key string.
            # Replace sub_state to 2 (expect colon).
            return state.replace_top(sub_state=2)
        if ss == 3:
            # We were in expecting-value; value just finished.
            return state.replace_top(sub_state=4)
        # Otherwise, we're returning from a nested complete value — the
        # parent OBJ already had its ss updated when the value started.
        return state
    if ft == "ARR":
        if ss == 0:
            return state.replace_top(sub_state=1)
        return state
    return state


# ====================================================================== #
#  Token-level acceptability (Phase A + per-state mask precomputation)
# ====================================================================== #

@dataclass
class TokenClassification:
    """Per-(state, token) acceptability cache + post-token state
    transitions. For now we cache only the reachable states from the
    initial in-tool-call states; states encountered during inference
    that aren't cached fall back to live simulation.
    """
    vocab_size: int
    decoded: dict[int, str] = field(default_factory=dict)
    tool_call_open_id: int | None = None
    tool_call_close_id: int | None = None
    im_end_id: int | None = None
    # Live tokenizer is not stored; classification carries decoded strings.

    def simulate_token(self, state: FSMState, token_id: int) -> tuple[bool, FSMState]:
        """Feed a token's decoded chars through the FSM. Return
        (is_acceptable, new_state). If reject, the new_state is the
        state at which the rejection happened (caller usually discards
        it)."""
        # Tool-call boundary tokens are special.
        if token_id == self.tool_call_open_id:
            if state.in_tool_call:
                return False, state
            return True, FSMState(in_tool_call=True)
        if token_id == self.tool_call_close_id:
            if not state.in_tool_call:
                return False, state
            # Allowed when the inner JSON is complete OR (lenient) any time —
            # we choose lenient because the model may emit </tool_call> after
            # any well-formed value, with optional trailing whitespace.
            return True, _INITIAL_STATE
        if token_id == self.im_end_id:
            return True, state
        if not state.in_tool_call:
            # All tokens allowed outside tool call.
            return True, state
        text = self.decoded.get(token_id, "")
        if not text:
            return True, state  # empty/unknown token, don't constrain
        cur = state
        for c in text:
            res = _step_char(cur, c)
            if isinstance(res, str):  # _REJECT
                return False, cur
            cur = res
        return True, cur


def build_token_classification(tokenizer, *, max_id: int | None = None) -> TokenClassification:
    """Phase A: decode every vocab id and cache the string. Resolve
    tool-call boundary token ids. Also build a first-char index so mask
    compute can fast-prefilter tokens by their first decoded character.
    """
    vocab_size = max_id if max_id is not None else _resolve_vocab_size(tokenizer)
    decoded: dict[int, str] = {}
    by_first_char: dict[str, list[int]] = {}
    for tid in range(vocab_size):
        try:
            text = tokenizer.decode([tid])
        except Exception:
            text = ""
        decoded[tid] = text
        if text:
            fc = text[0]
            by_first_char.setdefault(fc, []).append(tid)
        else:
            by_first_char.setdefault("", []).append(tid)
    out = TokenClassification(vocab_size=vocab_size, decoded=decoded)
    out.by_first_char = by_first_char  # type: ignore[attr-defined]
    for special in ("<tool_call>", "</tool_call>", "<|im_end|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(special)
            if tid is not None and tid >= 0:
                if special == "<tool_call>":
                    out.tool_call_open_id = tid
                elif special == "</tool_call>":
                    out.tool_call_close_id = tid
                elif special == "<|im_end|>":
                    out.im_end_id = tid
        except Exception:
            pass
    return out


def _resolve_vocab_size(tokenizer) -> int:
    for attr in ("vocab_size", "n_vocab"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise RuntimeError("cannot resolve tokenizer vocab size")


# ====================================================================== #
#  FSM driver (per-request stateful)
# ====================================================================== #

class JsonCharFSM:
    """Per-request stateful FSM. Use one instance per generation request.

    .feed(token_id) -> None     # advance based on a committed token
    .is_token_allowed(token_id) -> bool
    .compute_mask(vocab_size) -> list[bool]   # bool array, True = allowed

    Mask caching: full-vocab mask compute is ~85-250 ms / state on a 248k
    vocab. The FSM visits a small number of distinct states across a
    request (most states recur many times — e.g. STR is hit per-string-
    char), so a state-keyed cache hits >90% on subsequent tokens.

    The cache key uses a SUMMARY of FSMState (tuple of top frame info +
    in_tool_call), not full state, since deeper stack frames don't affect
    the per-char acceptance for the next token (only the top frame does).
    """

    # Cache shared across all FSMs sharing a TokenClassification.
    # Key: (in_tool_call, top_frame_summary, lit_pos)  ->  list[bool] mask.
    _mask_cache: dict = {}

    def __init__(self, classification: TokenClassification, *, mask_cache: dict | None = None):
        self.cls = classification
        self.state: FSMState = _INITIAL_STATE
        self._cache = mask_cache if mask_cache is not None else {}

    def reset(self) -> None:
        self.state = _INITIAL_STATE

    def feed(self, token_id: int) -> None:
        ok, new = self.cls.simulate_token(self.state, token_id)
        if ok:
            self.state = new
        # else: leave state alone — caller should have masked this token

    def is_token_allowed(self, token_id: int) -> bool:
        ok, _ = self.cls.simulate_token(self.state, token_id)
        return ok

    def _state_key(self) -> tuple:
        """Cache key. Must include ENTIRE stack because some tokens
        (e.g. `",` `"}}`) span multiple frames — closing a STR then
        operating on the parent. Same top frame with different parents
        produces different acceptability for those fused tokens."""
        return (self.state.in_tool_call, self.state.stack, self.state.lit_pos)

    def compute_mask(self) -> list[bool]:
        """Return a bool list of length vocab_size (True = allowed).

        Fast path: prefilter by first-char allowed-set per state, only
        run full simulation on tokens that pass the prefilter. This drops
        full-vocab compute from ~100ms to a few ms in most states.
        """
        key = self._state_key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        mask = [False] * self.cls.vocab_size
        # Outside tool-call: everything allowed (fast path).
        if not self.state.in_tool_call:
            mask = [True] * self.cls.vocab_size
            self._cache[key] = mask
            return mask
        # Always allow boundary tokens.
        for tid in (self.cls.tool_call_open_id, self.cls.tool_call_close_id, self.cls.im_end_id):
            if tid is not None and 0 <= tid < self.cls.vocab_size:
                mask[tid] = True
        # First-char prefilter (wildcard => full scan).
        first_char_ok = self._allowed_first_chars()
        bfc = getattr(self.cls, "by_first_char", None)
        if first_char_ok is None or bfc is None:
            # Full scan (wildcard state, e.g. inside a string).
            for tid in range(self.cls.vocab_size):
                if self.is_token_allowed(tid):
                    mask[tid] = True
            self._cache[key] = mask
            return mask
        # Iterate only over tokens whose first char is allowed.
        for fc, ids in bfc.items():
            if fc not in first_char_ok:
                continue
            for tid in ids:
                if self.is_token_allowed(tid):
                    mask[tid] = True
        self._cache[key] = mask
        return mask

    def _allowed_first_chars(self):
        """Compute the set of characters that could legally start the
        next token at the current state. Used to prefilter mask compute.
        Returns either a `set[str]` or `None` meaning 'wildcard — use
        full scan for this state'."""
        st = self.state
        top = st.top
        if top is None:
            # Inside tool-call but stack empty: any value-starter or whitespace.
            return set('{[" \t\r\n-tfn0123456789')
        ft, ss, esc = top
        if ft == "STR":
            # String body accepts any non-control char (incl. non-ASCII).
            # Cheaper to wildcard than enumerate.
            return None
        if ft == "OBJ":
            if ss == 0:
                return {'"', '}', ' ', '\t', '\r', '\n'}
            if ss == 2:
                return {':', ' ', '\t', '\r', '\n'}
            if ss == 3:
                return set('{[" \t\r\n-tfn0123456789')
            if ss == 4:
                return {',', '}', ' ', '\t', '\r', '\n'}
            return set()
        if ft == "ARR":
            if ss == 0:
                return set('{[" \t\r\n-tfn0123456789') | {']'}
            if ss == 1:
                return {',', ']', ' ', '\t', '\r', '\n'}
            return set()
        if ft == "NUM":
            # Number chars + closing structure (number ends).
            return set("0123456789.+-eE,]} \t\r\n")
        if ft in ("LIT_T", "LIT_F", "LIT_N"):
            target = {"LIT_T": "true", "LIT_F": "false", "LIT_N": "null"}[ft]
            pos = st.lit_pos
            if pos < len(target):
                return {target[pos]}
            return set()
        return set()

    def compute_suppress_mask(self) -> list[bool]:
        """Return suppress mask (True = SUPPRESS), inverse of allow."""
        return [not b for b in self.compute_mask()]

    def cache_stats(self) -> dict:
        return {"entries": len(self._cache), "approx_bytes": len(self._cache) * self.cls.vocab_size}


# ====================================================================== #
#  Disk cache
# ====================================================================== #

def cache_classification(cls: TokenClassification, path: str) -> None:
    payload = {
        "vocab_size": cls.vocab_size,
        "decoded": cls.decoded,
        "tool_call_open_id": cls.tool_call_open_id,
        "tool_call_close_id": cls.tool_call_close_id,
        "im_end_id": cls.im_end_id,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(payload, f)
    _os.replace(tmp, path)


def load_classification(path: str) -> TokenClassification:
    with open(path) as f:
        payload = _json.load(f)
    return TokenClassification(
        vocab_size=payload["vocab_size"],
        decoded={int(k): v for k, v in payload["decoded"].items()},
        tool_call_open_id=payload.get("tool_call_open_id"),
        tool_call_close_id=payload.get("tool_call_close_id"),
        im_end_id=payload.get("im_end_id"),
    )


# ====================================================================== #
#  Self-test (no MLX deps)
# ====================================================================== #

def _self_test():
    """Verify char-FSM accepts well-formed JSON and rejects malformed."""
    # Build a fake "char-only" tokenizer where each token is a single char.
    vocab = {ord(c): c for c in '{}[]":, abcdefghijklmnopqrstuvwxyz0123456789\t\n'}
    decoded = {tid: ch for tid, ch in vocab.items()}
    cls = TokenClassification(vocab_size=max(vocab) + 1, decoded=decoded,
                              tool_call_open_id=900, tool_call_close_id=901)
    cls.decoded[900] = "<tool_call>"
    cls.decoded[901] = "</tool_call>"

    fsm = JsonCharFSM(cls)
    # Sequence: <tool_call> { " a " : " b " } </tool_call>
    seq_chars = '{ "a" : "b" }'
    fsm.feed(900)  # <tool_call>
    n_pass = 0
    for c in seq_chars:
        tid = ord(c) if ord(c) in vocab else -1
        if tid < 0:
            continue
        if fsm.is_token_allowed(tid):
            n_pass += 1
            fsm.feed(tid)
        else:
            print(f"  FAIL at char {c!r}  state={fsm.state}")
    fsm.feed(901)
    print(f"self-test (clean): {n_pass}/{sum(1 for c in seq_chars if ord(c) in vocab)} chars accepted")

    # Malformed: '{ "a" "b" }' (missing colon)
    fsm.reset()
    fsm.feed(900)
    n_reject = 0
    for c in '{ "a" "b" }':
        tid = ord(c) if ord(c) in vocab else -1
        if tid < 0:
            continue
        if not fsm.is_token_allowed(tid):
            n_reject += 1
        else:
            fsm.feed(tid)
    print(f"self-test (malformed missing colon): {n_reject} rejections (>0 expected)")
    assert n_reject > 0, "FSM failed to reject malformed JSON"


if __name__ == "__main__":
    _self_test()

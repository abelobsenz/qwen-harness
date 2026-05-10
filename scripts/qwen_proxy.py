#!/usr/bin/env python3
"""qwen-proxy: tool-call parsing proxy that wraps a dflash-serve upstream.

Sits between the agent and dflash-serve, doing inline what
mlx-openai-server's FunctionParameterToolParser does server-side:

  1. Inject tool descriptions into the system prompt so the model knows
     what's available (dflash-serve doesn't honor the OpenAI `tools` field).
  2. Parse the model's <tool_call><function=name><parameter=arg>val</parameter>
     ...</function></tool_call> output back into OpenAI-format tool_calls.
  3. Strip <think>...</think> reasoning into a separate reasoning_content
     field on the response.

With this proxy in front, the agent's view of the API is identical to
mlx-openai-server, so agent.py can be the same code as the legacy version.

Usage (typically invoked by the qwen launcher):
    python qwen_proxy.py --listen-port 8000 --upstream http://127.0.0.1:8002

Stdlib-only (http.server + urllib) so the MTP venv doesn't pick up extra deps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Local: n-gram-based loop detector. dflash_mlx uses greedy argmax decoding
# with no temperature / top_p / repetition_penalty, so generation can collapse
# into pathological loops. The proxy is the right layer to fix this — it
# already mediates every chat-completion request, and a Python-side detector
# is fast enough to run on every streamed chunk.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_guard import StreamingLoopGuard, check_text, LoopGuardConfig  # noqa: E402

# ---------- regex patterns (mirror FunctionParameterToolParser) -------------
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_REGEX = re.compile(
    r"<function\s*=\s*([^>\s]+)\s*>\s*(.*?)\s*</function>", re.DOTALL
)
PARAMETER_REGEX = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL
)
THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
# Qwen3.6's chat template auto-prepends <think>, so the model only emits
# the CLOSING </think>. Anything before that closer is reasoning_content.
THINK_CLOSE_ONLY = re.compile(r"^\s*(.*?)\s*</think>\s*", re.DOTALL)
# Chat-template control marker that some runtimes leak through.
IM_END_RE = re.compile(r"\s*<\|im_end\|>\s*$", re.DOTALL)

# Marker so we never inject the tool blurb twice into the same conversation.
INJECTED_MARKER = "<<TOOLS_INJECTED_BY_PROXY>>"

# Chat-template control tokens that the model may emit as the last token
# of its turn. Any of these in visible content is the runtime leaking
# template machinery to the user — strip them. Order doesn't matter
# (we replace each independently). Round 31 added <|im_end|>; Round 32
# broadened to siblings that surface in some Qwen3 chat-template
# variants and finetunes.
_CHAT_CONTROL_TOKENS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
)


def _strip_control_tokens(text: str) -> str:
    """Remove every chat-template control token from `text`. Returns
    the same string if none are present (no allocation in the common
    case)."""
    if not text:
        return text
    for tok in _CHAT_CONTROL_TOKENS:
        if tok in text:
            text = text.replace(tok, "")
    return text

logger = logging.getLogger("qwen-proxy")


# ---------- lightweight optimization metrics --------------------------------

_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, int] = {
    "requests_stream": 0,
    "requests_nonstream": 0,
    "requests_rejected_oversize": 0,
    "stream_tool_calls_parsed": 0,
    "stream_tool_call_functions": 0,
    "loop_guard_stream_aborts": 0,
    "loop_guard_stream_final_aborts": 0,
    "loop_guard_nonstream_aborts": 0,
    "loop_guard_nonstream_final_aborts": 0,
    "control_token_chunks_stripped": 0,
}


def _metric_add(name: str, n: int = 1) -> None:
    with _METRICS_LOCK:
        _METRICS[name] = int(_METRICS.get(name, 0)) + int(n)


def _metrics_snapshot() -> dict:
    with _METRICS_LOCK:
        counters = dict(_METRICS)
    loop_cfg = LoopGuardConfig()
    return {
        "counters": counters,
        "config": {
            "compact_schema": _COMPACT_SCHEMA,
            "max_body_mb": _MAX_BODY_MB,
            "loop_guard": {
                "enabled": loop_cfg.enabled,
                "suffix_min_len": loop_cfg.suffix_min_len,
                "suffix_repeats": loop_cfg.suffix_repeats,
                "ngram_window": loop_cfg.ngram_window,
                "ngram_n": loop_cfg.ngram_n,
                "ngram_floor": loop_cfg.ngram_floor,
                "min_text": loop_cfg.min_text,
                "phrase_window": loop_cfg.phrase_window,
                "phrase_min_len": loop_cfg.phrase_min_len,
                "phrase_max_len": loop_cfg.phrase_max_len,
                "phrase_repeats": loop_cfg.phrase_min_repeats,
            },
        },
    }


# ---------- response parsing ------------------------------------------------

def _coerce_value(s: str):
    """Best-effort: parse JSON if it looks like JSON, else return string.
    Mirrors FunctionParameterToolParser._coerce_parameter_value."""
    s = s.strip()
    if not s:
        return ""
    try:
        loaded = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s
    if isinstance(loaded, (str, int, float, bool, list, dict)):
        return loaded
    return s


def parse_tool_calls(content: str) -> list[dict]:
    """Extract OpenAI-format tool_calls from raw model output.

    Handles multiple <tool_call> blocks and multiple <function=...> blocks
    inside each (parallel calls). Arguments are JSON-stringified per the
    OpenAI Chat Completions spec.
    """
    out: list[dict] = []
    for block_match in TOOL_CALL_BLOCK.finditer(content):
        block = block_match.group(1)
        for fn_match in FUNCTION_REGEX.finditer(block):
            name = fn_match.group(1).strip()
            body = fn_match.group(2)
            args: dict = {}
            for pname, pval in PARAMETER_REGEX.findall(body):
                args[pname.strip()] = _coerce_value(pval)
            out.append({
                "id": f"call_{len(out)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
    return out


def split_reasoning(content: str) -> tuple[str, str]:
    """Return (visible_content, reasoning_content) by extracting <think>.

    Two formats handled:
      1. Standard `<think>...</think>` blocks anywhere in the content.
      2. Qwen3.6 chat-template format where the prompt prepends `<think>`
         so the model only emits the closing `</think>` after its reasoning.
    """
    reasoning_chunks: list[str] = []
    visible = content

    # 1. Standard balanced blocks.
    for m in THINK_BLOCK.finditer(visible):
        reasoning_chunks.append(m.group(1))
    visible = THINK_BLOCK.sub("", visible)

    # 2. Bare-closer form (no opening <think>): everything before the
    # leading </think> is reasoning. Only treat the FIRST </think> as the
    # implicit-think closer; later ones are stray tokens.
    m = THINK_CLOSE_ONLY.match(visible)
    if m and "<think>" not in m.group(1):
        reasoning_chunks.append(m.group(1))
        visible = visible[m.end():]

    # Drop any leftover stray </think>, <|im_end|>, and other
    # chat-template control tokens. The IM_END_RE is the original
    # end-of-string-only stripper kept for the trailing-whitespace
    # case; _strip_control_tokens covers the marker appearing
    # anywhere mid-text (e.g. when a generation ends prematurely or
    # the upstream emits the marker as a non-final token).
    visible = visible.replace("</think>", "")
    visible = IM_END_RE.sub("", visible)
    visible = _strip_control_tokens(visible).strip()

    reasoning = _strip_control_tokens(
        "\n".join(c.strip() for c in reasoning_chunks if c.strip()).strip()
    )
    return visible, reasoning


# ---------- request rewriting -----------------------------------------------

def _compact_schema(params: dict) -> str:
    """Render a JSON-schema dict as a terse Python-style signature.

    Verbose JSON schema for one tool is ~600 bytes; the signature form is
    ~80 bytes — ~7× smaller — and arguably easier to read. Saves roughly
    13 KB / 3 200 tokens of system-prompt context across the full toolset.

    Format: `name(arg1: type, arg2: type=default, optional?: type) → required: arg1`

    Type mapping:
      "string"  → str
      "integer" → int
      "number"  → float
      "boolean" → bool
      "array"   → list (of inner type if known: list[str])
      "object"  → dict
      otherwise → "any"

    Per-arg descriptions ARE preserved separately via
    `_compact_param_descriptions` so they can be emitted as indented
    lines beneath the signature — see `format_tool_blurb`. Round 30
    reverted the original "drop arg descriptions" decision: real-world
    use surfaced that some tools (memory_*, web_search, …) carry
    parameter-level guidance the model needs (e.g. example queries,
    valid value ranges) that can't be inferred from the type alone.

    Disable with QWEN_PROXY_COMPACT_SCHEMA=0 to fall back to verbose JSON.
    """
    props = params.get("properties") or {}
    required = set(params.get("required") or [])

    def _ts(p: dict) -> str:
        t = p.get("type")
        if t == "string":
            # Enum becomes a literal-union for clarity.
            enum = p.get("enum")
            if isinstance(enum, list) and enum and all(isinstance(e, str) for e in enum):
                if len(enum) <= 6:
                    return "|".join(repr(e) for e in enum)
            return "str"
        if t == "integer":
            return "int"
        if t == "number":
            return "float"
        if t == "boolean":
            return "bool"
        if t == "array":
            items = p.get("items") or {}
            inner = _ts(items) if items else None
            return f"list[{inner}]" if inner else "list"
        if t == "object":
            return "dict"
        if isinstance(t, list):
            return "|".join(_ts({"type": x}) for x in t)
        return "any"

    args = []
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        ts = _ts(spec)
        if name in required:
            args.append(f"{name}: {ts}")
        else:
            default = spec.get("default")
            if default is not None:
                args.append(f"{name}: {ts}={default!r}")
            else:
                args.append(f"{name}?: {ts}")
    return "(" + ", ".join(args) + ")"


def _compact_param_descriptions(params: dict) -> list[str]:
    """Return one `name=description` line per parameter that has a
    description. Skips parameters with no `description` field so we
    only emit lines that carry information.

    The model gets the signature on the function line and these
    indented descriptions beneath, mirroring the verbose JSON's
    nested `properties[name].description` field at a fraction of the
    bytes.
    """
    props = params.get("properties") or {}
    out: list[str] = []
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        d = spec.get("description")
        if not d or not isinstance(d, str):
            continue
        # Collapse internal whitespace so multi-line schema descriptions
        # render cleanly on one line. Keep it readable.
        flat = " ".join(d.split())
        out.append(f"    {name}={flat}")
    return out


# Env override: set QWEN_PROXY_COMPACT_SCHEMA=0 to keep verbose JSON.
_COMPACT_SCHEMA = os.environ.get(
    "QWEN_PROXY_COMPACT_SCHEMA", "1") not in ("0", "false", "False")

# Maximum request body size in megabytes. A hostile or buggy client
# can declare `Content-Length: 1000000000` and the proxy would read
# 1 GB into memory before any validation. Cap defensively at 50 MB —
# generous for legitimate prompts (a 50 MB JSON body holds ~12 M
# tokens, far past any model's context window) but enough of a ceiling
# to prevent OOM under abusive clients. Set =0 to disable.
_MAX_BODY_MB = int(os.environ.get("QWEN_PROXY_MAX_BODY_MB", "50"))
_MAX_BODY_BYTES = _MAX_BODY_MB * 1024 * 1024 if _MAX_BODY_MB > 0 else 0


def format_tool_blurb(tools: list[dict]) -> str:
    """Build the system-prompt addendum that teaches the model the format.

    Schema rendering is controlled by QWEN_PROXY_COMPACT_SCHEMA (default on).
    Compact mode saves ~13 KB / 3 200 tokens per request vs. verbose JSON
    while preserving tool semantics (Qwen3 has no trouble using compact
    signatures — they're literally just Python-like type hints).
    """
    lines = [
        "",
        "",
        "You have these tools available. To call one, emit EXACTLY:",
        "<tool_call>",
        "<function=tool_name>",
        "<parameter=arg_name>arg_value</parameter>",
        "</function>",
        "</tool_call>",
        "",
        "Multiple <function=...> blocks inside the same <tool_call> are "
        "treated as parallel calls. You may produce reasoning in <think>"
        "..</think> blocks before the tool call. After tool results return,"
        " continue the conversation. When done, respond in plain text with "
        "no further tool_call blocks.",
        "",
        "Tools:",
    ]
    for t in tools:
        f = t.get("function", {}) or {}
        params = f.get("parameters", {}) or {}
        name = f.get("name")
        desc = f.get("description", "")
        if _COMPACT_SCHEMA and params.get("properties"):
            sig = _compact_schema(params)
            lines.append(f"- {name}{sig}: {desc}")
            # Emit per-parameter descriptions on indented lines
            # beneath the signature. Preserves the same guidance
            # the verbose JSON had (e.g. example query strings,
            # valid value ranges) at a small token cost vs sig-only.
            lines.extend(_compact_param_descriptions(params))
        elif params.get("properties"):
            lines.append(f"- {name}: {desc}")
            lines.append(
                f"  schema: {json.dumps(params, separators=(',', ':'), ensure_ascii=False)}"
            )
        else:
            lines.append(f"- {name}(): {desc}")
    return "\n".join(lines)


def _stringargs_to_dict(messages: list[dict]) -> None:
    """In-place: convert any tool_call.arguments from JSON string to dict.

    OpenAI Chat Completions spec gives `arguments` as a JSON string. The
    Qwen3.6 chat template uses `tool_call.arguments|items` to iterate args
    as a mapping, which fails on a string. Mirrors what mlx-openai-server's
    GLM4MoEMessageConverter._convert_tool_calls does.
    """
    for m in messages:
        tcs = m.get("tool_calls")
        if not tcs or not isinstance(tcs, list):
            continue
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not fn or not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    fn["arguments"] = {}


def transform_request(req: dict) -> dict:
    """Inject tool blurb into system message; strip OpenAI-only fields;
    convert tool_call.arguments string→dict for chat-template compatibility."""
    tools = req.pop("tools", None) or []
    req.pop("tool_choice", None)

    msgs = req.get("messages") or []
    # Always normalize arguments — agents may have history with string args
    # even on requests that don't carry a tools field.
    _stringargs_to_dict(msgs)

    if not tools:
        req["messages"] = msgs
        return req

    blurb = format_tool_blurb(tools)
    if msgs and msgs[0].get("role") == "system":
        existing = msgs[0].get("content") or ""
        if INJECTED_MARKER not in existing:
            msgs[0] = {
                **msgs[0],
                "content": existing + blurb + "\n" + INJECTED_MARKER,
            }
    else:
        msgs.insert(0, {
            "role": "system",
            "content": blurb + "\n" + INJECTED_MARKER,
        })
    req["messages"] = msgs
    return req


def transform_response(body: bytes) -> bytes:
    """Parse model output → OpenAI tool_calls + reasoning_content."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body
    for choice in data.get("choices") or []:
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if not content:
            continue
        # Already-structured upstream? leave alone.
        if msg.get("tool_calls"):
            continue
        tool_calls = parse_tool_calls(content)
        visible, reasoning = split_reasoning(content)
        # Strip tool-call blocks from visible content
        visible = TOOL_CALL_BLOCK.sub("", visible).strip()
        # finish_reason: "tool_calls" if we parsed any (mirror OpenAI)
        if tool_calls:
            choice["finish_reason"] = "tool_calls"
        msg["content"] = visible if visible else None
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning:
            msg["reasoning_content"] = reasoning
        choice["message"] = msg
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


# ---------- streaming → non-streaming assembler -----------------------------
# When the client asks for non-streaming output, we still drive upstream in
# streaming mode so the loop guard can early-abort. This assembler merges
# SSE chunk deltas back into a single chat.completion JSON identical in
# shape to what dflash-serve would have returned for stream=False.

class _StreamAssembler:
    def __init__(self) -> None:
        self.id: str | None = None
        self.created: int | None = None
        self.model: str | None = None
        self._content_parts: list[str] = []
        self._last_chunk: str = ""
        self._last_finish_reason: str | None = None

    def absorb(self, obj: dict) -> None:
        if self.id is None:
            self.id = obj.get("id")
        if self.created is None:
            self.created = obj.get("created")
        if self.model is None:
            self.model = obj.get("model")
        choices = obj.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        # We only fold visible content here. <think>... and tool blocks are
        # parsed downstream by transform_response, which expects to see them
        # in `content`. The reasoning_content delta (if dflash-serve emitted
        # it) is also concatenated into content for downstream parsing —
        # split_reasoning will pull it back out.
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        merged = content + reasoning
        if merged:
            self._content_parts.append(merged)
            self._last_chunk = merged
        if choices[0].get("finish_reason"):
            self._last_finish_reason = choices[0]["finish_reason"]

    def last_chunk_text(self) -> str:
        return self._last_chunk

    def full_text(self) -> str:
        return "".join(self._content_parts)

    def build_chat_completion(self, aborted_reason: str | None = None) -> str:
        text = self.full_text()
        if aborted_reason is not None:
            # Append a short, model-visible marker so the agent knows the
            # generation was cut off by the safety net (rather than a
            # natural completion). The agent's downstream parsing will
            # see this on .content.
            text = (text.rstrip() +
                    f"\n\n[{aborted_reason} — output stopped early]")
        finish = self._last_finish_reason or (
            "stop" if aborted_reason is None else "length"
        )
        body = {
            "id": self.id or "chatcmpl-assembled",
            "object": "chat.completion",
            "created": self.created or 0,
            "model": self.model or "",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }],
        }
        return json.dumps(body, ensure_ascii=False)


_TC_OPEN = "<tool_call>"
_TC_CLOSE = "</tool_call>"


def _chunk_base(obj: dict, *, delta: dict, finish_reason: str | None = None) -> dict:
    choices = obj.get("choices") or [{}]
    choice = choices[0] if choices else {}
    out: dict = {
        "id": obj.get("id", ""),
        "object": obj.get("object", "chat.completion.chunk"),
        "model": obj.get("model", ""),
        "choices": [{
            "index": choice.get("index", 0),
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if obj.get("created") is not None:
        out["created"] = obj.get("created")
    return out


class _StreamingToolCallAssembler:
    """Suppress streamed XML tool calls and emit structured tool_call chunks.

    dflash streams the literal Qwen XML syntax token-by-token. That works for
    non-streaming because transform_response sees the complete text, but
    streaming clients otherwise display XML until the final frame. This small
    state machine holds back a possible `<tool_call>` boundary, buffers the
    XML block, and emits a complete OpenAI-style delta.tool_calls frame once
    the closing tag arrives.
    """

    def __init__(self) -> None:
        self.pending = ""
        self.in_tool_call = False
        self.tool_buf = ""
        self.completed = False

    def feed(self, text: str) -> tuple[list[str], list[dict] | None]:
        if self.completed or not text:
            return [], None
        visible: list[str] = []
        buf = self.pending + text
        self.pending = ""
        while buf:
            if self.in_tool_call:
                self.tool_buf += buf
                idx = self.tool_buf.find(_TC_CLOSE)
                if idx == -1:
                    return visible, None
                self.tool_buf = self.tool_buf[: idx + len(_TC_CLOSE)]
                tool_calls = parse_tool_calls(self.tool_buf)
                self.in_tool_call = False
                self.completed = True
                return visible, tool_calls

            idx = buf.find(_TC_OPEN)
            if idx == -1:
                keep = max(0, len(buf) - (len(_TC_OPEN) - 1))
                if keep:
                    visible.append(buf[:keep])
                    self.pending = buf[keep:]
                else:
                    self.pending = buf
                return visible, None

            if idx:
                visible.append(buf[:idx])
            self.in_tool_call = True
            self.tool_buf = _TC_OPEN
            buf = buf[idx + len(_TC_OPEN):]
        return visible, None

    def flush_visible(self) -> str:
        if self.in_tool_call or self.completed:
            return ""
        out = self.pending
        self.pending = ""
        return out


# ---------- HTTP handler ----------------------------------------------------

class ProxyHandler(BaseHTTPRequestHandler):
    upstream_base: str = "http://127.0.0.1:8002"
    request_timeout: float = 600.0

    def log_message(self, fmt, *args):  # noqa: A003
        logger.debug(fmt, *args)

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._forward_passthrough(method="POST")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, b'{"error":"invalid Content-Length header"}')
            return
        # Reject oversized bodies BEFORE reading them — otherwise a
        # client that lies about Content-Length can force the proxy to
        # allocate up to that many bytes. Cap at QWEN_PROXY_MAX_BODY_MB
        # (default 50 MB; set to 0 to disable). 413 = Payload Too Large.
        if _MAX_BODY_BYTES and length > _MAX_BODY_BYTES:
            _metric_add("requests_rejected_oversize")
            msg = (f'{{"error":"request body exceeds '
                   f'{_MAX_BODY_MB} MB cap","limit_mb":{_MAX_BODY_MB}}}')
            self._send(413, msg.encode())
            return
        body = self.rfile.read(length) if length else b""
        try:
            req = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            # `json.loads` decodes bytes first; non-UTF-8 input raises
            # UnicodeDecodeError, not JSONDecodeError. Catching only
            # JSONDecodeError dropped the connection without a response,
            # which is a worse failure mode than 400 — the client gets
            # no actionable signal. ValueError covers any future
            # json-parser variants. Caught + caught at test_proxy_security.
            self._send(400, b'{"error":"invalid json body"}')
            return
        is_stream = bool(req.get("stream", False))
        _metric_add("requests_stream" if is_stream else "requests_nonstream")
        req = transform_request(req)
        try:
            up_req = urllib.request.Request(
                self.upstream_base + self.path,
                data=json.dumps(req, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if is_stream:
                # SSE pass-through with loop-guard. We forward chunks as they
                # arrive (preserves time-to-first-token), but accumulate
                # decoded content in a StreamingLoopGuard. When the guard
                # fires we close the upstream connection (Python GC then
                # raises GeneratorExit inside dflash-serve which triggers
                # its early-stop) and emit a synthetic stop frame with a
                # diagnostic delta so the client sees what happened.
                resp = urllib.request.urlopen(up_req, timeout=self.request_timeout)
                self.send_response(resp.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                guard = StreamingLoopGuard()
                tool_assembler = _StreamingToolCallAssembler()
                aborted = False
                tool_call_completed = False
                try:
                    for line in resp:
                        # Snoop the SSE payload to feed the guard.
                        # Also strip chat-template control tokens
                        # (<|im_end|>) that the model emits as the
                        # last content token of its turn — they're
                        # supposed to be a stop signal, not visible
                        # output. Without stripping here, streaming
                        # clients (chat REPL, qwen-ui) display the
                        # literal "<|im_end|>" string at the end of
                        # every assistant response.
                        rewritten: bytes | None = None
                        out_frames: list[dict] | None = None
                        try:
                            txt = line.decode("utf-8", errors="ignore")
                            if txt.startswith("data: "):
                                payload = txt[len("data: "):].strip()
                                if payload == "[DONE]" and not tool_call_completed:
                                    tail = tool_assembler.flush_visible()
                                    if tail:
                                        tail_frame = _chunk_base(
                                            {"choices": [{"index": 0}]},
                                            delta={"content": tail},
                                        )
                                        self.wfile.write(
                                            f"data: {json.dumps(tail_frame, ensure_ascii=False)}\n\n".encode("utf-8")
                                        )
                                if payload and payload != "[DONE]":
                                    obj = json.loads(payload)
                                    delta = (obj.get("choices") or [{}])[0].get("delta", {})
                                    # Strip ALL chat-template control
                                    # tokens (<|im_end|>, <|im_start|>,
                                    # <|endoftext|>) from any text field.
                                    # If anything was stripped, we
                                    # re-serialize the chunk and emit
                                    # the cleaned version below.
                                    cleaned = False
                                    for key in ("content", "reasoning_content", "reasoning"):
                                        v = delta.get(key)
                                        if isinstance(v, str):
                                            new_v = _strip_control_tokens(v)
                                            if new_v != v:
                                                delta[key] = new_v
                                                cleaned = True
                                    if cleaned:
                                        _metric_add("control_token_chunks_stripped")
                                    chunk_text = (
                                        (delta.get("content") or "")
                                        + (delta.get("reasoning_content") or "")
                                        + (delta.get("reasoning") or "")
                                    )
                                    content = delta.get("content")
                                    if isinstance(content, str):
                                        visible_parts, tool_calls = tool_assembler.feed(content)
                                        visible = "".join(visible_parts)
                                        chunk_text = (
                                            visible
                                            + (delta.get("reasoning_content") or "")
                                            + (delta.get("reasoning") or "")
                                        )
                                        if tool_calls is not None:
                                            tool_call_completed = True
                                            _metric_add("stream_tool_calls_parsed")
                                            _metric_add(
                                                "stream_tool_call_functions",
                                                len(tool_calls),
                                            )
                                        out_frames = []
                                        if visible:
                                            visible_delta = dict(delta)
                                            visible_delta["content"] = visible
                                            out_frames.append(
                                                _chunk_base(obj, delta=visible_delta)
                                            )
                                        elif any(k in delta for k in ("role", "reasoning_content", "reasoning")):
                                            non_content_delta = {
                                                k: v for k, v in delta.items()
                                                if k != "content"
                                            }
                                            if non_content_delta:
                                                out_frames.append(
                                                    _chunk_base(obj, delta=non_content_delta)
                                                )
                                        if tool_calls is not None:
                                            out_frames.append(
                                                _chunk_base(
                                                    obj,
                                                    delta={"tool_calls": tool_calls},
                                                    finish_reason="tool_calls",
                                                )
                                            )
                                    elif (obj.get("choices") or [{}])[0].get("finish_reason") and not tool_call_completed:
                                        tail = tool_assembler.flush_visible()
                                        if tail:
                                            out_frames = [
                                                _chunk_base(obj, delta={"content": tail}),
                                                obj,
                                            ]
                                    if out_frames is None and cleaned:
                                        rewritten = (
                                            f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
                                        ).encode("utf-8")
                                    if chunk_text:
                                        report = guard.observe(chunk_text)
                                        if report.triggered:
                                            aborted = True
                                            _metric_add("loop_guard_stream_aborts")
                                            logger.warning(
                                                "loop-guard: aborting stream (reason=%s, %s)",
                                                report.reason, report.detail,
                                            )
                                            # Synthetic abort frame: tells the
                                            # client the model looped + stopped.
                                            abort_payload = {
                                                "id": obj.get("id", ""),
                                                "object": "chat.completion.chunk",
                                                "model": obj.get("model", ""),
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"content":
                                                              f"\n\n[loop-guard: aborted "
                                                              f"({report.reason}) — "
                                                              f"the model fell into a "
                                                              f"repetition loop. "
                                                              f"Try rephrasing or asking "
                                                              f"a more specific question.]"},
                                                    "finish_reason": "stop",
                                                }],
                                            }
                                            try:
                                                self.wfile.write(
                                                    f"data: {json.dumps(abort_payload)}\n\n".encode("utf-8")
                                                )
                                                self.wfile.write(b"data: [DONE]\n\n")
                                                self.wfile.flush()
                                            except (BrokenPipeError, ConnectionResetError):
                                                pass
                                            break
                        except (json.JSONDecodeError, ValueError, KeyError):
                            pass
                        # Emit structured tool-call frames if the XML parser
                        # consumed/replaced this chunk; otherwise emit the
                        # cleaned line if we rewrote it (to strip control
                        # tokens), or pass through.
                        if out_frames is not None:
                            for frame in out_frames:
                                self.wfile.write(
                                    f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode("utf-8")
                                )
                        else:
                            self.wfile.write(rewritten if rewritten is not None else line)
                        self.wfile.flush()
                        if tool_call_completed:
                            try:
                                self.wfile.write(b"data: [DONE]\n\n")
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                pass
                            break
                except (BrokenPipeError, ConnectionResetError):
                    pass
                else:
                    # End-of-stream final check (catches loops in the trailing
                    # < check_every chars). Only runs if we exited the loop
                    # cleanly (no exception, no mid-stream abort).
                    if not aborted and not tool_call_completed:
                        tail = tool_assembler.flush_visible()
                        if tail:
                            try:
                                tail_frame = _chunk_base(
                                    {"choices": [{"index": 0}]},
                                    delta={"content": tail},
                                )
                                self.wfile.write(
                                    f"data: {json.dumps(tail_frame, ensure_ascii=False)}\n\n".encode("utf-8")
                                )
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                pass
                        final_report = guard.finalize()
                        if final_report.triggered:
                            _metric_add("loop_guard_stream_final_aborts")
                            logger.warning(
                                "loop-guard: end-of-stream final check fired "
                                "(reason=%s, %s) — emitting marker",
                                final_report.reason, final_report.detail,
                            )
                            try:
                                marker = {
                                    "id": "chatcmpl-finalcheck",
                                    "object": "chat.completion.chunk",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content":
                                                  f"\n\n[loop-guard: detected loop "
                                                  f"at end-of-stream "
                                                  f"({final_report.reason})]"},
                                        "finish_reason": "stop",
                                    }],
                                }
                                self.wfile.write(
                                    f"data: {json.dumps(marker)}\n\n".encode("utf-8")
                                )
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                pass
                finally:
                    if aborted or tool_call_completed:
                        try:
                            resp.close()  # signals upstream to stop generation
                        except Exception:  # noqa: BLE001
                            pass
                    else:
                        resp.close()
                return
            # Non-streaming path: switch to upstream-streaming under the
            # hood so we can early-abort on loops without wasting tokens.
            # The agent uses stream=False (line 188 of agent.py), so this
            # is the hot path. We assemble the full response ourselves and
            # return it as a single chat.completion JSON.
            up_req_streamed = urllib.request.Request(
                self.upstream_base + self.path,
                data=json.dumps({**req, "stream": True},
                                ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            assembled = _StreamAssembler()
            resp = urllib.request.urlopen(up_req_streamed,
                                          timeout=self.request_timeout)
            code = resp.status
            guard = StreamingLoopGuard()
            aborted_reason: str | None = None
            try:
                for line in resp:
                    txt = line.decode("utf-8", errors="ignore")
                    if not txt.startswith("data: "):
                        continue
                    payload = txt[len("data: "):].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    assembled.absorb(obj)
                    chunk_text = assembled.last_chunk_text()
                    if chunk_text:
                        report = guard.observe(chunk_text)
                        if report.triggered:
                            _metric_add("loop_guard_nonstream_aborts")
                            aborted_reason = (
                                f"loop-guard: {report.reason} "
                                f"({report.detail})"
                            )
                            logger.warning(
                                "loop-guard: aborting non-stream (reason=%s, %s)",
                                report.reason, report.detail,
                            )
                            break
                # End-of-stream final check: catch loops that developed in
                # the trailing < check_every chars (or where total length
                # never crossed a check_every threshold).
                if aborted_reason is None:
                    final_report = guard.finalize()
                    if final_report.triggered:
                        _metric_add("loop_guard_nonstream_final_aborts")
                        aborted_reason = (
                            f"loop-guard: {final_report.reason} "
                            f"({final_report.detail})"
                        )
                        logger.warning(
                            "loop-guard: end-of-stream final check fired "
                            "(reason=%s, %s)",
                            final_report.reason, final_report.detail,
                        )
            finally:
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
            # Build the consolidated response just like dflash-serve
            # would have, then run it through transform_response so
            # tool calls get parsed.
            up_body = assembled.build_chat_completion(
                aborted_reason=aborted_reason,
            ).encode("utf-8")
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
            return
        except urllib.error.URLError as e:
            self._send(502, json.dumps({"error": f"upstream: {e}"}).encode("utf-8"))
            return
        new_body = transform_response(up_body)
        self._send(code, new_body)

    # Cache the /v1/models response since it's metadata that doesn't change
    # while the proxy is alive. dflash serializes everything, so under chat
    # load /v1/models would otherwise queue behind in-flight generation —
    # which broke agent_tools._resolve_model_id() (timed out, fell back to
    # alias, proxy 400'd the alias). One-shot cache, never invalidated.
    _MODELS_CACHE: bytes | None = None
    _MODELS_CACHE_LOCK = threading.Lock()

    def do_GET(self):  # noqa: N802
        if self.path == "/debug/metrics":
            body = json.dumps(_metrics_snapshot(), ensure_ascii=False).encode("utf-8")
            self._send(200, body)
            return
        if self.path == "/v1/models":
            self._serve_models_cached()
            return
        self._forward_passthrough(method="GET")

    def _serve_models_cached(self):
        cache = ProxyHandler._MODELS_CACHE
        if cache is None:
            with ProxyHandler._MODELS_CACHE_LOCK:
                cache = ProxyHandler._MODELS_CACHE
                if cache is None:
                    try:
                        up_req = urllib.request.Request(
                            self.upstream_base + "/v1/models", method="GET"
                        )
                        with urllib.request.urlopen(up_req, timeout=self.request_timeout) as r:
                            ProxyHandler._MODELS_CACHE = cache = r.read()
                    except urllib.error.HTTPError as e:
                        self._send(e.code, e.read())
                        return
                    except urllib.error.URLError as e:
                        self._send(502, json.dumps({"error": f"upstream: {e}"}).encode("utf-8"))
                        return
        self._send(200, cache)

    def _forward_passthrough(self, method: str):
        try:
            up_req = urllib.request.Request(
                self.upstream_base + self.path, method=method
            )
            with urllib.request.urlopen(up_req, timeout=self.request_timeout) as r:
                self._send(r.status, r.read())
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
        except urllib.error.URLError as e:
            self._send(502, json.dumps({"error": f"upstream: {e}"}).encode("utf-8"))

    def _send(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------- server ----------------------------------------------------------

def wait_for_upstream(base: str, timeout: float = 180.0) -> bool:
    """Block until upstream's /v1/models responds, or timeout."""
    import time
    deadline = time.monotonic() + timeout
    url = base.rstrip("/") + "/v1/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=8000)
    ap.add_argument("--upstream", default="http://127.0.0.1:8002",
                    help="dflash-serve base URL")
    ap.add_argument("--wait-upstream", type=float, default=0.0,
                    help="block this many seconds for upstream readiness on startup")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="qwen-proxy %(levelname)s: %(message)s",
    )

    if args.wait_upstream > 0:
        logger.info("waiting up to %ss for upstream %s", args.wait_upstream, args.upstream)
        if not wait_for_upstream(args.upstream, args.wait_upstream):
            logger.error("upstream %s did not become ready", args.upstream)
            return 2

    ProxyHandler.upstream_base = args.upstream.rstrip("/")

    # Quiet the framework's default `Exception occurred during processing
    # of request from …` traceback that fires on every client disconnect
    # (BrokenPipeError on a keep-alive readline). These are routine in
    # SSE streaming use — the agent CLI aborts mid-stream when the user
    # hits Ctrl-C, the qwen-ui aborts when the user hits Esc — and we
    # already handle the disconnect cleanly in do_POST. The default
    # tracebacks are pure log noise and obscure real errors.
    class _QuietThreadingHTTPServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):  # noqa: D401
            import sys as _sys
            exc = _sys.exc_info()[1]
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, OSError)):
                return
            super().handle_error(request, client_address)

    server = _QuietThreadingHTTPServer(
        (args.listen_host, args.listen_port), ProxyHandler)

    # Startup banner — prints the active config so the user can confirm
    # on `qwen restart` that the proxy is running the expected build.
    # Without this, "did the proxy actually pick up my fix?" was an
    # opaque question requiring code-reading. Round 33 added this after
    # a real-world case where a user reported a fix wasn't taking
    # effect; the answer turned out to be "the daemon hadn't been
    # restarted." With the banner, that's visible in the first 5 lines
    # of `logs/server.log` after every restart.
    from loop_guard import LoopGuardConfig as _LGC
    _lg = _LGC()
    logger.info("=" * 60)
    logger.info("qwen-proxy starting (PID %d)", os.getpid())
    logger.info("  listen:   %s:%s", args.listen_host, args.listen_port)
    logger.info("  upstream: %s", args.upstream)
    logger.info("  loop_guard:    enabled=%s suffix=%d×%d ngram=%d/%.2f window=%d",
                _lg.enabled, _lg.suffix_min_len, _lg.suffix_repeats,
                _lg.ngram_n, _lg.ngram_floor, _lg.ngram_window)
    logger.info("  compact_schema: %s", _COMPACT_SCHEMA)
    logger.info("  max_body_mb:    %d (0 = unlimited)", _MAX_BODY_MB)
    logger.info("  control-token strip: %s", ", ".join(_CHAT_CONTROL_TOKENS))
    logger.info("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

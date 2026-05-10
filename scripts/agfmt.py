"""scripts/agfmt.py — Agent Graph Format.

A line-oriented format for compact, robust communication between agents in
a graph. Compared to JSON it:

  - Saves tokens on prose-heavy outputs: no escaping of " or \\n, no
    enclosing quotes around long strings.
  - Saves tokens on lists of short identifiers: one per line, no commas,
    no surrounding quotes.
  - Stays JSON for genuinely structured bits (numeric tables, nested objects)
    so the model still gets to use what it tokenizes most efficiently.
  - Is easier for a small/quantized model to produce reliably: no nested
    bracket tracking, no comma rules, no quote-escape decisions.

# Spec

    @<name>[:<type>]
    <content lines>
    @<name>[:<type>]
    <content lines>
    @END

Sections are separated by lines that start with `@` followed by an
identifier ([A-Za-z_][A-Za-z0-9_]*). The optional type tag is one of:

    :j      JSON value (one line or multiple lines that together parse)
    :l      list — one item per line, leading "- " is optional, blank lines skipped
    :t      text — raw multiline, preserved verbatim (default if tag omitted)
    :n      number — int or float, single line
    :b      bool — "true" or "false" (case insensitive), single line
    :kv     key:value pairs, one per line, "key: value"

`@END` is optional — encountering EOF terminates the last section. Lines
before the first `@` are ignored (lets the model emit an unfenced preamble
without breaking the parser, though we discourage that).

# Why "@" and short tags

Most BPE tokenizers fold "@name:j\\n" into 3-4 tokens. JSON's "{\"name\":"
is 4-6 tokens before any value content. The win compounds on payloads with
several fields.

# API

    encode({"facts": {"a": 1}, "notes": "ok"})
        →  "@facts:j\\n{\"a\": 1}\\n@notes:t\\nok\\n@END\\n"

    decode("@facts:j\\n{\"a\": 1}\\n@notes:t\\nok\\n")
        →  {"facts": {"a": 1}, "notes": "ok"}

    output_template(["facts:j", "notes:t", "score:n"])
        →  small Markdown block to drop in a system prompt
"""

from __future__ import annotations

import json
import re
from typing import Any


# Allow @SECTION at the very start of a line. The header captures (name, tag).
_HEADER_RE = re.compile(r"^@([A-Za-z_][A-Za-z0-9_]*)(?::([A-Za-z]+))?\s*$")
_END_RE = re.compile(r"^@END\s*$")


def _autodetect_type(value: Any) -> str:
    """Pick the most efficient type tag for a Python value.

    Heuristic favours raw-text where possible (no escaping = fewer tokens),
    falls back to JSON for nested or numeric structures.
    """
    if isinstance(value, bool):
        return "b"
    if isinstance(value, (int, float)):
        return "n"
    if isinstance(value, str):
        return "t"
    if isinstance(value, list):
        # If every element is a short identifier-like string, use list form.
        if all(isinstance(x, str) and "\n" not in x and len(x) < 200 for x in value):
            return "l"
        return "j"
    if isinstance(value, dict):
        return "j"
    return "j"


def _encode_section(name: str, value: Any, tag: str | None = None) -> str:
    if tag is None:
        tag = _autodetect_type(value)
    head = f"@{name}:{tag}"
    if tag == "j":
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    elif tag == "l":
        if not isinstance(value, list):
            raise TypeError(f"section {name!r}: tag 'l' needs list, got {type(value).__name__}")
        body = "\n".join(str(x) for x in value)
    elif tag == "t":
        body = "" if value is None else str(value)
    elif tag == "n":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"section {name!r}: tag 'n' needs number")
        body = str(value)
    elif tag == "b":
        body = "true" if bool(value) else "false"
    elif tag == "kv":
        if not isinstance(value, dict):
            raise TypeError(f"section {name!r}: tag 'kv' needs dict")
        body = "\n".join(f"{k}: {v}" for k, v in value.items())
    else:
        raise ValueError(f"unknown tag {tag!r}")
    if body:
        return head + "\n" + body + "\n"
    return head + "\n"


def encode(d: dict[str, Any], *, types: dict[str, str] | None = None,
           order: list[str] | None = None,
           include_end: bool = True) -> str:
    """Encode a dict to AGFMT.

    `types` lets you pin a specific tag per key (otherwise auto-detected).
    `order` lets you fix section order (default = insertion order).
    """
    types = types or {}
    keys = order if order is not None else list(d.keys())
    out = []
    for k in keys:
        if k not in d:
            continue
        out.append(_encode_section(k, d[k], types.get(k)))
    if include_end:
        out.append("@END\n")
    return "".join(out)


def _decode_section(tag: str | None, lines: list[str]) -> Any:
    """Convert raw section body lines into a Python value per tag.

    The decoder is intentionally tolerant — model output isn't always
    perfectly clean. We strip Markdown fences from JSON, accept "- item"
    or bare "item" in lists, and collapse trailing whitespace in text.
    """
    raw = "\n".join(lines)
    if tag in (None, "t"):
        # Strip a single trailing newline but preserve internal whitespace.
        return raw.rstrip("\n")
    if tag == "j":
        body = raw.strip()
        # Tolerate ```json fences if the model added them.
        if body.startswith("```"):
            body = body[3:]
            if body.lower().startswith("json"):
                body = body[4:]
            body = body.strip()
            if body.endswith("```"):
                body = body[:-3].rstrip()
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"AGFMT json section did not parse: {e}\n--- body ---\n{body}")
    if tag == "l":
        items = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("- "):
                s = s[2:].strip()
            items.append(s)
        return items
    if tag == "n":
        s = raw.strip()
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            raise ValueError(f"AGFMT n section did not parse as number: {s!r}")
    if tag == "b":
        s = raw.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        raise ValueError(f"AGFMT b section did not parse as bool: {s!r}")
    if tag == "kv":
        out: dict[str, str] = {}
        for ln in lines:
            if ":" not in ln:
                continue
            k, _, v = ln.partition(":")
            out[k.strip()] = v.strip()
        return out
    raise ValueError(f"unknown AGFMT tag {tag!r}")


def decode(text: str) -> dict[str, Any]:
    """Parse AGFMT into a dict. Tolerant of preamble / missing @END / fences."""
    out: dict[str, Any] = {}
    cur_name: str | None = None
    cur_tag: str | None = None
    cur_lines: list[str] = []

    def _flush() -> None:
        if cur_name is None:
            return
        out[cur_name] = _decode_section(cur_tag, cur_lines)

    for line in text.splitlines():
        if _END_RE.match(line):
            _flush()
            cur_name = None
            cur_tag = None
            cur_lines = []
            break
        m = _HEADER_RE.match(line)
        if m:
            _flush()
            cur_name = m.group(1)
            cur_tag = m.group(2)
            cur_lines = []
            continue
        if cur_name is not None:
            cur_lines.append(line)
        # else: preamble before first header — ignored
    _flush()
    return out


# Static format-spec block. Identical across every graph node, so the
# dflash prompt cache can match on this prefix and skip its prefill on
# subsequent nodes that begin with the same text. Keep it byte-stable.
AGFMT_STATIC_FORMAT_BLOCK = (
    "# Output format (AGFMT)\n"
    "Emit each named output as a section:\n"
    "    @<name>:<tag>\n"
    "    <content>\n"
    "Tags: :j JSON  :l list (one item per line)  :t text  "
    ":n number  :b bool  :kv key:value-pairs.\n"
    "Close with `@END`. No prose outside sections.\n"
)


def output_template(spec: list[str]) -> str:
    """Build the system-prompt snippet describing the AGFMT format the
    agent must produce.

    `spec` is a list of "name:tag" strings, e.g. ["facts:j", "themes:l",
    "commentary:t"]. The static format-spec block (`AGFMT_STATIC_FORMAT_BLOCK`)
    is reused verbatim so prompt-cache hits are possible across nodes; only
    the trailing "Required outputs" list varies per call.
    """
    parts = [AGFMT_STATIC_FORMAT_BLOCK, "Required outputs:"]
    for s in spec:
        parts.append(f"  @{s}")
    parts.append("@END")
    return "\n".join(parts)


def output_template_required_only(spec: list[str]) -> str:
    """Just the variable trailing block — used by callers that emit the
    static format block themselves at a deliberate position (e.g. up-front
    so prompt-cache hits across nodes)."""
    parts = ["Required outputs:"]
    for s in spec:
        parts.append(f"  @{s}")
    parts.append("@END")
    return "\n".join(parts)


# Self-test runs when executed directly. Cheap sanity check on encode/decode
# round-trip plus a deliberately ugly input that exercises tolerance paths.
def _selftest() -> None:
    sample = {
        "facts": {"sp500": 7173.91, "vix": 18.45, "themes": ["risk", "oil"]},
        "themes": ["geopolitical_risk", "oil_volatility"],
        "commentary": "Market closed flat.\nVolatility elevated on Hormuz situation.",
        "score": 0.73,
        "is_final": True,
        "kv_data": {"a": "1", "b": "two"},
    }
    enc = encode(sample, types={"score": "n", "is_final": "b", "kv_data": "kv"})
    dec = decode(enc)
    for k in sample:
        assert k in dec, f"missing key {k}"
    assert dec["facts"]["sp500"] == 7173.91
    assert dec["themes"] == ["geopolitical_risk", "oil_volatility"]
    assert dec["commentary"].startswith("Market closed flat")
    assert abs(dec["score"] - 0.73) < 1e-9
    assert dec["is_final"] is True
    assert dec["kv_data"]["a"] == "1"

    # Tolerance: leading prose, fenced JSON, missing @END, mixed list dashes.
    ugly = (
        "Here is your data:\n"
        "@facts:j\n"
        "```json\n"
        '{"a": 1, "b": [2, 3]}\n'
        "```\n"
        "@items:l\n"
        "- one\n"
        "two\n"
        "  - three  \n"
        "\n"
        "@note:t\n"
        "all good"
    )
    d2 = decode(ugly)
    assert d2["facts"] == {"a": 1, "b": [2, 3]}
    assert d2["items"] == ["one", "two", "three"]
    assert d2["note"] == "all good"
    print("agfmt self-test: ok")


if __name__ == "__main__":
    _selftest()

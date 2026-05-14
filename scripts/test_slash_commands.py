"""Standalone test of the chat slash-command routing.

Mocks the HTTP request/response wires so we can exercise
`UIHandler._maybe_handle_slash_command` without binding a TCP socket.
Tests routing, parsing, and that unknown commands fall through to the
model path.
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from unittest.mock import patch

# Make scripts/ importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen_ui  # type: ignore  # noqa: E402


class _DummyHandler:
    """Minimal stand-in for BaseHTTPRequestHandler. Captures SSE writes."""

    def __init__(self) -> None:
        self.responses: list[tuple[int, list[tuple[str, str]]]] = []
        self.wfile = io.BytesIO()
        self.close_connection = False
        self._status: int | None = None
        self._headers: list[tuple[str, str]] = []

    def send_response(self, status: int) -> None:
        self._status = status

    def send_header(self, k: str, v: str) -> None:
        self._headers.append((k, v))

    def end_headers(self) -> None:
        self.responses.append((self._status or 0, list(self._headers)))
        self._status = None
        self._headers = []

    def captured_sse(self) -> list[tuple[str, dict]]:
        """Parse the wfile bytes back into (event, data) tuples."""
        raw = self.wfile.getvalue().decode("utf-8", errors="replace")
        out: list[tuple[str, dict]] = []
        for frame in raw.split("\n\n"):
            frame = frame.strip()
            if not frame:
                continue
            event = "message"
            data_lines: list[str] = []
            for line in frame.splitlines():
                if line.startswith("event: "):
                    event = line[len("event: "):]
                elif line.startswith("data: "):
                    data_lines.append(line[len("data: "):])
            try:
                data = json.loads("\n".join(data_lines)) if data_lines else {}
            except json.JSONDecodeError:
                data = {"_raw": "\n".join(data_lines)}
            out.append((event, data))
        return out


def _invoke(handler: _DummyHandler, cmd_line: str,
             messages: list[dict] | None = None) -> bool:
    """Invoke the slash-command router with a fresh dummy handler."""
    if messages is None:
        messages = [{"role": "user", "content": cmd_line}]
    return qwen_ui.UIHandler._maybe_handle_slash_command(
        handler,  # type: ignore[arg-type]
        cmd_line,
        list(messages),
        session_id="testsession",
        cwd=os.getcwd(),
    )


class SlashCommandTests(unittest.TestCase):
    def test_help_routes(self) -> None:
        h = _DummyHandler()
        # Persistence touches the filesystem; stub it to a no-op.
        with patch.object(qwen_ui, "_persist_session", lambda *_a, **_k: None):
            handled = _invoke(h, "/help")
        self.assertTrue(handled, "/help should be handled server-side")
        events = h.captured_sse()
        # Should see at least one content delta containing 'slash commands'.
        deltas = [d.get("delta", "") for e, d in events if e == "content"]
        joined = "".join(deltas).lower()
        self.assertIn("slash command", joined,
                       f"/help reply should mention slash commands, got: {joined[:200]!r}")
        # Should emit 'started' and 'done'.
        kinds = [e for e, _ in events]
        self.assertIn("started", kinds)
        self.assertIn("done", kinds)

    def test_graphs_routes(self) -> None:
        h = _DummyHandler()
        with patch.object(qwen_ui, "_persist_session", lambda *_a, **_k: None):
            handled = _invoke(h, "/graphs")
        self.assertTrue(handled)
        events = h.captured_sse()
        joined = "".join(d.get("delta", "") for e, d in events if e == "content")
        # Either a table of graphs or 'no graphs defined' is acceptable.
        self.assertTrue(
            "graph" in joined.lower() or "no graphs" in joined.lower(),
            f"/graphs reply should reference graphs, got: {joined[:200]!r}",
        )

    def test_unknown_falls_through(self) -> None:
        h = _DummyHandler()
        # An unrecognized slash command should NOT be handled — the chat
        # handler routes it to the model instead.
        handled = _invoke(h, "/notacommand")
        self.assertFalse(handled, "unknown slash should fall through")

    def test_non_slash_falls_through(self) -> None:
        h = _DummyHandler()
        handled = _invoke(h, "hello world")
        self.assertFalse(handled)

    def test_tools_routes(self) -> None:
        h = _DummyHandler()
        with patch.object(qwen_ui, "_persist_session", lambda *_a, **_k: None):
            handled = _invoke(h, "/tools")
        self.assertTrue(handled)
        events = h.captured_sse()
        deltas = "".join(d.get("delta", "") for e, d in events if e == "content")
        # Should mention scratchpad and graph_compose (new chat-tier tools).
        self.assertIn("scratchpad", deltas.lower())
        self.assertIn("graph_compose", deltas.lower())

    def test_scratchpad_read_routes(self) -> None:
        h = _DummyHandler()
        with patch.object(qwen_ui, "_persist_session", lambda *_a, **_k: None):
            handled = _invoke(h, "/scratchpad read")
        self.assertTrue(handled)
        events = h.captured_sse()
        deltas = "".join(d.get("delta", "") for e, d in events if e == "content")
        # Either "empty" (no scratchpad yet) or some content; either is fine.
        self.assertTrue(len(deltas) > 0, "scratchpad reply should have content")


if __name__ == "__main__":
    unittest.main(verbosity=2)

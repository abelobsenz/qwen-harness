"""Regression test for the float-as-arxiv-id bug.

Observed in chat: model emitted `{"id_or_url": 2401.12345}` (JSON number, not
string) — `_arxiv_id_from_input(s).strip()` raised `AttributeError: 'float'
object has no attribute 'strip'`, and `arxiv_search` with a float `query`
raised `TypeError: quote_from_bytes() expected bytes`. Both surfaces are
guarded now by coercing non-string input at function entry.

This test runs OFFLINE — `_arxiv_id_from_input` is pure string parsing, so
no network. The two callable wrappers (`arxiv_search`, `arxiv_fetch`) are
exercised with mocked `urllib.request.urlopen` so we never hit arxiv.org
in CI.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_tools  # type: ignore  # noqa: E402


class _FakeResponse(io.BytesIO):
    """Stand-in for urlopen() context manager."""
    def __enter__(self): return self
    def __exit__(self, *a): self.close(); return False


class ArxivArgCoercionTests(unittest.TestCase):

    def test_id_parser_handles_float(self) -> None:
        """The bare id parser must not raise on a float arg."""
        self.assertEqual(
            agent_tools._arxiv_id_from_input(2401.12345),
            "2401.12345",
            "float arxiv id should be coerced to its bare-id string form",
        )

    def test_id_parser_handles_int(self) -> None:
        # Less common but still a valid float-like form for very old ids.
        self.assertIsNone(agent_tools._arxiv_id_from_input(12345))

    def test_id_parser_handles_none_and_empty(self) -> None:
        self.assertIsNone(agent_tools._arxiv_id_from_input(None))
        self.assertIsNone(agent_tools._arxiv_id_from_input(""))

    def test_id_parser_still_works_with_strings(self) -> None:
        self.assertEqual(agent_tools._arxiv_id_from_input("2401.12345"),
                          "2401.12345")
        self.assertEqual(
            agent_tools._arxiv_id_from_input(
                "https://arxiv.org/abs/2401.12345"),
            "2401.12345",
        )

    def test_arxiv_fetch_with_float_id_does_not_raise(self) -> None:
        """The function should reach the network layer, not crash at .strip()."""
        atom_response = (
            b"<feed xmlns='http://www.w3.org/2005/Atom'>"
            b"<entry><id>http://arxiv.org/abs/2401.12345</id>"
            b"<title>Test paper</title>"
            b"<summary>An abstract.</summary>"
            b"<author><name>A. Author</name></author>"
            b"</entry></feed>"
        )
        with patch.object(agent_tools.urllib.request, "urlopen",
                           return_value=_FakeResponse(atom_response)):
            result = agent_tools.arxiv_fetch(2401.12345)
        # Just assert no crash and we got SOMETHING (the exact format isn't
        # under test — that's covered by integration smoke).
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertNotIn("AttributeError", result)
        self.assertNotIn("TypeError", result)

    def test_arxiv_search_with_float_query_does_not_raise(self) -> None:
        """quote_plus() on a float used to raise TypeError; coercion fixes it."""
        atom_response = (
            b"<feed xmlns='http://www.w3.org/2005/Atom'>"
            b"<entry><id>http://arxiv.org/abs/2401.12345</id>"
            b"<title>Test paper</title>"
            b"<summary>An abstract.</summary>"
            b"<author><name>A. Author</name></author>"
            b"</entry></feed>"
        )
        with patch.object(agent_tools.urllib.request, "urlopen",
                           return_value=_FakeResponse(atom_response)):
            result = agent_tools.arxiv_search(2401.12345)
        self.assertIsInstance(result, str)
        self.assertNotIn("TypeError", result)
        self.assertNotIn("quote_from_bytes", result)

    def test_arxiv_search_with_none_query(self) -> None:
        """A None query coerces to '' and the API returns no entries —
        we should get '(no arxiv results)' or a clean error string."""
        atom_response = b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
        with patch.object(agent_tools.urllib.request, "urlopen",
                           return_value=_FakeResponse(atom_response)):
            result = agent_tools.arxiv_search(None)
        self.assertIsInstance(result, str)
        self.assertNotIn("TypeError", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

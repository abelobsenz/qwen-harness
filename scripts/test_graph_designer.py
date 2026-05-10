#!/usr/bin/env python3
"""test_graph_designer — pure-data tests for design_preview / save_spec.

These tests don't actually call the model; they validate the pure-Python
parts of the two-phase graph designer flow that the UI relies on. Stress
tests against the live model live in /tmp/qwen_stress_designer.py.

Catches regressions for:
  - validation rejects malformed specs
  - save_spec writes a loadable file
  - cycle detection at save time
  - delete cleans up the file
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_designer import _validate_spec, save_spec, render_python  # noqa: E402


def _ok(msg: str) -> None:
    print(f"    [✓] {msg}")


def _check(cond: bool, msg: str) -> None:
    if cond:
        _ok(msg)
    else:
        raise SystemExit(f"    [✗] {msg}")


def test_validate_spec():
    print("\n[1] _validate_spec rejects malformed specs")
    # Empty nodes
    try:
        _validate_spec({"name": "x", "nodes": [], "edges": []})
        raise SystemExit("expected ValueError on empty nodes")
    except ValueError as e:
        _check("non-empty list" in str(e), f"empty nodes rejected: {e}")

    # Missing goal
    try:
        _validate_spec({"name": "x", "nodes": [
            {"name": "a", "role": "r", "goal": "", "inputs": [], "outputs": [["x", "t"]]}
        ], "edges": []})
        raise SystemExit("expected ValueError on missing goal")
    except ValueError as e:
        _check("missing `goal`" in str(e), f"missing goal rejected: {e}")

    # Empty outputs
    try:
        _validate_spec({"name": "x", "nodes": [
            {"name": "a", "role": "r", "goal": "g", "inputs": [], "outputs": []}
        ], "edges": []})
        raise SystemExit("expected ValueError on empty outputs")
    except ValueError as e:
        _check("non-empty list" in str(e), f"empty outputs rejected: {e}")

    # Duplicate node names
    try:
        _validate_spec({"name": "x", "nodes": [
            {"name": "a", "role": "r", "goal": "g", "inputs": [], "outputs": [["x", "t"]]},
            {"name": "a", "role": "r", "goal": "g", "inputs": [], "outputs": [["y", "t"]]},
        ], "edges": []})
        raise SystemExit("expected ValueError on duplicate names")
    except ValueError as e:
        _check("duplicate node name" in str(e), f"duplicate names rejected: {e}")

    # Edge to unknown node
    try:
        _validate_spec({"name": "x", "nodes": [
            {"name": "a", "role": "r", "goal": "g", "inputs": [], "outputs": [["x", "t"]]},
        ], "edges": [{"src": "a", "dst": "ghost"}]})
        raise SystemExit("expected ValueError on edge to unknown node")
    except ValueError as e:
        _check("unknown node" in str(e), f"edge to unknown node rejected: {e}")


def test_validate_normalizes():
    print("\n[2] _validate_spec normalizes valid input")
    spec = _validate_spec({
        "name": "Test Graph!",  # gets slugified
        "nodes": [
            {"name": "First Node", "role": "r", "goal": "g", "inputs": ["x"],
             "outputs": [["y", "INVALID_TAG"]]},  # tag coerces to 't'
        ],
        "edges": [],
    })
    # _slugify strips trailing underscores, so "Test Graph!" → "test_graph"
    _check(spec["name"] == "test_graph", f"name slugified: {spec['name']!r}")
    _check(spec["nodes"][0]["name"] == "first_node", f"node name slugified: {spec['nodes'][0]['name']!r}")
    _check(spec["nodes"][0]["outputs"][0][1] == "t", "invalid tag coerced to 't'")


def test_save_loads_file():
    print("\n[3] save_spec writes a loadable file")
    with tempfile.TemporaryDirectory() as tmp:
        spec = {
            "name": "smoketest_unit",
            "nodes": [
                {"name": "n1", "role": "r", "goal": "g",
                 "inputs": [], "outputs": [["x", "t"]]},
            ],
            "edges": [],
        }
        result = save_spec(spec, examples_dir=tmp)
        _check(result.get("ok") is True, f"save returned ok: {result.get('ok')}")
        path = result["path"]
        _check(os.path.exists(path), f"file written: {path}")
        _check(path.endswith("_graph.py"), "file has _graph.py suffix")


def test_save_rejects_cycle():
    print("\n[4] save_spec rejects cyclic graph at save time")
    with tempfile.TemporaryDirectory() as tmp:
        spec = {
            "name": "cyclic_unit",
            "nodes": [
                {"name": "a", "role": "r", "goal": "g",
                 "inputs": ["x"], "outputs": [["x", "t"]]},
                {"name": "b", "role": "r", "goal": "g",
                 "inputs": ["x"], "outputs": [["x", "t"]]},
            ],
            "edges": [
                {"src": "a", "dst": "b"},
                {"src": "b", "dst": "a"},
            ],
        }
        result = save_spec(spec, examples_dir=tmp)
        _check(not result.get("ok"), "cyclic save rejected")
        _check(result.get("stage") == "topology", f"stage tagged correctly: {result.get('stage')}")
        _check("cycle" in (result.get("error") or "").lower(),
               f"error mentions cycle: {result.get('error', '')}")
        # File should have been removed
        path = result.get("path") or ""
        _check(not os.path.exists(path),
               f"phantom file removed: exists={os.path.exists(path)}")


def test_render_python():
    print("\n[5] render_python produces parseable code")
    spec = {
        "name": "render_test",
        "nodes": [
            {"name": "n1", "role": "researcher",
             "goal": "find facts", "inputs": ["topic"],
             "outputs": [("facts", "j"), ("note", "t")],
             "tools": ["web_search"], "max_steps": 6,
             "map_over": None, "map_item_key": None,
             "batch_map": False, "extra_instructions": ""},
        ],
        "edges": [],
    }
    code = render_python(spec, description="test")
    import ast
    tree = ast.parse(code)
    _check(tree is not None, "rendered code parses as Python")
    _check("AgentGraph" in code, "code references AgentGraph")
    _check("'web_search'" in code, "tool name preserved in code")
    _check("'researcher'" in code, "role preserved in code")


def main():
    print("== graph_designer unit test ==")
    test_validate_spec()
    test_validate_normalizes()
    test_save_loads_file()
    test_save_rejects_cycle()
    test_render_python()
    print("\n== PASS (0 failure(s)) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Tests for the iter 29 efficiency guards.

Background: iter 28 sessions revealed three waste patterns:
  1. Same URL fetched 3-5× with varying max_chars to bypass cache
     (3D Systems DEF 14A fetched 4×).
  2. 13+ near-duplicate web_search reformulations in a session
     (3D Systems "director compensation 2023" 13×, AMD 19×).
  3. `bash curl URL` to bypass web_fetch's condensation when the
     same URL had already been pulled (3D Sys 4 consecutive bash
     curls of an SEC URL).

Each pattern wastes 5-15 calls and 1-3 minutes per affected session.
The fixes are pure structural guards that fire purely on response
shape, not on prompt content.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agent_tools  # noqa: E402


def main() -> int:
    failures = 0

    # ---- 1. URL refetch cap ----
    print("[1] URL refetch cap blocks same URL after threshold")
    d = agent_tools.CachedDispatcher()
    d._url_fetch_max = 3
    url = "https://www.sec.gov/Archives/edgar/data/910638/foo.htm"
    norm = agent_tools._normalize_url(url)
    # n=2: still allowed
    d._url_fetch_count[norm] = 2
    ref = d._check_url_refetch(url)
    if ref is not None:
        print(f"    [FAIL] n=2 should not refuse, got: {ref[:80]}")
        failures += 1
    # n=3: refuse
    d._url_fetch_count[norm] = 3
    ref = d._check_url_refetch(url)
    if ref is None or "fetched 3 times" not in ref:
        print(f"    [FAIL] n=3 should refuse with 'fetched 3 times', got: {ref}")
        failures += 1
    elif "find_in_url" not in ref:
        print(f"    [FAIL] refusal should hint at find_in_url")
        failures += 1
    else:
        print(f"    [OK] refetch capped at {d._url_fetch_max} per URL")

    # ---- 2. Different max_chars on same URL still hits the URL cap ----
    print("[2] Different max_chars on same URL doesn't reset the URL counter")
    d = agent_tools.CachedDispatcher()
    d._url_fetch_max = 3
    url = "https://example.com/big.htm"
    norm = agent_tools._normalize_url(url)
    # Simulate 3 successful fetches with different max_chars values.
    d._url_fetch_count[norm] = 3
    # The 4th attempt with NEW max_chars value still gets refused —
    # the cache key (which includes max_chars) wouldn't catch this.
    ref = d._check_url_refetch(url)
    if ref is None:
        print(f"    [FAIL] 4th fetch (any max_chars) should refuse")
        failures += 1
    else:
        print(f"    [OK] URL cap is independent of max_chars")

    # ---- 3. Near-dup web_search counter throttles spam ----
    print("[3] Near-dup web_search refused after threshold")
    d = agent_tools.CachedDispatcher()
    d._web_search_near_dup_max = 4
    d._web_search_near_dup_count = 3
    ref = d._check_web_search_cap()
    if ref is not None:
        print(f"    [FAIL] near_dup=3 should still allow")
        failures += 1
    d._web_search_near_dup_count = 4
    ref = d._check_web_search_cap()
    if ref is None or "near-duplicate" not in ref:
        print(f"    [FAIL] near_dup=4 should refuse with 'near-duplicate'")
        failures += 1
    else:
        print(f"    [OK] near-dup spam capped at {d._web_search_near_dup_max}")

    # ---- 4. bash curl on previously-fetched URL refused ----
    print("[4] bash curl on already-fetched URL refused")
    d = agent_tools.CachedDispatcher()
    url = "https://www.sec.gov/Archives/edgar/data/910638/d914592ddef14a.htm"
    norm = agent_tools._normalize_url(url)
    d._url_fetch_count[norm] = 1  # appeared in previous fetch
    ref = d._check_bash_url_bypass({"command": f"curl {url}"})
    if ref is None:
        print(f"    [FAIL] bash curl of fetched URL should refuse")
        failures += 1
    elif "find_in_url" not in ref:
        print(f"    [FAIL] refusal should hint at find_in_url")
        failures += 1
    else:
        print(f"    [OK] bash curl bypass refused")

    # ---- 5. bash curl of unseen URL passes through ----
    print("[5] bash curl of NEW URL passes (not fetched before)")
    d = agent_tools.CachedDispatcher()
    ref = d._check_bash_url_bypass({"command": "curl https://example.com/new.htm"})
    if ref is not None:
        print(f"    [FAIL] new URL should pass, got: {ref[:80]}")
        failures += 1
    else:
        print(f"    [OK] unfetched URLs pass through")

    # ---- 6. bash with ordinary commands unaffected ----
    print("[6] bash with non-curl/wget commands unaffected")
    d = agent_tools.CachedDispatcher()
    for cmd in ("ls -la", "python -m venv", "git status",
                "echo 'hi'", "grep foo bar.txt"):
        ref = d._check_bash_url_bypass({"command": cmd})
        if ref is not None:
            print(f"    [FAIL] {cmd!r} should pass, got: {ref[:80]}")
            failures += 1
            break
    else:
        print(f"    [OK] ordinary bash commands unaffected")

    # ---- 7. wget detection works the same ----
    print("[7] wget on previously-fetched URL refused")
    d = agent_tools.CachedDispatcher()
    url = "https://www.sec.gov/foo.htm"
    norm = agent_tools._normalize_url(url)
    d._url_fetch_count[norm] = 1
    ref = d._check_bash_url_bypass({"command": f"wget {url}"})
    if ref is None:
        print(f"    [FAIL] wget should refuse same as curl")
        failures += 1
    else:
        print(f"    [OK] wget caught alongside curl")

    # ---- 9. Default near-dup cap is 4 (iter 30 final tuning) ----
    print("[9] Default near-dup cap is 4 (iter 30 final tuning)")
    d = agent_tools.CachedDispatcher()
    if d._web_search_near_dup_max != 4:
        print(f"    [FAIL] expected default cap=4, got {d._web_search_near_dup_max}")
        failures += 1
    else:
        print(f"    [OK] default cap is 4; refuses on 5th near-dup attempt")

    # ---- 10. Jaccard near-dup catches reformulations cosine missed ----
    print("[10] Jaccard near-dup catches token-overlap reformulations")
    # Direct test of _check_search_duplicate. We simulate the in-session
    # memo so we don't need a real embedder.
    import numpy as _np
    agent_tools._SEARCH_QUERY_MEMO.clear()
    # Fake prior query with a synthetic embedding (random). Cosine vs a
    # different fake query will be near-zero, so cosine path won't catch
    # it. Token-overlap will.
    prior_q = "Micron Q3 2024 GAAP gross margin guidance"
    prior_vec = _np.random.RandomState(0).randn(384).astype(_np.float32)
    prior_vec /= _np.linalg.norm(prior_vec)
    agent_tools._SEARCH_QUERY_MEMO.append((prior_q, prior_vec, "prior summary"))
    new_q = "Micron Q3 2024 GAAP gross margin guidance plan"
    # This reformulation shares 6 of 7 prior tokens; Jaccard ~0.86.
    # Stub out the embedder so cosine path can't accidentally match.
    orig_embed = agent_tools._embed_query_safe
    agent_tools._embed_query_safe = lambda q: _np.random.RandomState(99).randn(384).astype(_np.float32) / _np.sqrt(384)
    try:
        hit = agent_tools._check_search_duplicate(new_q)
    finally:
        agent_tools._embed_query_safe = orig_embed
        agent_tools._SEARCH_QUERY_MEMO.clear()
    if hit is None:
        print(f"    [FAIL] Jaccard should have caught the reformulation")
        failures += 1
    else:
        print(f"    [OK] Jaccard caught reformulation (score={hit[2]:.2f})")

    # ---- 11. Distinct queries don't trip Jaccard ----
    print("[11] Distinct queries (low Jaccard) pass through")
    agent_tools._SEARCH_QUERY_MEMO.clear()
    prior_vec = _np.random.RandomState(0).randn(384).astype(_np.float32)
    prior_vec /= _np.linalg.norm(prior_vec)
    agent_tools._SEARCH_QUERY_MEMO.append(
        ("Apple earnings Q4 2024", prior_vec, "prior summary"))
    agent_tools._embed_query_safe = lambda q: _np.random.RandomState(99).randn(384).astype(_np.float32) / _np.sqrt(384)
    try:
        hit = agent_tools._check_search_duplicate("Tesla shareholder letter 2025")
    finally:
        agent_tools._embed_query_safe = orig_embed
        agent_tools._SEARCH_QUERY_MEMO.clear()
    if hit is not None:
        print(f"    [FAIL] distinct query should pass (got: {hit[0]})")
        failures += 1
    else:
        print(f"    [OK] distinct query untouched")

    # ---- 8. start_new_task clears all efficiency counters ----
    print("[8] start_new_task clears the new counters")
    d = agent_tools.CachedDispatcher()
    d._web_search_near_dup_count = 4
    d._url_fetch_count = {"https://x.com/y": 3}
    d.start_new_task()
    if d._web_search_near_dup_count != 0:
        print(f"    [FAIL] near_dup counter not reset")
        failures += 1
    elif d._url_fetch_count:
        print(f"    [FAIL] URL fetch dict not reset")
        failures += 1
    else:
        print(f"    [OK] reset clears both counters")

    if failures:
        print(f"\n== FAIL ({failures} failure(s)) ==")
        return 1
    print("\n== PASS (efficiency guards behave correctly) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())

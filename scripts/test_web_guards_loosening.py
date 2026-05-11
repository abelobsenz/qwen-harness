#!/usr/bin/env python3
"""Pin the iter-37 web-guard loosening.

Three changes are tested:

  1. `_DEFAULT_PREAPPROVED_HOSTS` now includes major business news +
     finance data aggregator domains (cnbc, reuters, bloomberg, ft,
     wsj, macrotrends, stockanalysis, simplywall, nasdaq, nyse,
     investor.gov, treasury.gov, imf, worldbank, oecd, morningstar,
     tradingview, etc.). Each must allow a fresh URL through
     `_check_url_seen` without a prior search hit.

  2. `_IR_SUBDOMAIN_PREFIXES` (ir.*, investor.*, investors.*, corp.*,
     corporate.*, investorrelations.*, etc.) now allow any company's
     IR subdomain without a prior search hit.

  3. The near-dup search cap was bumped 4 → 8; per-URL fetch cap 3 → 5.
     Each cap's emit message still uses REFUSED_PREFIX (the upstream
     contract from R3 must survive).

The test does NOT load the model — it exercises `_check_url_seen` and
the cap-check methods directly. Runs in <50 ms.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    # Force-enable URL guard so we test the loose path explicitly.
    os.environ["QWEN_URL_GUARD_ENABLE"] = "1"
    os.environ.pop("QWEN_WEB_PREAPPROVED_HOSTS", None)

    # Match the new defaults from config/qwen.conf
    os.environ.setdefault("QWEN_WEB_SEARCH_NEAR_DUP_MAX", "8")
    os.environ.setdefault("QWEN_URL_FETCH_MAX", "5")
    os.environ.setdefault("QWEN_DUP_CALL_MAX", "5")

    from agent_tools import (
        CachedDispatcher,
        REFUSED_PREFIX,
        _DEFAULT_PREAPPROVED_HOSTS,
        _IR_SUBDOMAIN_PREFIXES,
        _preapproved_hosts,
    )

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    print("== web-guard loosening (iter 37) ==\n")
    t0 = time.perf_counter()

    d = CachedDispatcher()

    # --- 1. The new finance hosts are reachable from a cold cache -----
    NEW_FINANCE_HOSTS = [
        ("https://www.cnbc.com/2024/05/01/some-article.html", "cnbc"),
        ("https://www.reuters.com/business/some-piece", "reuters"),
        ("https://www.bloomberg.com/news/articles/2024-05-01", "bloomberg"),
        ("https://www.ft.com/content/abc123", "ft"),
        ("https://www.wsj.com/articles/abc-def", "wsj"),
        ("https://www.barrons.com/articles/abc", "barrons"),
        ("https://fortune.com/2024/05/01/article", "fortune"),
        ("https://www.forbes.com/sites/abc/2024/05/01", "forbes"),
        ("https://www.macrotrends.net/stocks/charts/NFLX/netflix/revenue", "macrotrends"),
        ("https://stockanalysis.com/stocks/nflx/financials", "stockanalysis"),
        ("https://www.nasdaq.com/market-activity/stocks/nflx", "nasdaq"),
        ("https://investor.gov/some-page", "investor.gov"),
        ("https://www.treasury.gov/data", "treasury.gov"),
        ("https://www.imf.org/external/datamapper", "imf"),
        ("https://www.morningstar.com/stocks/xnas/nflx/quote", "morningstar"),
    ]
    for url, label in NEW_FINANCE_HOSTS:
        result = d._check_url_seen(url)
        check(f"finance host allowed: {label}", result is None,
              detail=f"got {result[:60] if result else None}")

    # --- 2. IR-subdomain prefix matching ------------------------------
    IR_HOSTS = [
        "https://ir.netflix.com/financial-info/quarterly-results",
        "https://investor.apple.com/quarterly-earnings",
        "https://investors.tjx.com/news-and-events/press-releases",
        "https://corp.example.com/about",
        "https://corporate.aapl.com/governance",
        "https://investorrelations.example.com/financials",
        "https://investor-relations.example.com/news",
    ]
    for url in IR_HOSTS:
        result = d._check_url_seen(url)
        check(f"IR subdomain allowed: {url.split('//')[1].split('/')[0]}",
              result is None,
              detail=f"got {result[:60] if result else None}")

    # --- 3. Negative case — fabricated slug at NON-preapproved host ---
    # The hallucination case that originally motivated the guard MUST
    # still be caught. Generic news hosts like news.example.com aren't
    # in the allowlist and don't start with an IR prefix.
    HALLUCINATIONS = [
        "https://news.example.com/some-fabricated-slug-12345",
        "https://shop.walmart.com/ip/Fake-Product/9999999999",
        "https://random-blog.example.org/article",
    ]
    for url in HALLUCINATIONS:
        result = d._check_url_seen(url)
        check(f"hallucinated URL still refused: {url.split('//')[1].split('/')[0]}",
              result is not None and result.startswith(REFUSED_PREFIX))

    # --- 4. Near-dup cap raised to 8 ----------------------------------
    d2 = CachedDispatcher()
    # Simulate 7 near-dup hits — must NOT yet refuse.
    d2._web_search_near_dup_count = 7
    result = d2._check_web_search_cap()
    check("7 near-dups: cap NOT yet hit (was 4 before, now 8)",
          result is None, detail=f"got {result[:60] if result else None}")

    # At 8 it SHOULD refuse.
    d2._web_search_near_dup_count = 8
    result = d2._check_web_search_cap()
    check("8 near-dups: cap hit",
          result is not None and result.startswith(REFUSED_PREFIX),
          detail=f"got {result[:60] if result else None}")

    # --- 5. Per-URL fetch cap raised to 5 -----------------------------
    # IMPORTANT: _check_url_refetch keys by _normalize_url(url) which
    # preserves path case (only host is lowercased). Use the actual
    # normalized form so the count lookup hits.
    from agent_tools import _normalize_url
    d3 = CachedDispatcher()
    url = "https://www.sec.gov/Archives/edgar/data/XYZ/000123/abc.htm"
    norm = _normalize_url(url)
    # 4 prior fetches: must NOT refuse (5 is the cap, so 4 is the last allowed).
    d3._url_fetch_count[norm] = 4
    result = d3._check_url_refetch(url)
    check("4 same-URL fetches: cap NOT yet hit (was 3 before, now 5)",
          result is None, detail=f"got {result[:60] if result else None}")
    d3._url_fetch_count[norm] = 5
    result = d3._check_url_refetch(url)
    check("5 same-URL fetches: cap hit",
          result is not None and result.startswith(REFUSED_PREFIX),
          detail=f"got {result[:60] if result else None}")

    # --- 5b. Duplicate-call cap raised to 5 (was hardcoded 3 before) -
    # This is the SECOND-to-last backstop after the cache marker. It
    # fires when the model re-issues an EXACT cached call too many
    # times. Multi-period finance research can legitimately retry the
    # same query after a related variant got near-dup-blocked.
    from agent_tools import _arg_key  # noqa: WPS433
    d4 = CachedDispatcher()
    # Build the exact (fn, arg_key) tuple the dispatcher uses.
    args = {"query": "k1"}
    cache_key = ("web_search", _arg_key(args))
    d4.web_cache[cache_key] = "cached result body"
    # At count=4: next dispatch increments to 4, then 4 < 5 → cached marker.
    d4.web_call_counts[cache_key] = 3
    result, was_cached = d4.dispatch("web_search", args)
    check("4 duplicate web calls: cap NOT yet hit (was 3 before, now 5)",
          not result.startswith(REFUSED_PREFIX) and was_cached,
          detail=f"got {result[:60]}")
    # At count=4 prior, next dispatch increments to 5, then 5 >= 5 → refusal.
    d4.web_call_counts[cache_key] = 4
    result, _ = d4.dispatch("web_search", args)
    check("5 duplicate web calls: cap hit",
          result.startswith(REFUSED_PREFIX),
          detail=f"got {result[:80]}")

    # --- 6. Sanity: existing allowlist still intact -------------------
    LEGACY_HOSTS = [
        "https://www.sec.gov/cgi-bin/browse-edgar",
        "https://en.wikipedia.org/wiki/Netflix",
        "https://github.com/owner/repo",
        "https://docs.python.org/3/library/json.html",
        "https://arxiv.org/abs/2505.09388",
        "https://www.businesswire.com/news/home/abc",
        "https://finance.yahoo.com/quote/NFLX",
    ]
    for url in LEGACY_HOSTS:
        result = d._check_url_seen(url)
        check(f"legacy host still allowed: {url.split('//')[1].split('/')[0]}",
              result is None)

    # --- 7. Allowlist size sanity-bound (catches accidental deletion) -
    n_hosts = len(_DEFAULT_PREAPPROVED_HOSTS)
    check(f"preapproved host list has ≥60 entries (was ~30 pre-loosen)",
          n_hosts >= 60, detail=f"len={n_hosts}")

    # --- 8. IR prefix tuple is non-empty -----------------------------
    check("IR subdomain prefix tuple is non-empty",
          isinstance(_IR_SUBDOMAIN_PREFIXES, tuple)
          and len(_IR_SUBDOMAIN_PREFIXES) >= 5)

    # --- 9. Performance: each check stays in the µs range ------------
    # The check does a normalize_url() (urlparse + lowercase), set
    # membership, and prefix scan. ~3-10 µs is normal; we want to be
    # sure no quadratic regression slips in (e.g. iterating the
    # allowlist as a list).
    N = 10_000
    t_chk = time.perf_counter()
    for _ in range(N):
        d._check_url_seen("https://www.cnbc.com/article")
    per_call_ns = (time.perf_counter() - t_chk) / N * 1e9
    check("URL-seen check < 15 µs/call (target: low single-digit µs)",
          per_call_ns < 15_000, f"{per_call_ns:.0f} ns/call over {N:,} calls")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  elapsed: {elapsed_ms:.1f} ms")
    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

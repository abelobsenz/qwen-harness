#!/usr/bin/env python3
"""dflash_serve_patched — drop-in replacement for `venv/bin/dflash-serve`
that installs our monkey-patches before launching the real server.

Currently installs:
  - apc_patch.install() — fixes multi-turn APC for hybrid Qwen3.6 caches
    by also saving a post-prefill (prompt-only) snapshot.

Future patches go through the same hook so the daemon launcher only
needs to know about this one file.
"""
from __future__ import annotations
import os
import sys

# Make scripts/ importable so we can grab apc_patch.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import apc_patch  # noqa: E402

apc_patch.install()

# Hand off to the real dflash-serve entry point. argv is preserved so
# all CLI flags (--port, --model, --draft, etc.) flow through unchanged.
from dflash_mlx.serve import main  # noqa: E402

sys.exit(main())

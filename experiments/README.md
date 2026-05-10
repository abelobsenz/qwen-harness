# experiments/

Code that was prototyped but is **not wired into the running stack**.

Moved here from `scripts/` after audit issue H confirmed these files
were not imported by any production code path. Kept (rather than
deleted) because each one represents real exploration that may be
re-activated later. Until then they don't belong in `scripts/` where
every refactor and code review has to consider them.

| File | Status | Re-activation path |
|---|---|---|
| `paged_attn.py` (590 lines) | Storage + indexing complete; fused Metal kernel never written. Only consumer was `paged_integration_probe.py` (also moved here). | Write the fused kernel; benchmark vs flat-KV; if it wins on long contexts, wire into the inference path. |
| `paged_integration_probe.py` | Probe-only consumer of `paged_attn`. | Useful for the activation work above; otherwise inert. |
| `json_grammar.py` (629 lines) | Tool-call grammar masking, intended to enforce well-formed JSON during decode. | Set `DFLASH_JSON_GRAMMAR=1` AND wire `json_grammar_runtime.maybe_install_grammar_hook()` into `dflash_serve_patched.py`. Currently neither is done. |
| `json_grammar_runtime.py` (230 lines) | The runtime bridge for the above; nothing calls `maybe_install_grammar_hook()` today. | Same as above. |
| `router.py` (139 lines) | Two-model router scaffold (35B for hard turns, 8B for routine). Explicit "SCAFFOLD — does not run today" in its own docstring. | Stand up a second `dflash-serve` on port 8003; teach `agent.py` to dispatch turns based on a complexity heuristic. |

If you re-activate any of these, move the file (and any tests) BACK
into `scripts/`. The split is "is this on a code path that runs in
production today, yes or no." Unused-but-might-be-someday-useful is
the criterion for staying here.

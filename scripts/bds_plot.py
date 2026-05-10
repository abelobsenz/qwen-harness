"""Read eval_data/bds_qwen36/trials.jsonl, compute Span90 / Threshold50,
and write the forward-vs-backward plot for Qwen3.6.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "eval_data" / "bds_qwen36"
TRIALS = OUT_DIR / "trials.jsonl"
SUMMARY = OUT_DIR / "summary.csv"
PLOT = OUT_DIR / "qwen36_bds.png"


def main():
    by_n: dict[int, dict[str, list[dict]]] = {}
    with TRIALS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_n.setdefault(r["n"], {"forward": [], "backward": []})[r["direction"]].append(r)

    ns = sorted(by_n)
    print(f"Loaded {sum(len(v['forward']) + len(v['backward']) for v in by_n.values())} trials across N = {ns}")

    fwd_acc: dict[int, float] = {}
    bwd_acc: dict[int, float] = {}
    fwd_counts: dict[int, tuple[int, int]] = {}
    bwd_counts: dict[int, tuple[int, int]] = {}

    for n in ns:
        # Treat "incomplete" trials (max_tokens hit before answer) as
        # excluded from the denominator — they're a budget signal, not a
        # capacity signal. Trials without the field default to "not
        # incomplete" (older entries from the original run).
        f_trials = [t for t in by_n[n]["forward"] if not t.get("incomplete")]
        b_trials = [t for t in by_n[n]["backward"] if not t.get("incomplete")]
        f_correct = sum(1 for t in f_trials if t["correct"])
        b_correct = sum(1 for t in b_trials if t["correct"])
        fwd_acc[n] = f_correct / len(f_trials) if f_trials else 0.0
        bwd_acc[n] = b_correct / len(b_trials) if b_trials else 0.0
        fwd_counts[n] = (f_correct, len(f_trials))
        bwd_counts[n] = (b_correct, len(b_trials))
        print(f"  N={n:>3}: forward {f_correct}/{len(f_trials)} = {fwd_acc[n]:.2f}   "
              f"backward {b_correct}/{len(b_trials)} = {bwd_acc[n]:.2f}")

    # Span90 / Threshold50 (paper's definitions)
    span90 = max((n for n in ns if bwd_acc[n] >= 0.9), default=None)
    thr50 = max((n for n in ns if bwd_acc[n] >= 0.5), default=None)
    print()
    print(f"Span90 (largest N with backward >= 0.9):     {span90}")
    print(f"Threshold50 (largest N with backward >= 0.5): {thr50}")

    with SUMMARY.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "direction", "n_trials", "n_correct", "accuracy"])
        for n in ns:
            for d, counts in (("forward", fwd_counts[n]), ("backward", bwd_counts[n])):
                c, t = counts
                w.writerow([n, d, t, c, c / t if t else 0])

    # Plot — match the per-panel style of Figure 1 / Figure 2 in
    # Diak et al. (CogSci 2026) so it's visually comparable.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fwd_y = [fwd_acc[n] for n in ns]
    bwd_y = [bwd_acc[n] for n in ns]

    # Paper's panels are square-ish (~3.5×3"), x:0-500, y:0-1.04, with
    # Span/N50 box in the top-right and the legend inset on the left.
    fig, ax = plt.subplots(figsize=(3.6, 3.0))

    ax.plot(ns, fwd_y, color="#1f78b4", linewidth=1.4,
            marker="o", markersize=4, label="Forward")
    ax.plot(ns, bwd_y, color="#e31a1c", linewidth=1.4,
            marker="o", markersize=4, label="Backward")

    ax.set_xlim(-15, 515)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Set Size (N)", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("qwen3.6-35b-a3b", fontsize=10, fontweight="bold", pad=4)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Span90 / N50 box in the upper-right (paper's anchor).
    s90_str = str(span90) if span90 is not None else f"<{min(ns)}"
    t50_str = str(thr50) if thr50 is not None else f"<{min(ns)}"
    ax.text(
        0.97, 0.97,
        f"Span$_{{90}}$={s90_str}\nN$_{{50}}$={t50_str}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="0.5",
                  linewidth=0.6, alpha=0.95, boxstyle="round,pad=0.3"),
    )

    # Legend: small, inside the plot, lower-left like the paper.
    leg = ax.legend(loc="lower left", fontsize=8, frameon=True,
                    handlelength=1.4, handletextpad=0.5,
                    borderpad=0.3, borderaxespad=0.3)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.9)

    # Outer spines: keep visible like the paper, slightly thinner.
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.3)
    fig.savefig(PLOT, dpi=180)
    print(f"\nplot:    {PLOT}")
    print(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()

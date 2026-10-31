"""Figure: violation rate vs billed reminder tokens per policy (E3).

Usage: python scripts/plot_e3.py results/analysis/e3_qwen7b.json --out docs/results/e3_tradeoff.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"B0_no_reinforcement": "B0 none", "B1_every_turn": "B1 every turn",
          "B2_every_5": "B2 every 5", "B3_every_10": "B3 every 10",
          "B4_summary_10": "B4 summary/10", "B5_rule_threshold": "B5 rule",
          "P1_predictive": "P1 predictive"}
# Label offsets (points) chosen so neighbouring points' labels do not overlap.
OFFSETS = {"B3_every_10": (-82, -3), "B4_summary_10": (8, 6), "B5_rule_threshold": (-62, 6),
           "P1_predictive": (8, -14)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = json.loads(args.analysis.read_text(encoding="utf-8"))["per_policy"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    for name, v in r.items():
        x, y = v["billed_reminder_tokens_per_plan"], v["violation_rate_plan_mean"]
        lo, hi = v["violation_rate_ci95"]
        p1 = name == "P1_predictive"
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", capsize=3,
                    color="#c2410c" if p1 else "#334155", markersize=8 if p1 else 6)
        ax.annotate(LABELS.get(name, name), (x, y), textcoords="offset points",
                    xytext=OFFSETS.get(name, (8, 4)), fontsize=9,
                    fontweight="bold" if p1 else "normal")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Billed reminder tokens per conversation (log scale)")
    ax.set_ylabel("Violation rate per conversation (mean, 95% CI)")
    ax.set_title("E3, qwen2.5:7b, 100 matched conversations")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

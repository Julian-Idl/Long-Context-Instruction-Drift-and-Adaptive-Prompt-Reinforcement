"""E3 analysis: per-policy compliance and overhead, matched comparisons.

Pre-registered primary comparisons (configs/experiments/e3_reinforcement_kaggle.yaml):
  1. P1 vs B0: per-instruction violation rate.
  2. P1 vs B1: violation rate and billed reminder tokens.
Unit of analysis: the conversation plan (all policies ran on the same plans), so
comparisons are paired per plan. Tests: Wilcoxon signed-rank on per-plan differences,
Holm-corrected over the primary family; 95% CIs from a plan-level bootstrap.
Everything else printed is exploratory.

Usage: python scripts/analyze_e3.py RUN_DIR [RUN_DIR ...] --out results/analysis/e3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from instruction_drift.storage import STREAMS

B0, B1, P1 = "B0_no_reinforcement", "B1_every_turn", "P1_predictive"
PRIMARY = [("P1_vs_B0_violation", P1, B0, "viol_rate"),
           ("P1_vs_B1_violation", P1, B1, "viol_rate"),
           ("P1_vs_B1_billed_tokens", P1, B1, "billed")]


def _read(run: Path, stream: str) -> list[dict]:
    with open(run / STREAMS[stream], encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    comp, prompts, plans = [], [], {}
    for run in runs:
        comp += [c for c in _read(run, "compliance")
                 if c["applicable"] and c["compliance"] is not None]
        prompts += _read(run, "prompts")
        plans |= {p["conversation_id"]: p for p in _read(run, "plans")}
    c = pd.DataFrame(comp)
    c["violation"] = 1 - c["compliance"]
    p = pd.DataFrame(prompts)
    return c, p, plans


def per_plan(c: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    v = c.groupby(["policy_name", "conversation_id"]).agg(
        checks=("violation", "size"), violations=("violation", "sum"))
    t = p.groupby(["policy_name", "conversation_id"]).agg(
        billed=("reinforcement_token_count", "sum"), prompt_tokens=("final_token_count", "sum"))
    df = v.join(t).reset_index()
    df["viol_rate"] = df.violations / df.checks
    return df


def boot_ci(values: np.ndarray, rng, n: int = 5000) -> list[float]:
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    c, p, plans = load(args.runs)
    pp = per_plan(c, p)
    pp["conflict"] = pp.conversation_id.map(lambda k: plans[k]["spec"]["conflict_intensity"])
    policies = sorted(pp.policy_name.unique())
    n_plans = pp.groupby("policy_name").size()
    assert n_plans.nunique() == 1, f"unmatched plans per policy: {n_plans.to_dict()}"

    summary = {}
    for pol in policies:
        g = pp[pp.policy_name == pol]
        summary[pol] = {
            "plans": len(g), "checks": int(g.checks.sum()), "violations": int(g.violations.sum()),
            "violation_rate_pooled": float(g.violations.sum() / g.checks.sum()),
            "violation_rate_plan_mean": float(g.viol_rate.mean()),
            "violation_rate_ci95": boot_ci(g.viol_rate.to_numpy(), rng),
            "billed_reminder_tokens_per_plan": float(g.billed.mean()),
            "billed_share_of_prompt_tokens": float(g.billed.sum() / g.prompt_tokens.sum()),
            "by_conflict": {str(k): float(s.violations.sum() / s.checks.sum())
                            for k, s in g.groupby("conflict")},
        }

    wide = {m: pp.pivot(index="conversation_id", columns="policy_name", values=m)
            for m in ("viol_rate", "billed")}
    primary, pvals = {}, []
    for name, a, b, metric in PRIMARY:
        d = (wide[metric][a] - wide[metric][b]).to_numpy()
        nz = d[d != 0]
        pv = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
        pvals.append(pv)
        primary[name] = {"metric": metric, "mean_difference": float(d.mean()),
                         "ci95": boot_ci(d, rng), "plans_better": int((d < 0).sum()),
                         "plans_worse": int((d > 0).sum()), "plans_tied": int((d == 0).sum()),
                         "wilcoxon_p": pv}
    for (name, *_), adj in zip(PRIMARY, holm(pvals), strict=True):
        primary[name]["holm_p"] = adj

    exploratory = {}
    for pol in policies:
        if pol == B0:
            continue
        d = (wide["viol_rate"][pol] - wide["viol_rate"][B0]).to_numpy()
        exploratory[f"{pol}_vs_B0_violation"] = {"mean_difference": float(d.mean()),
                                                 "ci95": boot_ci(d, rng)}

    out = {"runs": [str(r) for r in args.runs], "seed": args.seed,
           "per_policy": summary, "primary": primary, "exploratory_vs_B0": exploratory}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

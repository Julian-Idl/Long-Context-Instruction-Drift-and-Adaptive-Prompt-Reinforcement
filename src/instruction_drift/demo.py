"""Live, readable walk-through of one conversation through the real pipeline.

Runs `run_conversation` unchanged and prints each exchange as it happens: conflicts
when they are injected, and at each checkpoint the predicted risk per instruction, the
controller's action and reminder text, the model's reply, and the evaluator verdicts.
Output is also logged to a normal run directory, so nothing shown is staged.
"""

from __future__ import annotations

import textwrap
import time

from instruction_drift.config import ExperimentConfig
from instruction_drift.llm import make_provider
from instruction_drift.reinforcement import make_policy
from instruction_drift.runner import _risk_model, build_plans, run_conversation
from instruction_drift.schemas import Action
from instruction_drift.storage import RunStore
from instruction_drift.tokenization import reference_tokenizer

WIDTH = 92
BAR = "-" * WIDTH


def _wrap(text: str, indent: str = "    ", lines: int = 4) -> str:
    out = textwrap.wrap(" ".join(text.split()), WIDTH - len(indent))
    if len(out) > lines:
        out = out[:lines - 1] + [out[lines - 1][: WIDTH - len(indent) - 5] + " ..."]
    return "\n".join(indent + line for line in out)


def run_demo(cfg: ExperimentConfig, policy_name: str, plan_index: int | None = None,
             out=print) -> dict:
    plans, excluded = build_plans(cfg)
    if not plans:
        raise ValueError(f"no valid plan in this config: {[e.errors for e in excluded]}")
    if plan_index is None:  # first plan that injects a conflict: the interesting case
        plan_index = next((i for i, p in enumerate(plans)
                           if any(e.conflict_kind for e in p.exchanges)), 0)
    plan = plans[plan_index]
    pc = next((p for p in cfg.policies if p.name == policy_name), None)
    if pc is None:
        raise ValueError(f"policy {policy_name!r} not in config: "
                         f"{[p.name for p in cfg.policies]}")
    extra = {"replies": {}, "seed": cfg.seed} if cfg.provider.kind == "mock" else {}
    provider = make_provider(cfg.provider, **extra)
    policy = make_policy(pc, reference_tokenizer())
    risk = _risk_model(pc, provider.kind == "mock") if policy.needs_risk else None
    store = RunStore.create(cfg)

    out(BAR)
    out(f"Conversation {plan.conversation_id}   ({len(plan.exchanges)} exchanges, "
        f"{plan.achieved.checkpoint_count} checkpoints)")
    out(f"Target model: {provider.kind} / {provider.model_id}"
        + ("   [SIMULATED: mock provider]" if provider.kind == "mock" else ""))
    out(f"Policy: {policy.name}")
    out("Instructions:")
    for i in plan.instructions:
        out(_wrap(f"[{i.instruction_id.rsplit('_', 1)[-1]}] {i.original_text}", lines=2))
    out(BAR)
    started = time.monotonic()
    tally = {"checks": 0, "violations": 0, "reminders": 0}

    def show(ev: dict) -> None:
        ex, d = ev["exchange"], ev["decision"]
        k = ex.index
        if ex.conflict_kind and not ex.is_checkpoint:
            out(f"[{k:02d}] CONFLICT injected ({ex.conflict_kind.value}):")
            out(_wrap(ex.user_text, lines=2))
        if d.text:
            tally["reminders"] += 1
        if not ex.is_checkpoint:
            if d.text:
                out(f"[{k:02d}] {d.action.value}: reminder inserted ({d.tokens} tokens)")
            return
        out(f"[{k:02d}] CHECKPOINT   context {ev['prompt_tokens']} tokens"
            + (f"   CONFLICT in this turn ({ex.conflict_kind.value})" if ex.conflict_kind
               else ""))
        out(_wrap("User: " + ex.user_text, lines=2))
        if ev["risks"]:
            risks = ", ".join(f"{iid.rsplit('_', 1)[-1]}={p:.2f}"
                              for iid, p in ev["risks"].items())
            out(f"    Predicted P(violation): {risks}")
        if d.action == Action.NO_ACTION:
            why = "; ".join(dict.fromkeys(r.split(";")[0] for r in d.reasons.values()
                                          if "cooldown" in r or "budget" in r))
            out("    Controller: NO_ACTION" + (f"   ({why})" if why else ""))
        else:
            out(f"    Controller: {d.action.value} ({d.tokens} tokens)")
            out(_wrap(d.text, indent="      > ", lines=3))
        out(_wrap("Model: " + ev["reply"], lines=3))
        for c in ev["compliance"]:
            if not c.applicable:
                continue
            tally["checks"] += 1
            tally["violations"] += c.compliance == 0
            verdict = "PASS" if c.compliance == 1 else "FAIL"
            out(f"    {verdict}  {c.instruction_id.rsplit('_', 1)[-1]:6s} "
                f"{c.evaluator_name}: {c.evidence[:WIDTH - 30]}")
        out("")

    stats = run_conversation(plan, provider, store, cfg, policy, risk, on_exchange=show)
    out(BAR)
    out(f"{tally['checks']} checks, {tally['violations']} violations, "
        f"{tally['reminders']} reminders ({stats['reminder_tokens_billed']} billed reminder "
        f"tokens), {time.monotonic() - started:.0f} s")
    out(f"Full log: {store.run_dir}")
    return stats

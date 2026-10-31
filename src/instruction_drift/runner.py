"""Reproducible experiment runner.

For each validated conversation plan, replay the exchanges. At a checkpoint the exact
prompt is logged, the target model is called, every introduced instruction is
evaluated against the reply, and the model's reply (not the scripted one) goes into
the history. Target execution, evaluation, and storage stay separate steps.

Every configured reinforcement policy runs on the same plans (matched design). The
controller decides from pre-response features only; a reminder is then prepended to
the current user message and stays in the history like any other user text.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from instruction_drift.config import ExperimentConfig, PolicyConfig
from instruction_drift.evaluation import evaluate
from instruction_drift.features import FEATURE_COLUMNS, InstructionHistory, extract_features
from instruction_drift.generation import (
    ConversationPlan,
    Corpus,
    expand_design,
    generate,
    validate_plan,
)
from instruction_drift.llm import BaseProvider, MockProvider, make_provider
from instruction_drift.logs import setup_logging
from instruction_drift.reinforcement import (
    Policy,
    ReinforcementState,
    make_policy,
    max_reminder_tokens,
)
from instruction_drift.schemas import (
    B0_POLICY,
    Action,
    CheckpointRecord,
    ConflictKind,
    ContentType,
    ControllerDecision,
    ExclusionRecord,
    PromptRecord,
    ResponseRecord,
    TurnRecord,
    Usage,
)
from instruction_drift.storage import RunStore
from instruction_drift.tokenization import reference_tokenizer

POLICY_B0 = B0_POLICY
CONTEXT_MARGIN = 1.25  # same margin the Ollama adapter enforces per call
log = logging.getLogger("instruction_drift.runner")

_USER_TYPE = {
    "task": ContentType.TASK,
    "distractor": ContentType.DISTRACTOR,
    "instruction": ContentType.INITIAL_INSTRUCTION,
    "conflict": ContentType.CONFLICT,
    "update": ContentType.CONFLICT,
}


def build_plans(cfg: ExperimentConfig) -> tuple[list[ConversationPlan], list[ExclusionRecord]]:
    if cfg.design is None:
        raise ValueError("config has no design section")
    tok = reference_tokenizer()
    corpus_path = Path(cfg.design.corpus_path)
    corpus = Corpus(corpus_path) if corpus_path.exists() else None
    expected = cfg.design.corpus_sha256
    if expected and (corpus is None or corpus.sha256 != expected):
        raise ValueError(f"corpus at {corpus_path} does not match expected sha256 {expected}")
    gen = cfg.generation
    for pc in cfg.policies:
        make_policy(pc, tok)  # fail on a bad policy before any planning work
    plans, excluded = [], []
    for i, spec in enumerate(expand_design(cfg.design, cfg.seed)):
        cid = f"C{i:05d}_{spec.template_id}"
        try:
            plan = generate(spec, cid, corpus, tok, cfg.system_prompt)
            errors = validate_plan(plan, cfg.design)
        except ValueError as e:
            errors = [str(e)]
        else:
            if gen.context_window:
                # Plans are shared by every policy, so budget for the most verbose one.
                reminders = max(max_reminder_tokens(
                    pc, plan.registry(), len(plan.exchanges), plan.achieved.checkpoint_count,
                    tok) for pc in cfg.policies)
                # Worst case: every model reply runs to max_output_tokens. The margin
                # covers tokenizer mismatch on text; reply budgets are already model tokens.
                text = (plan.achieved.total_tokens + reminders
                        + 8 * (2 * len(plan.exchanges) + 1))
                worst = (text * CONTEXT_MARGIN
                         + (plan.achieved.checkpoint_count + 1) * gen.max_output_tokens)
                if worst > gen.context_window:
                    errors.append(f"may exceed context window {gen.context_window} "
                                  f"(worst case ~{worst:.0f} tokens)")
        if errors:
            excluded.append(ExclusionRecord(conversation_id=cid, spec=spec.model_dump(mode="json"),
                                            errors=errors))
        else:
            plans.append(plan)
    return plans, excluded


def _mock_replies(plans: list[ConversationPlan]) -> dict[str, tuple[str, str | None]]:
    return {ex.user_text: (ex.scripted_reply, ex.violating_reply)
            for p in plans for ex in p.exchanges if ex.is_checkpoint}


def _risk_model(pc: PolicyConfig, simulated: bool):
    from instruction_drift.predictor import Predictor, RuleScore

    if pc.risk_source == "rule":
        return RuleScore()
    pred = Predictor(pc.model_path)
    if pred.meta["is_simulated"] != simulated:
        raise ValueError(f"{pc.name}: predictor trained on "
                         f"{'simulated' if pred.meta['is_simulated'] else 'real'} data")
    return pred


def run_conversation(plan: ConversationPlan, provider: BaseProvider, store: RunStore,
                     cfg: ExperimentConfig, policy: Policy | None = None, risk_model=None,
                     log_plan: bool = True, on_exchange=None) -> dict:
    """`on_exchange`, if given, is called after each exchange with what happened in it
    (used by the live demo; never needed for experiments)."""
    tok = reference_tokenizer()
    policy = policy or make_policy(cfg.policies[0], tok)
    name = policy.name
    reg = plan.registry()
    cid = plan.conversation_id
    if isinstance(provider, MockProvider):
        # Task prompts repeat across templates, so the table must be per conversation.
        provider.replies = _mock_replies([plan])
    if log_plan:
        store.append_many("instructions", plan.instructions)
        if plan.lifecycle_events:
            store.append_many("lifecycle", plan.lifecycle_events)

    group_sizes = Counter(i.group_id for i in plan.instructions)
    first_exchange = min(i.created_at_turn for i in plan.instructions)
    history = {i.instruction_id: InstructionHistory() for i in plan.instructions}
    state = ReinforcementState()
    realized: list[TurnRecord] = []  # turns already in the history, for causal features
    reminder_tokens = 0  # reminder tokens currently in the context
    messages = [{"role": "system", "content": plan.system_prompt}]
    stats = {"checkpoints": 0, "aborted": False, "scored": 0, "violations": 0, "reminders": 0,
             "reminder_tokens_inserted": 0, "reminder_tokens_billed": 0}
    for ex in plan.exchanges:
        k = ex.index
        user_type = (ContentType.UNTRUSTED_QUOTE if ex.conflict_kind == ConflictKind.UNTRUSTED_QUOTE
                     else _USER_TYPE[ex.kind])
        user_turn = TurnRecord(
            conversation_id=cid, turn_id=f"{cid}_X{k:04d}_u", turn_index=k,
            role="user", content=ex.user_text, token_count=ex.user_tokens,
            content_type=user_type, is_distractor=ex.kind == "distractor",
            contains_conflict=ex.conflict_kind is not None, conflict_kind=ex.conflict_kind,
            conflict_target_instruction_id=(plan.instructions[0].group_id
                                            if ex.conflict_kind else None),
            is_checkpoint=ex.is_checkpoint, source_template_id=plan.spec.template_id,
            policy_name=name)
        # B0 keeps its original id format so reruns across code versions still match.
        cp_id = f"{cid}_CP{k:04d}" if name == B0_POLICY else f"{cid}_{name}_CP{k:04d}"
        decision_id = cp_id if ex.is_checkpoint else f"{cid}_{name}_X{k:04d}"

        candidates = reg.applicable_at(k, set(ex.tags))
        risks = None
        if policy.needs_risk and ex.is_checkpoint and candidates:
            rows = [extract_features(plan, k, [*realized, user_turn], i,
                                     history[i.instruction_id], group_sizes[i.group_id])
                    for i in candidates]
            p = risk_model.predict_proba(pd.DataFrame(rows, columns=FEATURE_COLUMNS))
            risks = {i.instruction_id: min(1.0, max(0.0, float(pi)))
                     for i, pi in zip(candidates, p, strict=True)}
        d = policy.decide(k=k, is_checkpoint=ex.is_checkpoint, registry=reg,
                          candidates=candidates, risks=risks, state=state,
                          instruction_exchange=first_exchange)
        ids = set(d.instruction_ids)
        if d.reasons:
            store.append_many("decisions", [ControllerDecision(
                run_id=store.run_id, conversation_id=cid, exchange_index=k,
                checkpoint_id=decision_id, instruction_id=iid,
                predicted_failure_probability=risks.get(iid) if risks else None,
                instruction_priority=reg.get(iid).priority,
                selected_action=d.action if iid in ids else Action.NO_ACTION,
                reason=d.reasons[iid],
                reinforcement_text=d.text if iid in ids else None,
                # The whole reminder's size, shared by every instruction it restates.
                reinforcement_token_count=d.tokens if iid in ids else 0,
                cooldown_applied=iid in d.cooldown_applied, policy_name=name,
                policy_version=policy.version) for iid in sorted(d.reasons)])

        turns = [user_turn]
        content = ex.user_text
        if d.text:
            content = f"{d.text}\n\n{ex.user_text}"
            turns.insert(0, TurnRecord(
                conversation_id=cid, turn_id=f"{cid}_X{k:04d}_r", turn_index=k, role="user",
                content=d.text, token_count=d.tokens, content_type=ContentType.REINFORCEMENT,
                is_checkpoint=ex.is_checkpoint, source_template_id=plan.spec.template_id,
                policy_name=name))
            reminder_tokens += d.tokens
            stats["reminders"] += 1
            stats["reminder_tokens_inserted"] += d.tokens
            if isinstance(provider, MockProvider) and ex.user_text in provider.replies:
                provider.replies[content] = provider.replies[ex.user_text]
        messages.append({"role": "user", "content": content})

        reply, scripted = ex.scripted_reply, True
        outcome: dict[str, int] = {}
        if ex.is_checkpoint:
            stats["checkpoints"] += 1
            resp_id = f"{store.run_id}_{cp_id}"
            prompt_tokens = sum(tok.count(m["content"]) for m in messages)
            stats["reminder_tokens_billed"] += reminder_tokens
            store.append("prompts", PromptRecord(
                run_id=store.run_id, conversation_id=cid, checkpoint_id=cp_id,
                policy_name=name, messages=[dict(m) for m in messages],
                original_token_count=max(0, prompt_tokens - reminder_tokens),
                reinforcement_token_count=reminder_tokens, final_token_count=prompt_tokens,
                reinforcement_role="user" if d.text else None,
                reinforcement_insert_index=len(messages) - 1 if d.text else None))
            res = provider.generate(messages, cfg.generation)
            store.append("responses", ResponseRecord(
                run_id=store.run_id, conversation_id=cid, checkpoint_id=cp_id,
                response_id=resp_id, policy_name=name, provider=provider.kind,
                model_id=provider.model_id, model_version=res.model_version,
                generation=cfg.generation, text=res.text,
                usage=Usage(provider_input_tokens=res.provider_input_tokens,
                            provider_output_tokens=res.provider_output_tokens,
                            local_input_tokens=prompt_tokens,
                            local_output_tokens=tok.count(res.text), tokenizer=tok.name),
                latency_ms=res.latency_ms, request_id=res.request_id,
                retry_count=res.retry_count, error=res.error, finish_reason=res.finish_reason))
            store.append("checkpoints", CheckpointRecord(
                run_id=store.run_id, conversation_id=cid, checkpoint_id=cp_id,
                exchange_index=k, policy_name=name, task_id=ex.task_id,
                topic=ex.topic, task_tags=ex.tags, conflict_kind=ex.conflict_kind,
                prompt_token_count=prompt_tokens, response_id=resp_id, error=res.error))
            if res.error:
                # The history cannot continue without the model's reply.
                log.error("checkpoint failed, aborting conversation",
                          extra={"conversation_id": cid, "checkpoint_id": cp_id,
                                 "error": res.error})
                store.append_many("conversations", turns)
                stats["aborted"] = True
                return stats
            records = []
            for instr in reg:
                if reg.status_at(instr.instruction_id, k) is None:
                    continue
                applicable = reg.is_applicable(instr.instruction_id, k, set(ex.tags))
                rec = evaluate(instr, res.text, applicable=applicable, run_id=store.run_id,
                               conversation_id=cid, checkpoint_id=cp_id, response_id=resp_id,
                               policy_name=name)
                records.append(rec)
                if rec.applicable and rec.compliance is not None:
                    outcome[instr.instruction_id] = rec.compliance
                    stats["scored"] += 1
                    stats["violations"] += rec.compliance == 0
            store.append_many("compliance", records)
            reply, scripted = res.text, False

        # Outcomes and reminders at k enter the history only after the decision at k.
        for iid, c in outcome.items():
            history[iid].outcomes.append((k, c))
        for iid in d.instruction_ids:
            history[iid].reinforcements.append((k, outcome.get(iid)))

        messages.append({"role": "assistant", "content": reply})
        turns.append(TurnRecord(
            conversation_id=cid, turn_id=f"{cid}_X{k:04d}_a", turn_index=k,
            role="assistant", content=reply, token_count=tok.count(reply),
            content_type=ContentType.ASSISTANT_RESPONSE, is_checkpoint=ex.is_checkpoint,
            is_scripted=scripted, source_template_id=plan.spec.template_id, policy_name=name))
        store.append_many("conversations", turns)
        realized.extend(turns)
        if on_exchange:
            on_exchange({"exchange": ex, "decision": d, "risks": risks, "reply": reply,
                         "compliance": records if ex.is_checkpoint else [],
                         "prompt_tokens": prompt_tokens if ex.is_checkpoint else None,
                         "registry": reg})
    return stats


def run_experiment(cfg: ExperimentConfig, provider: BaseProvider | None = None,
                   max_conversations: int | None = None, offset: int = 0,
                   time_budget_s: float | None = None) -> tuple[RunStore, dict]:
    """Run plans[offset : offset + max_conversations] under every configured policy.

    Plans are deterministic, so a long design can be split across sessions with
    `offset`; each session is its own run directory. With a time budget, no new
    conversation starts after the budget is spent and `next_offset` says where to resume.
    """
    started = time.monotonic()
    all_plans, excluded = build_plans(cfg)
    end = None if max_conversations is None else offset + max_conversations
    plans = all_plans[offset:end]
    if not plans:
        raise ValueError(f"no valid conversation plans at offset {offset} "
                         f"({len(all_plans)} valid, {len(excluded)} excluded)")
    if provider is None:
        extra = {"replies": {}, "seed": cfg.seed} if cfg.provider.kind == "mock" else {}
        provider = make_provider(cfg.provider, **extra)
    tok = reference_tokenizer()
    policies = [make_policy(pc, tok) for pc in cfg.policies]
    if len({p.name for p in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    risk_models = {p.name: _risk_model(p.cfg, provider.kind == "mock")
                   for p in policies if p.needs_risk}

    store = RunStore.create(cfg)
    setup_logging(store.run_dir)
    store.append_many("plans", plans)
    if excluded:
        store.append_many("exclusions", excluded)
    log.info("run started", extra={"run_id": store.run_id, "plans": len(plans),
                                   "excluded": len(excluded), "provider": provider.kind,
                                   "model_id": provider.model_id,
                                   "policies": [p.name for p in policies]})

    per_policy: dict[str, Counter] = {p.name: Counter() for p in policies}
    by_template: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    totals = {"conversations": 0, "aborted": 0, "checkpoints": 0}
    stopped_early = False
    for k, plan in enumerate(plans, 1):
        if time_budget_s is not None and time.monotonic() - started > time_budget_s:
            stopped_early = True
            log.warning("time budget spent; stopping before next conversation",
                        extra={"next_offset": offset + k - 1})
            break
        # Every policy runs on this plan before the next one starts, so a time-budget
        # stop never leaves an unmatched plan.
        totals["conversations"] += 1
        for j, pol in enumerate(policies):
            stats = run_conversation(plan, provider, store, cfg, pol,
                                     risk_models.get(pol.name), log_plan=j == 0)
            totals["aborted"] += stats["aborted"]
            totals["checkpoints"] += stats["checkpoints"]
            per_policy[pol.name].update({key: int(v) for key, v in stats.items()})
            t = by_template[(pol.name, plan.spec.template_id)]
            t[0] += stats["scored"]
            t[1] += stats["violations"]
            log.info("conversation done", extra={"conversation_id": plan.conversation_id,
                                                 "policy": pol.name,
                                                 "progress": f"{k}/{len(plans)}", **stats})
    summary = {
        "run_id": store.run_id,
        "is_simulated": provider.kind == "mock",
        "model_version": provider.model_version(),
        "valid_plans": len(all_plans),
        "excluded_plans": len(excluded),
        "plan_offset": offset,
        "next_offset": offset + totals["conversations"],
        "complete": offset + totals["conversations"] >= len(all_plans),
        "stopped_early": stopped_early,
        "wall_clock_s": round(time.monotonic() - started, 1),
        **totals,
        "per_policy": {
            name: {**c, "violation_rate": c["violations"] / c["scored"] if c["scored"] else None,
                   "per_template": {t: {"scored": s, "violations": v,
                                        "violation_rate": v / s if s else None}
                                    for (pn, t), (s, v) in sorted(by_template.items())
                                    if pn == name}}
            for name, c in per_policy.items()},
    }
    if len(policies) == 1:
        summary["per_template"] = summary["per_policy"][policies[0].name]["per_template"]
    with open(store.run_dir / "summary.json", "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return store, summary

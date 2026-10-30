"""Reproducible experiment runner (B0: no reinforcement).

For each validated conversation plan, replay the exchanges. At a checkpoint the exact
prompt is logged, the target model is called, every introduced instruction is
evaluated against the reply, and the model's reply (not the scripted one) goes into
the history. Target execution, evaluation, and storage stay separate steps.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from instruction_drift.config import ExperimentConfig
from instruction_drift.evaluation import evaluate
from instruction_drift.generation import (
    ConversationPlan,
    Corpus,
    expand_design,
    generate,
    validate_plan,
)
from instruction_drift.llm import BaseProvider, MockProvider, make_provider
from instruction_drift.logs import setup_logging
from instruction_drift.schemas import (
    CheckpointRecord,
    ConflictKind,
    ContentType,
    ExclusionRecord,
    PromptRecord,
    ResponseRecord,
    TurnRecord,
    Usage,
)
from instruction_drift.storage import RunStore
from instruction_drift.tokenization import reference_tokenizer

POLICY_B0 = "B0_no_reinforcement"
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
                # Worst case: every model reply runs to max_output_tokens. The margin
                # covers tokenizer mismatch on text; reply budgets are already model tokens.
                text = plan.achieved.total_tokens + 8 * (2 * len(plan.exchanges) + 1)
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


def run_conversation(plan: ConversationPlan, provider: BaseProvider, store: RunStore,
                     cfg: ExperimentConfig, policy: str = POLICY_B0) -> dict:
    tok = reference_tokenizer()
    reg = plan.registry()
    cid = plan.conversation_id
    if isinstance(provider, MockProvider):
        # Task prompts repeat across templates, so the table must be per conversation.
        provider.replies = _mock_replies([plan])
    store.append_many("instructions", plan.instructions)
    if plan.lifecycle_events:
        store.append_many("lifecycle", plan.lifecycle_events)

    messages = [{"role": "system", "content": plan.system_prompt}]
    stats = {"checkpoints": 0, "aborted": False, "scored": 0, "violations": 0}
    for ex in plan.exchanges:
        messages.append({"role": "user", "content": ex.user_text})
        user_type = (ContentType.UNTRUSTED_QUOTE if ex.conflict_kind == ConflictKind.UNTRUSTED_QUOTE
                     else _USER_TYPE[ex.kind])
        turns = [TurnRecord(
            conversation_id=cid, turn_id=f"{cid}_X{ex.index:04d}_u", turn_index=ex.index,
            role="user", content=ex.user_text, token_count=ex.user_tokens,
            content_type=user_type, is_distractor=ex.kind == "distractor",
            contains_conflict=ex.conflict_kind is not None, conflict_kind=ex.conflict_kind,
            conflict_target_instruction_id=(plan.instructions[0].group_id
                                            if ex.conflict_kind else None),
            is_checkpoint=ex.is_checkpoint, source_template_id=plan.spec.template_id)]

        reply, scripted = ex.scripted_reply, True
        if ex.is_checkpoint:
            stats["checkpoints"] += 1
            cp_id = f"{cid}_CP{ex.index:04d}"
            resp_id = f"{store.run_id}_{cp_id}"
            prompt_tokens = sum(tok.count(m["content"]) for m in messages)
            store.append("prompts", PromptRecord(
                run_id=store.run_id, conversation_id=cid, checkpoint_id=cp_id,
                policy_name=policy, messages=[dict(m) for m in messages],
                original_token_count=prompt_tokens, reinforcement_token_count=0,
                final_token_count=prompt_tokens))
            res = provider.generate(messages, cfg.generation)
            store.append("responses", ResponseRecord(
                run_id=store.run_id, conversation_id=cid, checkpoint_id=cp_id,
                response_id=resp_id, policy_name=policy, provider=provider.kind,
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
                exchange_index=ex.index, policy_name=policy, task_id=ex.task_id,
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
                if reg.status_at(instr.instruction_id, ex.index) is None:
                    continue
                applicable = reg.is_applicable(instr.instruction_id, ex.index, set(ex.tags))
                rec = evaluate(instr, res.text, applicable=applicable, run_id=store.run_id,
                               conversation_id=cid, checkpoint_id=cp_id, response_id=resp_id,
                               policy_name=policy)
                records.append(rec)
                if rec.applicable:
                    stats["scored"] += 1
                    stats["violations"] += rec.compliance == 0
            store.append_many("compliance", records)
            reply, scripted = res.text, False

        messages.append({"role": "assistant", "content": reply})
        turns.append(TurnRecord(
            conversation_id=cid, turn_id=f"{cid}_X{ex.index:04d}_a", turn_index=ex.index,
            role="assistant", content=reply, token_count=tok.count(reply),
            content_type=ContentType.ASSISTANT_RESPONSE, is_checkpoint=ex.is_checkpoint,
            is_scripted=scripted, source_template_id=plan.spec.template_id))
        store.append_many("conversations", turns)
    return stats


def run_experiment(cfg: ExperimentConfig, provider: BaseProvider | None = None,
                   max_conversations: int | None = None, offset: int = 0,
                   time_budget_s: float | None = None) -> tuple[RunStore, dict]:
    """Run plans[offset : offset + max_conversations].

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

    store = RunStore.create(cfg)
    setup_logging(store.run_dir)
    store.append_many("plans", plans)
    if excluded:
        store.append_many("exclusions", excluded)
    log.info("run started", extra={"run_id": store.run_id, "plans": len(plans),
                                   "excluded": len(excluded), "provider": provider.kind,
                                   "model_id": provider.model_id})

    by_template: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    totals = {"conversations": 0, "aborted": 0, "checkpoints": 0}
    stopped_early = False
    for k, plan in enumerate(plans, 1):
        if time_budget_s is not None and time.monotonic() - started > time_budget_s:
            stopped_early = True
            log.warning("time budget spent; stopping before next conversation",
                        extra={"next_offset": offset + k - 1})
            break
        stats = run_conversation(plan, provider, store, cfg)
        totals["conversations"] += 1
        totals["aborted"] += stats["aborted"]
        totals["checkpoints"] += stats["checkpoints"]
        t = by_template[plan.spec.template_id]
        t[0] += stats["scored"]
        t[1] += stats["violations"]
        log.info("conversation done", extra={"conversation_id": plan.conversation_id,
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
        "per_template": {k: {"scored": s, "violations": v,
                             "violation_rate": v / s if s else None}
                         for k, (s, v) in sorted(by_template.items())},
    }
    with open(store.run_dir / "summary.json", "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return store, summary

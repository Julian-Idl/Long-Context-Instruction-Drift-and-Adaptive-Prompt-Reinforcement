"""Causal, pre-response feature extraction.

One row = one applicable instruction at one checkpoint. `extract_features` receives only
what exists before the response being predicted:

- the realized, unreinforced conversation up to and including the current user message
  (TurnRecords, so token counts come from the reference tokenizer at run time),
- generator metadata for those same exchanges (distractor/conflict/topic ground truth),
- the instruction record,
- outcomes and reinforcements from strictly earlier checkpoints.

Design factors (planned turn count, planned total tokens, planned position ratio) are
NOT features: they are computed from the whole planned conversation, including turns
that have not happened yet. They are kept as `meta_*` columns for analysis only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from instruction_drift.generation import ConversationPlan
from instruction_drift.schemas import (
    ConflictKind,
    InstructionCategory,
    InstructionRecord,
    TurnRecord,
)

FEATURE_SCHEMA_VERSION = "1.0"
RECENT_WINDOW = 5
NAN = math.nan

FEATURE_COLUMNS = [
    # positional
    "instruction_position_ratio", "tokens_since_instruction", "turns_since_instruction",
    "instruction_token_start", "current_context_tokens", "distance_to_current_task_tokens",
    # context
    "turn_count", "mean_tokens_per_turn", "task_transition_count", "distractor_token_ratio",
    "repeated_content_ratio", "code_char_ratio", "current_task_is_code",
    "current_task_is_repeat", "current_user_tokens",
    # instruction
    "instruction_priority", "is_negative_constraint", "is_conditional", "constraint_count",
    "instruction_word_count",
    *[f"category_{c.value}" for c in InstructionCategory],
    # conflict
    "conflict_count", "conflict_token_ratio", "recent_conflict_count",
    "authorized_conflict_count", "unauthorized_conflict_count",
    "untrusted_instruction_like_content_count", "current_turn_has_conflict",
    # history
    "prior_checkpoint_count", "prior_compliance_rate", "prior_failure_count",
    "consecutive_success_count", "consecutive_failure_count", "turns_since_last_failure",
    "reinforcement_count", "turns_since_last_reinforcement",
    "prior_reinforcement_success_rate",
]


@dataclass
class InstructionHistory:
    """Outcomes known before the current checkpoint, in exchange order."""

    outcomes: list[tuple[int, int]] = field(default_factory=list)  # (exchange, compliance)
    # (exchange, compliance at that checkpoint or None if not yet evaluated)
    reinforcements: list[tuple[int, int | None]] = field(default_factory=list)


def _code_chars(text: str) -> int:
    parts = text.split("```")
    return sum(len(parts[i]) for i in range(1, len(parts), 2))


def extract_features(
    plan: ConversationPlan,
    k: int,
    turns: Sequence[TurnRecord],
    instruction: InstructionRecord,
    history: InstructionHistory,
    group_size: int,
) -> dict[str, float]:
    """Features for `instruction` at checkpoint exchange `k`.

    `turns` must be the realized turns of exchanges 0..k-1 (user and assistant) plus the
    user turn of exchange k, and nothing after it.
    """
    if not turns or turns[-1].turn_index != k or turns[-1].role != "user":
        raise ValueError("turns must end with the user message of exchange k")
    if any(t.turn_index > k for t in turns) or any(
            t.turn_index == k and t.role != "user" for t in turns):
        raise ValueError("turns contain content from after the pre-response point")
    # The controller acts after features are computed, so even a reinforcement at k is future.
    if any(o[0] >= k for o in history.outcomes) or any(r[0] >= k for r in history.reinforcements):
        raise ValueError("history contains outcomes from the current or a later checkpoint")

    exchanges = plan.exchanges[: k + 1]
    j = instruction.created_at_turn
    sys_tokens = plan.system_tokens
    body_tokens = sum(t.token_count for t in turns)
    ctx = sys_tokens + body_tokens
    current_user = turns[-1]
    instr_start = instruction.original_token_start

    tokens_by_exchange: dict[int, int] = {}
    for t in turns:
        tokens_by_exchange[t.turn_index] = tokens_by_exchange.get(t.turn_index, 0) + t.token_count
    distractor_tokens = sum(tokens_by_exchange.get(e.index, 0) for e in exchanges
                            if e.kind == "distractor")

    seen: set[str] = set()
    repeated_tokens = user_tokens = 0
    for t in turns:
        if t.role != "user":
            continue
        user_tokens += t.token_count
        if t.content in seen:
            repeated_tokens += t.token_count
        seen.add(t.content)
    current_is_repeat = any(t.role == "user" and t.content == current_user.content
                            for t in turns[:-1])
    total_chars = sum(len(t.content) for t in turns)
    code_chars = sum(_code_chars(t.content) for t in turns)

    post = [e for e in exchanges if e.index > j]
    topics = [e.topic for e in post if e.kind in ("task", "conflict", "update")]
    conflicts = [e for e in post if e.conflict_kind is not None]
    conflict_user_tokens = sum(e.user_tokens for e in conflicts)

    def count(kind: ConflictKind) -> int:
        return sum(1 for e in conflicts if e.conflict_kind == kind)

    outcomes = [c for _, c in history.outcomes]
    consec_s = consec_f = 0
    for c in reversed(outcomes):
        if c == 1 and consec_f == 0:
            consec_s += 1
        elif c == 0 and consec_s == 0:
            consec_f += 1
        else:
            break
    failures = [e for e, c in history.outcomes if c == 0]
    reinf_outcomes = [c for _, c in history.reinforcements if c is not None]

    feats: dict[str, float] = {
        "instruction_position_ratio": instr_start / ctx,
        "tokens_since_instruction": ctx - instr_start,
        "turns_since_instruction": k - j,
        "instruction_token_start": instr_start,
        "current_context_tokens": ctx,
        "distance_to_current_task_tokens": ctx - current_user.token_count - instr_start,
        "turn_count": k + 1,
        "mean_tokens_per_turn": body_tokens / (k + 1),
        "task_transition_count": sum(a != b for a, b in zip(topics, topics[1:], strict=False)),
        "distractor_token_ratio": distractor_tokens / body_tokens,
        "repeated_content_ratio": repeated_tokens / user_tokens if user_tokens else 0.0,
        "code_char_ratio": code_chars / total_chars if total_chars else 0.0,
        "current_task_is_code": float("code_requested" in exchanges[k].tags),
        "current_task_is_repeat": float(current_is_repeat),
        "current_user_tokens": current_user.token_count,
        "instruction_priority": instruction.priority,
        "is_negative_constraint": float(
            instruction.category == InstructionCategory.NEGATIVE_CONSTRAINT),
        "is_conditional": float(instruction.scope.kind == "conditional"),
        "constraint_count": group_size,
        "instruction_word_count": len(instruction.original_text.split()),
        **{f"category_{c.value}": float(instruction.category == c) for c in InstructionCategory},
        "conflict_count": len(conflicts),
        "conflict_token_ratio": conflict_user_tokens / body_tokens,
        "recent_conflict_count": sum(1 for e in conflicts if e.index > k - RECENT_WINDOW),
        "authorized_conflict_count": count(ConflictKind.AUTHORIZED_UPDATE),
        "unauthorized_conflict_count": count(ConflictKind.UNAUTHORIZED_CONTRADICTION),
        "untrusted_instruction_like_content_count": count(ConflictKind.UNTRUSTED_QUOTE),
        "current_turn_has_conflict": float(exchanges[k].conflict_kind is not None),
        "prior_checkpoint_count": len(outcomes),
        "prior_compliance_rate": sum(outcomes) / len(outcomes) if outcomes else NAN,
        "prior_failure_count": len(failures),
        "consecutive_success_count": consec_s,
        "consecutive_failure_count": consec_f,
        "turns_since_last_failure": k - failures[-1] if failures else NAN,
        "reinforcement_count": len(history.reinforcements),
        "turns_since_last_reinforcement": (k - history.reinforcements[-1][0]
                                           if history.reinforcements else NAN),
        "prior_reinforcement_success_rate": (sum(reinf_outcomes) / len(reinf_outcomes)
                                             if reinf_outcomes else NAN),
    }
    assert list(feats) == FEATURE_COLUMNS
    return feats

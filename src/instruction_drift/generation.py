"""Controlled conversation generation and pre-flight scenario validation.

A conversation is a sequence of exchanges (one user message + one assistant reply).
Every assistant reply is scripted, except at checkpoints, where the target model
answers and its reply replaces the scripted one in the history.

Definitions (also in docs/GENERATION.md):
- Instruction position ratio = reference tokens before the instruction message /
  planned total tokens (system prompt + all scripted messages).
- Distractor density = tokens of distractor exchanges (user text + scripted ack) /
  planned total tokens excluding the system prompt.
- Conflict intensity = conflict exchanges / eligible exchanges, where eligible means
  post-instruction task exchanges other than the first (baseline) checkpoint.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from instruction_drift.config import DesignConfig
from instruction_drift.evaluation import EVALUATORS
from instruction_drift.registry import InstructionRegistry
from instruction_drift.schemas import (
    ConflictKind,
    InstructionRecord,
    LifecycleEvent,
    LifecycleStatus,
    Record,
    Scope,
    Verification,
)
from instruction_drift.templates import (
    DISTRACTOR_ACK,
    INSTRUCTION_ACK,
    PROSE_TASKS,
    TEMPLATE_VERSION,
    TEMPLATES,
    Task,
    Template,
    _plain,
)
from instruction_drift.tokenization import Tokenizer

GENERATOR_VERSION = "1.0"
DISTRACTOR_INTRO = "Unrelated, but I was reading this earlier:\n\n"

ExchangeKind = Literal["task", "distractor", "instruction", "conflict", "update"]


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    template_id: str
    turn_count: int = Field(ge=2)
    instruction_position_ratio: float = Field(ge=0.0, lt=1.0)
    distractor_density: float = Field(ge=0.0, lt=1.0)
    conflict_intensity: float = Field(ge=0.0, le=1.0)
    conflict_kinds: list[ConflictKind]
    task_transition_count: int = Field(ge=0)
    distractor_tokens_per_exchange: int = Field(gt=0)
    checkpoint_interval: int = Field(gt=0)
    replicate: int = 0
    seed: int


class Exchange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    index: int
    kind: ExchangeKind
    user_text: str
    scripted_reply: str
    violating_reply: str | None = None  # mock provider only; never shown to real models
    tags: list[str]
    task_id: str | None = None
    topic: str | None = None
    is_checkpoint: bool = False
    conflict_kind: ConflictKind | None = None
    user_tokens: int
    reply_tokens: int


class Achieved(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total_tokens: int
    instruction_exchange: int
    instruction_token_start: int
    instruction_position_ratio: float
    distractor_density: float
    conflict_counts: dict[str, int]
    conflict_intensity: float
    task_transition_count: int
    checkpoint_count: int


class ConversationPlan(Record):
    conversation_id: str
    generator_version: str = GENERATOR_VERSION
    template_version: str = TEMPLATE_VERSION
    tokenizer: str
    corpus_sha256: str | None
    system_prompt: str
    system_tokens: int
    spec: ScenarioSpec
    achieved: Achieved
    instructions: list[InstructionRecord]
    lifecycle_events: list[LifecycleEvent]
    exchanges: list[Exchange]

    def registry(self) -> InstructionRegistry:
        return InstructionRegistry(self.instructions, self.lifecycle_events)


class Corpus:
    def __init__(self, path: Path):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.paragraphs = [json.loads(line)["text"] for line in raw.decode("utf-8").splitlines()
                           if line.strip()]

    def take(self, n_tokens: int, rng: random.Random, tok: Tokenizer, used: set[int]) -> str:
        """Consecutive paragraphs from a random start, cut to ~n_tokens at a sentence end."""
        start = rng.randrange(len(self.paragraphs))
        for _ in range(100):
            if start not in used:
                break
            start = rng.randrange(len(self.paragraphs))
        parts, i = [], start
        while tok.count("\n\n".join(parts)) < n_tokens and len(parts) < len(self.paragraphs):
            used.add(i % len(self.paragraphs))
            parts.append(self.paragraphs[i % len(self.paragraphs)])
            i += 1
        text = tok.truncate("\n\n".join(parts), n_tokens)
        cut = text.rfind(". ")
        return text[: cut + 1] if cut > 0.9 * len(text) else text


def _derive_seed(base: int, index: int) -> int:
    return int(hashlib.sha256(f"{base}:{index}".encode()).hexdigest()[:8], 16)


def expand_design(design: DesignConfig, seed: int) -> list[ScenarioSpec]:
    grid = list(itertools.product(
        design.templates, design.turn_counts, design.position_ratios,
        design.distractor_densities, design.conflict_intensities,
        design.task_transition_counts, range(design.replicates)))
    if design.max_conversations and design.max_conversations < len(grid):
        grid = random.Random(seed).sample(grid, design.max_conversations)
    return [
        ScenarioSpec(
            template_id=t, turn_count=n, instruction_position_ratio=r, distractor_density=d,
            conflict_intensity=c, conflict_kinds=design.conflict_kinds,
            task_transition_count=k, replicate=rep,
            distractor_tokens_per_exchange=design.distractor_tokens_per_exchange,
            checkpoint_interval=design.checkpoint_interval, seed=_derive_seed(seed, i))
        for i, (t, n, r, d, c, k, rep) in enumerate(grid)
    ]


class _TaskCycle:
    """Draws tasks per topic without replacement, reshuffling when a topic runs out."""

    def __init__(self, tasks: tuple[Task, ...] | list[Task], rng: random.Random):
        self.rng = rng
        self.by_topic: dict[str, list[Task]] = {}
        for t in tasks:
            self.by_topic.setdefault(t.topic, []).append(t)
        self.queues: dict[str, list[Task]] = {k: [] for k in self.by_topic}

    @cached_property
    def topics(self) -> list[str]:
        return sorted(self.by_topic)

    def next(self, topic: str) -> Task:
        if not self.queues[topic]:
            self.queues[topic] = self.rng.sample(self.by_topic[topic], len(self.by_topic[topic]))
        return self.queues[topic].pop()


def _topic_schedule(n_slots: int, transitions: int, topics: list[str], rng: random.Random):
    if n_slots == 0:
        return []
    segments = (min(transitions, n_slots - 1) if len(topics) > 1 else 0) + 1
    seq = [rng.choice(topics)]
    for _ in range(segments - 1):
        seq.append(rng.choice([t for t in topics if t != seq[-1]]))
    bounds = [round(i * n_slots / segments) for i in range(segments + 1)]
    return [seq[s] for s in range(segments) for _ in range(bounds[s + 1] - bounds[s])]


MIN_DISTRACTOR_TOKENS = 40
LAYOUT_SEARCH = 32


def _solve_distractor_budget(a: float, b: float, sys_tokens: int, n_pre: int, n_post: int,
                             r: float, d: float) -> tuple[float, float, float]:
    """Split distractor tokens into (before, after) the instruction.

    a = non-distractor tokens before the instruction (incl. system prompt), b = the
    instruction exchange plus non-distractor tokens after it. The total distractor
    budget S is fixed by the density target; the split is chosen to hit the position
    target, clamped by which side actually has distractor slots.
    Returns (x_before, y_after, achieved_position).
    """
    if n_pre + n_post == 0:
        return 0.0, 0.0, a / (a + b)
    s = d * (a + b - sys_tokens) / (1 - d)
    t = a + b + s
    lo = MIN_DISTRACTOR_TOKENS * n_pre
    hi = s - MIN_DISTRACTOR_TOKENS * n_post
    x = 0.0 if n_pre == 0 else s if n_post == 0 else min(max(r * t - a, lo), max(lo, hi))
    return x, s - x, (a + x) / t


def generate(spec: ScenarioSpec, conversation_id: str, corpus: Corpus | None, tok: Tokenizer,
             system_prompt: str) -> ConversationPlan:
    tpl: Template = TEMPLATES[spec.template_id]
    rng = random.Random(spec.seed)
    n, d, r = spec.turn_count, spec.distractor_density, spec.instruction_position_ratio
    L = spec.distractor_tokens_per_exchange
    sys_tokens = tok.count(system_prompt)

    s_t = sum(tok.count(t.prompt) + tok.count(tpl.render(t.answer, t))
              for t in tpl.tasks) / len(tpl.tasks)
    s_pre = sum(tok.count(t.prompt) + tok.count(t.answer) for t in PROSE_TASKS) / len(PROSE_TASKS)
    instr_msg = tpl.instruction_message()
    instr_tokens = tok.count(instr_msg) + tok.count(tpl.render(INSTRUCTION_ACK, None))

    # Number of distractor exchanges from the nominal size L; their exact sizes are
    # solved later so that position and density hit their targets.
    n_d = 0 if d == 0 else round(d * n * s_t / (L * (1 - d) + d * s_t))
    n_d = max(0, min(n_d, n - 2))
    if n_d and corpus is None:
        raise ValueError("Distractors requested but no corpus provided")
    def estimate(kinds: list[str], j: int) -> tuple[float, float]:
        n_pre = kinds[:j].count("distractor")
        n_post = kinds[j + 1:].count("distractor")
        a = sys_tokens + kinds[:j].count("task") * s_pre
        b = instr_tokens + kinds[j + 1:].count("task") * s_t
        x, y, pos = _solve_distractor_budget(a, b, sys_tokens, n_pre, n_post, r, d)
        sizes = ([x / n_pre] * n_pre if n_pre else []) + ([y / n_post] * n_post if n_post else [])
        # Prefer exact position, then distractor sizes close to the nominal L.
        return abs(pos - r), sum(abs(s - L) for s in sizes) / max(1, len(sizes))

    # Random layouts of distractor slots; keep the one whose best instruction slot
    # gets closest to the targets. All draws come from the seeded rng.
    best = None
    for _ in range(LAYOUT_SEARCH):
        layout = ["distractor"] * n_d + ["task"] * (n - n_d)
        rng.shuffle(layout)
        for jj, k in enumerate(layout):
            if k == "task" and "task" in layout[jj + 1:]:
                score = estimate(layout, jj)
                if best is None or score < best[0]:
                    best = (score, layout, jj)
    if best is None:
        raise ValueError("No task slot leaves a task after the instruction")
    _, kinds, j = best
    kinds[j] = "instruction"

    post = [i for i in range(j + 1, n) if kinds[i] == "task"]
    eligible = post[1:]
    n_c = round(spec.conflict_intensity * len(eligible))
    conflict_at = sorted(rng.sample(eligible, n_c)) if n_c else []
    kind_cycle = rng.sample(spec.conflict_kinds, len(spec.conflict_kinds))
    non_auth = [k for k in kind_cycle if k != ConflictKind.AUTHORIZED_UPDATE]
    conflict_kind: dict[int, ConflictKind] = {}
    auth_used = False
    for q, i in enumerate(conflict_at):
        k = kind_cycle[q % len(kind_cycle)]
        if k == ConflictKind.AUTHORIZED_UPDATE:
            if auth_used:  # one authorized update retires the rule; later ones would be moot
                if not non_auth:
                    continue
                k = non_auth[q % len(non_auth)]
            auth_used = True
        conflict_kind[i] = k
    for i, k in conflict_kind.items():
        kinds[i] = "update" if k == ConflictKind.AUTHORIZED_UPDATE else "conflict"
    update_at = next((i for i, k in conflict_kind.items()
                      if k == ConflictKind.AUTHORIZED_UPDATE), None)

    task_bearing = [i for i in range(j + 1, n) if kinds[i] in ("task", "conflict", "update")]
    cycle = _TaskCycle(tpl.tasks, rng)
    topics = dict(zip(task_bearing, _topic_schedule(
        len(task_bearing), spec.task_transition_count, cycle.topics, rng), strict=True))
    pre_cycle = _TaskCycle(PROSE_TASKS, rng)

    checkpoints = set(task_bearing[:1]) | set(conflict_kind)
    tb = set(task_bearing)
    for target in range(j + spec.checkpoint_interval, n, spec.checkpoint_interval):
        hit = next((i for i in range(target, min(n, target + spec.checkpoint_interval))
                    if i in tb), None)
        if hit is not None:
            checkpoints.add(hit)

    def live(i: int) -> bool:
        return j < i and (update_at is None or i < update_at)

    # Pass 1: every non-distractor exchange, with exact token counts.
    fields: dict[int, dict] = {}
    for i, kind in enumerate(kinds):
        if kind == "distractor":
            continue
        violating = task_id = topic = None
        if kind == "instruction":
            user, reply, tags = instr_msg, tpl.render(INSTRUCTION_ACK, None), ["instruction"]
        elif i < j:
            task = pre_cycle.next(pre_cycle.topics[i % len(pre_cycle.topics)])
            user, reply, tags = task.prompt, task.answer, sorted(task.tags)
            task_id, topic = task.task_id, task.topic
        else:
            task = cycle.next(topics[i])
            task_id, topic = task.task_id, task.topic
            user = task.prompt
            if i in conflict_kind:
                user = f"{tpl.conflicts[conflict_kind[i]]}\n\n{task.prompt}"
            reply = tpl.render(task.answer, task) if live(i) else _plain(task.answer, task)
            violating = tpl.render_violation(task.answer, task) if live(i) else None
            tags = sorted(task.tags)
        fields[i] = dict(kind=kind, user_text=user, scripted_reply=reply,
                         violating_reply=violating, tags=tags, task_id=task_id, topic=topic)

    def tokens_of(i: int) -> int:
        return tok.count(fields[i]["user_text"]) + tok.count(fields[i]["scripted_reply"])

    a = sys_tokens + sum(tokens_of(i) for i in range(j) if i in fields)
    b = sum(tokens_of(i) for i in range(j, n) if i in fields)
    n_pre = kinds[:j].count("distractor")
    n_post = kinds[j + 1:].count("distractor")
    x, y, _ = _solve_distractor_budget(a, b, sys_tokens, n_pre, n_post, r, d)

    # Pass 2: distractors sized to their side's share of the budget.
    used: set[int] = set()
    for i, kind in enumerate(kinds):
        if kind != "distractor":
            continue
        ack = tpl.render(DISTRACTOR_ACK, None) if live(i) else DISTRACTOR_ACK
        share = x / n_pre if i < j else y / n_post
        budget = max(1, round(share) - tok.count(DISTRACTOR_INTRO) - tok.count(ack))
        text = corpus.take(budget, rng, tok, used)
        fields[i] = dict(kind=kind, user_text=DISTRACTOR_INTRO + text,
                         scripted_reply=ack, violating_reply=None, tags=["distractor"],
                         task_id=None, topic=None)

    exchanges = [
        Exchange(index=i, **fields[i], is_checkpoint=i in checkpoints,
                 conflict_kind=conflict_kind.get(i), user_tokens=tok.count(fields[i]["user_text"]),
                 reply_tokens=tok.count(fields[i]["scripted_reply"]))
        for i in range(n)
    ]
    instr_start = sys_tokens + sum(e.user_tokens + e.reply_tokens for e in exchanges[:j])

    group_id = f"{conversation_id}_G0"
    instructions: list[InstructionRecord] = []
    for q, s in enumerate(tpl.instructions):
        ev = EVALUATORS[s.evaluator]
        instructions.append(InstructionRecord(
            instruction_id=f"{conversation_id}_I{q}", group_id=group_id,
            conversation_id=conversation_id, original_text=s.text, normalized_text=s.normalized,
            compressed_text=s.compressed, category=tpl.category, priority=tpl.priority,
            scope=Scope(kind=s.scope_kind, start_turn=j, condition=s.condition),
            verification=Verification(method="deterministic", evaluator_name=ev.name,
                                      evaluator_version=ev.version, params=s.params),
            created_at_turn=j, original_token_start=instr_start))
    events = [] if update_at is None else [
        LifecycleEvent(conversation_id=conversation_id, instruction_id=rec.instruction_id,
                       turn_index=update_at, new_status=LifecycleStatus.SUPERSEDED,
                       source_turn_id=f"{conversation_id}_X{update_at:04d}",
                       reason="authorized update by user")
        for rec in instructions]

    total = sys_tokens + sum(e.user_tokens + e.reply_tokens for e in exchanges)
    distractor_tokens = sum(e.user_tokens + e.reply_tokens for e in exchanges
                            if e.kind == "distractor")
    post_topics = [exchanges[i].topic for i in task_bearing]
    counts = {k.value: sum(1 for v in conflict_kind.values() if v == k) for k in ConflictKind}
    achieved = Achieved(
        total_tokens=total, instruction_exchange=j, instruction_token_start=instr_start,
        instruction_position_ratio=instr_start / total,
        distractor_density=distractor_tokens / (total - sys_tokens),
        conflict_counts=counts,
        conflict_intensity=len(conflict_kind) / len(eligible) if eligible else 0.0,
        task_transition_count=sum(p != q for p, q in itertools.pairwise(post_topics)),
        checkpoint_count=len(checkpoints))
    return ConversationPlan(
        conversation_id=conversation_id, tokenizer=tok.name,
        corpus_sha256=corpus.sha256 if corpus else None, system_prompt=system_prompt,
        system_tokens=sys_tokens, spec=spec, achieved=achieved, instructions=instructions,
        lifecycle_events=events, exchanges=exchanges)


def validate_plan(plan: ConversationPlan, design: DesignConfig | None = None) -> list[str]:
    """Pre-flight checks. An empty list means the plan may be sent to a model."""
    spec, a = plan.spec, plan.achieved
    pos_tol = design.position_tolerance if design else 0.05
    den_tol = design.density_tolerance if design else 0.05
    errors = []
    if len(plan.exchanges) != spec.turn_count:
        errors.append(f"turn count {len(plan.exchanges)} != {spec.turn_count}")
    if abs(a.instruction_position_ratio - spec.instruction_position_ratio) > pos_tol:
        errors.append(f"position {a.instruction_position_ratio:.3f} not within {pos_tol} of "
                      f"{spec.instruction_position_ratio}")
    if abs(a.distractor_density - spec.distractor_density) > den_tol:
        errors.append(f"distractor density {a.distractor_density:.3f} not within {den_tol} of "
                      f"{spec.distractor_density}")
    if spec.conflict_intensity > 0 and sum(a.conflict_counts.values()) == 0:
        errors.append("conflicts requested but none placed")
    if a.task_transition_count != spec.task_transition_count:
        errors.append(f"task transitions {a.task_transition_count} != "
                      f"{spec.task_transition_count}")
    if plan.lifecycle_events and ConflictKind.AUTHORIZED_UPDATE not in spec.conflict_kinds:
        errors.append("supersession present but authorized updates were not requested")

    reg = plan.registry()
    for instr in plan.instructions:
        ev = EVALUATORS.get(instr.verification.evaluator_name)
        if ev is None or ev.version != instr.verification.evaluator_version:
            errors.append(f"{instr.instruction_id}: evaluator unavailable")
    scored = 0
    for ex in plan.exchanges:
        for instr in plan.instructions:
            if not reg.is_applicable(instr.instruction_id, ex.index, set(ex.tags)):
                continue
            if ex.is_checkpoint:
                scored += 1
            ev = EVALUATORS[instr.verification.evaluator_name]
            res = ev.fn(ex.scripted_reply, instr.verification.params)
            if res.compliance is False:
                errors.append(f"scripted reply at exchange {ex.index} violates "
                              f"{instr.instruction_id}: {res.evidence}")
    if scored == 0:
        errors.append("no checkpoint has an applicable instruction")
    return errors

"""Reinforcement actions and policies.

Reminder text is deterministic (no LLM compression):
- FULL / SELECTIVE restate the original text of every instruction group involved, so
  meaning is preserved exactly (a partially selected group is restated whole rather
  than cut mid-sentence).
- COMPRESSED uses each instruction's hand-written `compressed_text`.
- POLICY_CHECKPOINT lists every reinforceable instruction's normalized text with its
  priority and, for conditional instructions, the condition under which it applies.

Superseded, expired and ambiguous instructions are never mentioned
(`InstructionRegistry.reinforceable_at`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instruction_drift.config import PolicyConfig
from instruction_drift.registry import InstructionRegistry
from instruction_drift.schemas import Action, InstructionRecord
from instruction_drift.tokenization import Tokenizer

REINFORCEMENT_VERSION = "1.0"
HEADER = "Reminder of the rules that still apply in this conversation:"
CHECKPOINT_HEADER = "Policy checkpoint. These rules are still in force, unchanged:"
CONDITION_TEXT = {"code_requested": "applies whenever code is requested"}


def render(action: Action, selected: list[InstructionRecord],
           group_members: dict[str, list[InstructionRecord]]) -> str | None:
    if action == Action.NO_ACTION or not selected:
        return None
    if action in (Action.FULL_REINFORCEMENT, Action.SELECTIVE_REINFORCEMENT):
        groups = list(dict.fromkeys(i.group_id for i in selected))
        lines = [" ".join(m.original_text for m in group_members[g]) for g in groups]
        return HEADER + "\n" + "\n".join(f"- {line}" for line in lines)
    if action == Action.COMPRESSED_REINFORCEMENT:
        return HEADER + "\n" + "\n".join(f"- {i.compressed_text}" for i in selected)
    if action == Action.POLICY_CHECKPOINT:
        lines = []
        for n, i in enumerate(selected, 1):
            scope = (f" ({CONDITION_TEXT.get(i.scope.condition, i.scope.condition)})"
                     if i.scope.kind == "conditional" else "")
            lines.append(f"{n}. [priority {i.priority:g}] {i.normalized_text}{scope}")
        return CHECKPOINT_HEADER + "\n" + "\n".join(lines)
    raise ValueError(action)


@dataclass
class ReinforcementState:
    """Per-conversation controller memory."""

    last_reinforced: dict[str, int] = field(default_factory=dict)
    tokens_inserted: int = 0


@dataclass
class Decision:
    action: Action
    instruction_ids: list[str]
    text: str | None
    tokens: int
    reasons: dict[str, str]
    cooldown_applied: set[str] = field(default_factory=set)


class Policy:
    def __init__(self, cfg: PolicyConfig, tok: Tokenizer):
        self.cfg = cfg
        self.name = cfg.name
        self.version = REINFORCEMENT_VERSION
        self.tok = tok
        self.needs_risk = cfg.kind == "threshold"

    def decide(self, *, k: int, is_checkpoint: bool, registry: InstructionRegistry,
               candidates: list[InstructionRecord], risks: dict[str, float] | None,
               state: ReinforcementState, instruction_exchange: int) -> Decision:
        """`candidates`: instructions applicable at exchange k. `risks`: their predicted
        failure probabilities, computed from pre-response features only."""
        groups = group_members(registry)
        none = Decision(Action.NO_ACTION, [], None, 0, {})
        cfg = self.cfg
        if cfg.kind == "none":
            return none

        if cfg.kind == "fixed_interval":
            if k <= instruction_exchange or (k - instruction_exchange) % cfg.interval:
                return none
            # A fixed schedule restates the whole rule set, like a user repeating it.
            action, selected = cfg.action, registry.reinforceable_at(k)
            reasons = {i.instruction_id: f"fixed interval {cfg.interval}" for i in selected}
            return self._finish(action, selected, groups, reasons, state, k, set())

        # threshold
        if not is_checkpoint or risks is None or not candidates:
            return none
        reasons, cooled = {}, set()
        eligible = []
        for i in candidates:
            last = state.last_reinforced.get(i.instruction_id)
            if last is not None and k - last <= cfg.cooldown_exchanges:
                cooled.add(i.instruction_id)
                reasons[i.instruction_id] = f"cooldown (last reinforced at exchange {last})"
            elif i.priority < cfg.min_priority:
                reasons[i.instruction_id] = "below minimum priority"
            else:
                eligible.append(i)
        if not eligible:
            d = none
            d.reasons, d.cooldown_applied = reasons, cooled
            return d
        risk = {i.instruction_id: risks[i.instruction_id] for i in eligible}
        if cfg.single_threshold is not None:
            at_risk = [i for i in eligible if risk[i.instruction_id] >= cfg.single_threshold]
            action = cfg.top_action if at_risk else Action.NO_ACTION
        else:
            b = cfg.bands
            top = max(risk.values())
            at_risk = [i for i in eligible if risk[i.instruction_id] >= b[1]]
            if top < b[1]:
                action = Action.NO_ACTION
            elif top < b[2]:
                action = Action.COMPRESSED_REINFORCEMENT
            elif top < b[3]:
                action = Action.SELECTIVE_REINFORCEMENT
            else:
                action = cfg.top_action
        if action in (Action.FULL_REINFORCEMENT, Action.POLICY_CHECKPOINT):
            selected = (registry.reinforceable_at(k) if action == Action.POLICY_CHECKPOINT
                        else eligible)
        else:
            selected = at_risk
        for i in eligible:
            band = "at risk" if i in at_risk else "below band"
            reasons[i.instruction_id] = (f"p={risk[i.instruction_id]:.3f} {band}; action "
                                         f"{action.value}")
        if action == Action.NO_ACTION:
            return Decision(action, [], None, 0, reasons, cooled)
        return self._finish(action, selected, groups, reasons, state, k, cooled)

    def _finish(self, action: Action, selected: list[InstructionRecord], groups, reasons,
                state: ReinforcementState, k: int, cooled: set[str]) -> Decision:
        text = render(action, selected, groups)
        if text is None:
            return Decision(Action.NO_ACTION, [], None, 0, reasons, cooled)
        tokens = self.tok.count(text)
        budget = self.cfg.max_reinforcement_tokens
        if budget is not None and state.tokens_inserted + tokens > budget:
            # Downgrade once to the compressed form before giving up.
            short = render(Action.COMPRESSED_REINFORCEMENT, selected, groups)
            if state.tokens_inserted + self.tok.count(short) <= budget:
                action, text, tokens = Action.COMPRESSED_REINFORCEMENT, short, self.tok.count(short)
                for i in selected:
                    reasons[i.instruction_id] = reasons.get(i.instruction_id, "") + \
                        "; downgraded to compressed by token budget"
            else:
                for i in selected:
                    reasons[i.instruction_id] = reasons.get(i.instruction_id, "") + \
                        "; skipped: token budget exhausted"
                return Decision(Action.NO_ACTION, [], None, 0, reasons, cooled)
        state.tokens_inserted += tokens
        for i in selected:
            state.last_reinforced[i.instruction_id] = k
        return Decision(action, [i.instruction_id for i in selected], text, tokens, reasons,
                        cooled)


def group_members(registry: InstructionRegistry) -> dict[str, list[InstructionRecord]]:
    groups: dict[str, list[InstructionRecord]] = {}
    for i in registry:
        groups.setdefault(i.group_id, []).append(i)
    return groups


def max_reminder_tokens(cfg: PolicyConfig, registry: InstructionRegistry, n_exchanges: int,
                        n_checkpoints: int, tok: Tokenizer) -> int:
    """Upper bound on reminder tokens one conversation can accumulate under `cfg`."""
    if cfg.kind == "none":
        return 0
    instrs, groups = list(registry), group_members(registry)
    per = max(tok.count(render(a, instrs, groups)) for a in Action if a != Action.NO_ACTION)
    n = n_exchanges // cfg.interval if cfg.kind == "fixed_interval" else n_checkpoints
    budget = cfg.max_reinforcement_tokens
    return min(n * per, budget) if budget else n * per


def make_policy(cfg: PolicyConfig, tok: Tokenizer) -> Policy:
    if cfg.kind == "fixed_interval" and cfg.interval is None:
        raise ValueError(f"{cfg.name}: fixed_interval needs interval")
    if cfg.kind == "threshold" and cfg.risk_source == "model" and cfg.model_path is None:
        raise ValueError(f"{cfg.name}: risk_source model needs model_path")
    if cfg.kind == "threshold" and cfg.single_threshold is None and len(cfg.bands) != 4:
        raise ValueError(f"{cfg.name}: bands needs 4 ascending edges")
    return Policy(cfg, tok)

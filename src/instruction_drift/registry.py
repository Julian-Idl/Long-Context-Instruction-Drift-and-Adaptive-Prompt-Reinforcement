"""Instruction registry: the authoritative answer to "which instructions apply at turn t?".

Instruction records are immutable. Status at a turn is derived from the record's scope
plus lifecycle events (supersession, ambiguity), so it can be recomputed for any turn
without replaying mutable state.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from instruction_drift.schemas import InstructionRecord, LifecycleEvent, LifecycleStatus

APPLICABLE = {LifecycleStatus.ACTIVE, LifecycleStatus.TEMPORARY}
NEVER_REINFORCE = {LifecycleStatus.SUPERSEDED, LifecycleStatus.EXPIRED, LifecycleStatus.AMBIGUOUS}


class InstructionExtractor(Protocol):
    """Turns free-text instructions into atomic registry entries.

    Controlled synthetic conversations do not need extraction: the generator emits
    ground-truth InstructionRecords. An implementation is only required for natural
    benchmark conversations, and must document how it assigns category and scope.
    """

    def extract(self, text: str, conversation_id: str, turn_index: int) -> list[InstructionRecord]:
        ...


class InstructionRegistry:
    def __init__(
        self, instructions: Iterable[InstructionRecord], events: Iterable[LifecycleEvent] = ()
    ):
        self._instructions = {i.instruction_id: i for i in instructions}
        self._events: dict[str, list[LifecycleEvent]] = {}
        for e in events:
            self.add_event(e)

    def add_event(self, event: LifecycleEvent) -> None:
        if event.instruction_id not in self._instructions:
            raise KeyError(f"Unknown instruction {event.instruction_id}")
        if event.new_status not in {LifecycleStatus.SUPERSEDED, LifecycleStatus.AMBIGUOUS}:
            raise ValueError("Only supersession and ambiguity are event-driven; other statuses "
                             "follow from the instruction's scope")
        self._events.setdefault(event.instruction_id, []).append(event)

    def __iter__(self):
        return iter(self._instructions.values())

    def get(self, instruction_id: str) -> InstructionRecord:
        return self._instructions[instruction_id]

    def status_at(self, instruction_id: str, turn: int) -> LifecycleStatus | None:
        """Status at `turn`, or None if the instruction has not been given yet."""
        instr = self._instructions[instruction_id]
        if turn < instr.scope.start_turn:
            return None
        past = [e for e in self._events.get(instruction_id, []) if e.turn_index <= turn]
        if any(e.new_status == LifecycleStatus.SUPERSEDED for e in past):
            return LifecycleStatus.SUPERSEDED
        if instr.scope.end_turn is not None and turn > instr.scope.end_turn:
            return LifecycleStatus.EXPIRED
        if any(e.new_status == LifecycleStatus.AMBIGUOUS for e in past):
            return LifecycleStatus.AMBIGUOUS
        if instr.scope.kind == "conditional":
            return LifecycleStatus.CONDITIONAL
        if instr.scope.kind == "temporary":
            return LifecycleStatus.TEMPORARY
        return LifecycleStatus.ACTIVE

    def is_applicable(self, instruction_id: str, turn: int, task_tags: set[str]) -> bool:
        status = self.status_at(instruction_id, turn)
        if status == LifecycleStatus.CONDITIONAL:
            return self._instructions[instruction_id].scope.condition in task_tags
        return status in APPLICABLE

    def applicable_at(self, turn: int, task_tags: set[str]) -> list[InstructionRecord]:
        return [i for i in self if self.is_applicable(i.instruction_id, turn, task_tags)]

    def reinforceable_at(self, turn: int) -> list[InstructionRecord]:
        """Instructions a reminder may mention: given, and not superseded/expired/ambiguous.

        Conditional instructions are included so a policy checkpoint can restate them
        with their condition intact.
        """
        out = []
        for i in self:
            status = self.status_at(i.instruction_id, turn)
            if status is not None and status not in NEVER_REINFORCE:
                out.append(i)
        return out

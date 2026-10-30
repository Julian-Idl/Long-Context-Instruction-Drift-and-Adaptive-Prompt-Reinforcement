import pytest

from instruction_drift.registry import InstructionRegistry
from instruction_drift.schemas import (
    InstructionCategory,
    InstructionRecord,
    LifecycleEvent,
    LifecycleStatus,
    Scope,
    Verification,
)


def _instr(iid, kind="persistent", start=0, end=None, condition=None):
    return InstructionRecord(
        instruction_id=iid, group_id="G", conversation_id="C", original_text="x",
        normalized_text="x", compressed_text="x", category=InstructionCategory.FORMATTING,
        priority=0.5, scope=Scope(kind=kind, start_turn=start, end_turn=end, condition=condition),
        verification=Verification(method="m", evaluator_name="json_valid", evaluator_version="1.0"),
        created_at_turn=start, original_token_start=0,
    )


def _event(iid, turn, status=LifecycleStatus.SUPERSEDED):
    return LifecycleEvent(conversation_id="C", instruction_id=iid, turn_index=turn,
                          new_status=status, source_turn_id=f"T{turn}", reason="test")


def test_supersession_takes_effect_from_its_turn_only():
    reg = InstructionRegistry([_instr("A")], [_event("A", 10)])
    assert reg.status_at("A", 9) == LifecycleStatus.ACTIVE
    assert reg.status_at("A", 10) == LifecycleStatus.SUPERSEDED
    assert reg.applicable_at(9, set()) and not reg.applicable_at(10, set())
    assert reg.reinforceable_at(10) == []


def test_not_yet_given_temporary_and_expired():
    reg = InstructionRegistry([_instr("T", kind="temporary", start=3, end=6)])
    assert reg.status_at("T", 2) is None
    assert reg.status_at("T", 6) == LifecycleStatus.TEMPORARY
    assert reg.status_at("T", 7) == LifecycleStatus.EXPIRED
    assert not reg.is_applicable("T", 7, set())


def test_conditional_applies_only_when_condition_tag_present():
    reg = InstructionRegistry([_instr("K", kind="conditional", condition="code_requested")])
    assert reg.status_at("K", 1) == LifecycleStatus.CONDITIONAL
    assert not reg.is_applicable("K", 1, {"prose"})
    assert reg.is_applicable("K", 1, {"code_requested"})
    assert [i.instruction_id for i in reg.reinforceable_at(1)] == ["K"]


def test_ambiguous_is_excluded_and_bad_events_rejected():
    reg = InstructionRegistry([_instr("A")], [_event("A", 2, LifecycleStatus.AMBIGUOUS)])
    assert reg.status_at("A", 2) == LifecycleStatus.AMBIGUOUS
    assert not reg.is_applicable("A", 2, set()) and reg.reinforceable_at(2) == []
    with pytest.raises(ValueError):
        reg.add_event(_event("A", 3, LifecycleStatus.ACTIVE))
    with pytest.raises(KeyError):
        reg.add_event(_event("missing", 3))

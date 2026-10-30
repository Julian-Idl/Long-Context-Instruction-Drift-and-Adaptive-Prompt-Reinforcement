import random
from pathlib import Path

import pytest

from instruction_drift.config import DesignConfig
from instruction_drift.evaluation import EVALUATORS
from instruction_drift.generation import Corpus, ScenarioSpec, generate, validate_plan
from instruction_drift.schemas import ConflictKind, LifecycleStatus
from instruction_drift.templates import MODULE, STRUCT_TASKS, TEMPLATES
from instruction_drift.tokenization import reference_tokenizer

CORPUS = Path("data/corpus/wikitext103_test.jsonl")
TOK = reference_tokenizer()


class FakeCorpus(Corpus):
    """Deterministic stand-in so tests do not need the downloaded corpus."""

    def __init__(self):
        rng = random.Random(0)
        words = "the river city council market history archive winter bridge station".split()
        self.sha256 = "fake"
        self.paragraphs = [" ".join(rng.choice(words) for _ in range(120)) + "." for _ in range(50)]


def _spec(**over):
    base = dict(template_id="FMT_JSON", turn_count=30, instruction_position_ratio=0.3,
                distractor_density=0.4, conflict_intensity=0.2,
                conflict_kinds=[ConflictKind.UNAUTHORIZED_CONTRADICTION,
                                ConflictKind.UNTRUSTED_QUOTE],
                task_transition_count=2, distractor_tokens_per_exchange=250,
                checkpoint_interval=5, seed=42)
    base.update(over)
    return ScenarioSpec(**base)


def _gen(**over):
    return generate(_spec(**over), "C1", FakeCorpus(), TOK, "You are a helpful assistant.")


def test_generation_is_deterministic():
    assert _gen().model_dump_json() == _gen().model_dump_json()
    assert _gen().model_dump_json() != _gen(seed=43).model_dump_json()


@pytest.mark.parametrize("template_id", sorted(TEMPLATES))
def test_every_template_yields_valid_plan_on_target(template_id):
    plan = _gen(template_id=template_id)
    assert validate_plan(plan) == []
    a = plan.achieved
    assert abs(a.instruction_position_ratio - 0.3) <= 0.05
    assert abs(a.distractor_density - 0.4) <= 0.05
    assert a.task_transition_count == 2
    assert plan.lifecycle_events == []  # unauthorized/quoted conflicts never supersede


def test_token_accounting_matches_reference_tokenizer():
    plan = _gen()
    for ex in plan.exchanges:
        assert ex.user_tokens == TOK.count(ex.user_text)
        assert ex.reply_tokens == TOK.count(ex.scripted_reply)
    total = plan.system_tokens + sum(e.user_tokens + e.reply_tokens for e in plan.exchanges)
    assert total == plan.achieved.total_tokens
    j = plan.achieved.instruction_exchange
    before = plan.system_tokens + sum(e.user_tokens + e.reply_tokens for e in plan.exchanges[:j])
    assert plan.instructions[0].original_token_start == before
    distractor = sum(e.user_tokens + e.reply_tokens for e in plan.exchanges
                     if e.kind == "distractor")
    assert plan.achieved.distractor_density == pytest.approx(
        distractor / (total - plan.system_tokens))


def test_checkpoints_only_after_instruction_and_conflicts_are_checkpoints():
    plan = _gen()
    j = plan.achieved.instruction_exchange
    assert all(e.index > j for e in plan.exchanges if e.is_checkpoint)
    assert all(e.is_checkpoint for e in plan.exchanges if e.conflict_kind)
    assert all(e.kind != "distractor" for e in plan.exchanges if e.is_checkpoint)


def test_zero_density_has_no_distractors():
    plan = _gen(distractor_density=0.0)
    assert not any(e.kind == "distractor" for e in plan.exchanges)
    assert plan.achieved.distractor_density == 0.0


def test_authorized_update_supersedes_and_stops_scoring():
    plan = _gen(conflict_intensity=0.3, conflict_kinds=[ConflictKind.AUTHORIZED_UPDATE])
    assert validate_plan(plan) == []
    updates = [e for e in plan.exchanges if e.kind == "update"]
    assert len(updates) == 1  # only one authorized update is meaningful
    u = updates[0].index
    reg = plan.registry()
    iid = plan.instructions[0].instruction_id
    assert reg.status_at(iid, u - 1) == LifecycleStatus.ACTIVE
    assert reg.status_at(iid, u) == LifecycleStatus.SUPERSEDED
    assert all(e.violating_reply is None for e in plan.exchanges if e.index >= u)


def test_supersession_without_authorized_kind_is_rejected():
    plan = _gen(conflict_intensity=0.3, conflict_kinds=[ConflictKind.AUTHORIZED_UPDATE])
    forged = plan.model_copy(update={"spec": plan.spec.model_copy(
        update={"conflict_kinds": [ConflictKind.UNTRUSTED_QUOTE]})})
    assert any("supersession" in e for e in validate_plan(forged))


def test_validator_catches_scripted_violation():
    plan = _gen()
    j = plan.achieved.instruction_exchange
    exs = list(plan.exchanges)
    k = next(i for i, e in enumerate(exs) if e.index > j and e.kind == "task")
    exs[k] = exs[k].model_copy(update={"scripted_reply": "plain text, not JSON"})
    errors = validate_plan(plan.model_copy(update={"exchanges": exs}))
    assert any("scripted reply" in e for e in errors)


def test_violating_replies_are_detected_by_their_evaluator():
    # The mock relies on this: each template's violation must actually be a violation.
    for tid, tpl in TEMPLATES.items():
        for task in tpl.tasks:
            bad = tpl.render_violation(task.answer, task)
            good = tpl.render(task.answer, task)
            applicable = [s for s in tpl.instructions
                          if not s.condition or s.condition in task.tags]
            if not applicable:
                continue
            for s in applicable:
                assert EVALUATORS[s.evaluator].fn(good, s.params).compliance is not False, (
                    tid, task.task_id)
            assert any(EVALUATORS[s.evaluator].fn(bad, s.params).compliance is False
                       for s in applicable), (tid, task.task_id)


def test_struct_task_solutions_differ_from_module():
    assert all(t.code != MODULE for t in STRUCT_TASKS)


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not downloaded")
def test_real_corpus_design_mostly_valid():
    corpus = Corpus(CORPUS)
    design = DesignConfig(templates=sorted(TEMPLATES), turn_counts=[30],
                          position_ratios=[0.1, 0.5, 0.9], distractor_densities=[0.0, 0.4, 0.8])
    from instruction_drift.generation import expand_design
    specs = expand_design(design, 1)
    valid = sum(not validate_plan(generate(s, f"C{i}", corpus, TOK, "x"), design)
                for i, s in enumerate(specs))
    assert valid / len(specs) >= 0.9

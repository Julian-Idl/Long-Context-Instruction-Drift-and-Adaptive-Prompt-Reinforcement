import json

import pandas as pd
import pytest

from instruction_drift.config import (
    B0,
    DesignConfig,
    ExperimentConfig,
    PolicyConfig,
    ProviderConfig,
)
from instruction_drift.dataset import rows_from_run
from instruction_drift.reinforcement import (
    HEADER,
    ReinforcementState,
    group_members,
    make_policy,
    render,
)
from instruction_drift.runner import build_plans, run_experiment
from instruction_drift.schemas import Action, ConflictKind, LifecycleStatus
from instruction_drift.storage import RunStore
from instruction_drift.tokenization import reference_tokenizer

TOK = reference_tokenizer()
B1 = PolicyConfig(name="B1_every_turn", kind="fixed_interval", interval=1)
B3 = PolicyConfig(name="B3_every_10", kind="fixed_interval", interval=10)
B4 = PolicyConfig(name="B4_summary_5", kind="fixed_interval", interval=5,
                  action=Action.POLICY_CHECKPOINT)
B5 = PolicyConfig(name="B5_rule", kind="threshold", single_threshold=0.0)


def _cfg(root, policies, templates=("MULTI_LEN_ASSUME", "COND_CODE", "FMT_JSON"), **design):
    d = dict(templates=list(templates), turn_counts=[20], position_ratios=[0.1],
             distractor_densities=[0.0], corpus_path=root / "none.jsonl")
    d.update(design)
    return ExperimentConfig(name="rf", seed=5, output_root=root, policies=policies,
                            provider=ProviderConfig(kind="mock", model_id="mock"),
                            design=DesignConfig(**d))


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    root = tmp_path_factory.mktemp("plans")
    return {p.spec.template_id: p for p in build_plans(_cfg(root, [B0]))[0]}


@pytest.fixture(scope="module")
def updated_plans(tmp_path_factory):
    root = tmp_path_factory.mktemp("updated")
    return build_plans(_cfg(root, [B0], conflict_intensities=[0.2],
                            conflict_kinds=[ConflictKind.AUTHORIZED_UPDATE]))[0]


@pytest.fixture(scope="module")
def matched_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("matched")
    store, summary = run_experiment(_cfg(root, [B0, B1, B3, B4, B5]))
    return store, summary


def test_render_restates_whole_groups_and_conditions(plans):
    multi = plans["MULTI_LEN_ASSUME"]
    reg = multi.registry()
    instrs, groups = list(reg), group_members(reg)
    full = render(Action.SELECTIVE_REINFORCEMENT, instrs[:1], groups)
    assert full.startswith(HEADER)
    assert all(i.original_text in full for i in instrs)  # partial group restated whole
    short = render(Action.COMPRESSED_REINFORCEMENT, instrs, groups)
    assert all(i.compressed_text in short for i in instrs)
    cond = plans["COND_CODE"].registry()
    cp = render(Action.POLICY_CHECKPOINT, list(cond), group_members(cond))
    assert "applies whenever code is requested" in cp
    assert render(Action.NO_ACTION, instrs, groups) is None


def test_superseded_instruction_is_never_restated(updated_plans):
    plan = next(p for p in updated_plans if any(
        e.new_status == LifecycleStatus.SUPERSEDED for e in p.lifecycle_events))
    reg = plan.registry()
    ev = next(e for e in plan.lifecycle_events if e.new_status == LifecycleStatus.SUPERSEDED)
    old = reg.get(ev.instruction_id)
    pol = make_policy(B1, TOK)
    d = pol.decide(k=ev.turn_index + 1, is_checkpoint=False, registry=reg, candidates=[],
                   risks=None, state=ReinforcementState(), instruction_exchange=0)
    assert ev.instruction_id not in d.instruction_ids
    assert d.text is None or old.original_text not in d.text


def test_fixed_interval_schedule(plans):
    plan = plans["FMT_JSON"]
    reg, j = plan.registry(), min(i.created_at_turn for i in plan.instructions)
    pol, state = make_policy(B3, TOK), ReinforcementState()
    acted = [k for k in range(len(plan.exchanges))
             if pol.decide(k=k, is_checkpoint=False, registry=reg, candidates=[], risks=None,
                           state=state, instruction_exchange=j).text]
    assert acted == [k for k in range(j + 10, len(plan.exchanges), 10)]


def test_threshold_bands_cooldown_and_budget(plans):
    plan = plans["MULTI_LEN_ASSUME"]
    reg = plan.registry()
    cands = list(reg)
    ids = [i.instruction_id for i in cands]
    cfg = PolicyConfig(name="P", kind="threshold", cooldown_exchanges=3)
    pol = make_policy(cfg, TOK)

    def decide(k, p, state):
        return pol.decide(k=k, is_checkpoint=True, registry=reg, candidates=cands,
                          risks=dict.fromkeys(ids, p), state=state, instruction_exchange=0)

    assert decide(5, 0.1, ReinforcementState()).action == Action.NO_ACTION
    assert decide(5, 0.5, ReinforcementState()).action == Action.COMPRESSED_REINFORCEMENT
    assert decide(5, 0.7, ReinforcementState()).action == Action.SELECTIVE_REINFORCEMENT
    st = ReinforcementState()
    assert decide(5, 0.9, st).action == Action.FULL_REINFORCEMENT
    cooled = decide(7, 0.9, st)  # within cooldown
    assert cooled.action == Action.NO_ACTION and cooled.cooldown_applied == set(ids)
    assert decide(9, 0.9, st).action == Action.FULL_REINFORCEMENT
    # Not a checkpoint: a threshold policy never acts.
    assert pol.decide(k=5, is_checkpoint=False, registry=reg, candidates=cands,
                      risks=dict.fromkeys(ids, 0.9), state=ReinforcementState(),
                      instruction_exchange=0).action == Action.NO_ACTION

    groups = group_members(reg)
    full = TOK.count(render(Action.FULL_REINFORCEMENT, cands, groups))
    short = TOK.count(render(Action.COMPRESSED_REINFORCEMENT, cands, groups))
    tight = make_policy(cfg.model_copy(update={"max_reinforcement_tokens": short}), TOK)
    st = ReinforcementState()
    d = tight.decide(k=5, is_checkpoint=True, registry=reg, candidates=cands,
                     risks=dict.fromkeys(ids, 0.9), state=st, instruction_exchange=0)
    assert short < full and d.action == Action.COMPRESSED_REINFORCEMENT
    d2 = tight.decide(k=20, is_checkpoint=True, registry=reg, candidates=cands,
                      risks=dict.fromkeys(ids, 0.9), state=st, instruction_exchange=0)
    assert d2.action == Action.NO_ACTION and st.tokens_inserted == short


def test_bad_policies_are_rejected():
    with pytest.raises(ValueError, match="interval"):
        make_policy(PolicyConfig(name="x", kind="fixed_interval"), TOK)
    with pytest.raises(ValueError, match="model_path"):
        make_policy(PolicyConfig(name="x", kind="threshold", risk_source="model"), TOK)


def test_matched_run_logs_every_policy_on_the_same_plans(matched_run):
    store, summary = matched_run
    pp = summary["per_policy"]
    assert set(pp) == {B0.name, B1.name, B3.name, B4.name, B5.name}
    assert pp[B0.name]["reminders"] == 0 and pp[B1.name]["reminders"] > pp[B3.name]["reminders"]
    for name in (B1.name, B3.name, B4.name):
        assert pp[name]["reminder_tokens_billed"] >= pp[name]["reminder_tokens_inserted"] > 0
    rd = store.run_dir
    read = lambda s: [json.loads(x) for x in open(rd / s, encoding="utf-8")]  # noqa: E731
    cps = pd.DataFrame(read("raw/checkpoints.jsonl"))
    per = cps.groupby("policy_name")["conversation_id"].apply(lambda s: sorted(set(s)))
    assert all(v == per[B0.name] for v in per)  # matched
    assert cps["checkpoint_id"].is_unique and cps["response_id"].is_unique
    prompts = pd.DataFrame(read("raw/prompts.jsonl"))
    assert (prompts.loc[prompts.policy_name == B0.name, "reinforcement_token_count"] == 0).all()
    b1 = prompts[prompts.policy_name == B1.name]
    assert (b1.final_token_count > b1.original_token_count).all()
    assert (b1.reinforcement_role == "user").all()
    turns = pd.DataFrame(read("raw/conversations.jsonl"))
    reminders = turns[turns.content_type == "reinforcement"]
    assert set(reminders.policy_name) == {B1.name, B3.name, B4.name, B5.name}
    instr = read("raw/instructions.jsonl")
    assert len(instr) == len({i["instruction_id"] for i in instr})  # logged once per plan


def test_threshold_decisions_record_causal_risk(matched_run):
    store, _ = matched_run
    dec = pd.DataFrame([json.loads(x) for x in open(store.run_dir / "processed/decisions.jsonl",
                                                    encoding="utf-8")])
    b5 = dec[dec.policy_name == B5.name]
    assert len(b5) and b5.predicted_failure_probability.between(0, 1).all()
    assert b5.checkpoint_id.str.contains("_CP").all()
    b1 = dec[dec.policy_name == B1.name]
    assert b1.predicted_failure_probability.isna().all()
    assert (b1.selected_action == Action.FULL_REINFORCEMENT).all()


def test_dataset_uses_only_the_b0_arm(matched_run):
    store, summary = matched_run
    df = rows_from_run(store.run_dir)
    assert set(df.policy_name) == {B0.name}
    assert len(df) == summary["per_policy"][B0.name]["scored"]


def test_context_check_budgets_for_reminders(tmp_path):
    base = _cfg(tmp_path, [B0], templates=("FMT_JSON",), turn_counts=[60])
    ctx = {"context_window": 6000, "max_output_tokens": 50}
    plain = base.model_copy(update={"generation": base.generation.model_copy(update=ctx)})
    noisy = plain.model_copy(update={"policies": [B0, B1]})
    assert len(build_plans(plain)[0]) == 1
    excluded = build_plans(noisy)[1]
    assert len(excluded) == 1 and "context window" in excluded[0].errors[0]


def test_run_store_reads_policy_runs(matched_run):
    store, _ = matched_run
    assert RunStore.open(store.run_dir).manifest()["is_simulated"] is True

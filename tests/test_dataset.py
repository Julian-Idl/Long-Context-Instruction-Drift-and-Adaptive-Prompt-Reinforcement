import json
import math

import pandas as pd
import pytest

from instruction_drift.config import DesignConfig, ExperimentConfig, ProviderConfig
from instruction_drift.dataset import (
    LABEL,
    assign_splits,
    build_dataset,
    check_split_integrity,
    rows_from_run,
    validate_dataset,
)
from instruction_drift.features import FEATURE_COLUMNS, InstructionHistory, extract_features
from instruction_drift.generation import ConversationPlan
from instruction_drift.runner import run_experiment
from instruction_drift.schemas import TurnRecord
from instruction_drift.templates import TEMPLATES


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    cfg = ExperimentConfig(
        name="ds", seed=11, output_root=root,
        provider=ProviderConfig(kind="mock", model_id="mock"),
        design=DesignConfig(templates=sorted(TEMPLATES), turn_counts=[30],
                            position_ratios=[0.1, 0.3], distractor_densities=[0.0],
                            conflict_intensities=[0.0, 0.2], replicates=2,
                            corpus_path=root / "none.jsonl"))
    store, _ = run_experiment(cfg)
    return store.run_dir


def _read(run_dir, stream, model):
    from instruction_drift.storage import RunStore
    return list(RunStore.open(run_dir).read(stream, model))


def test_rows_have_labels_features_and_meta(mock_run):
    df = rows_from_run(mock_run)
    assert len(df) > 50 and set(FEATURE_COLUMNS) <= set(df.columns)
    assert df[LABEL].isin([0, 1]).all() and 0 < df[LABEL].mean() < 1
    assert validate_dataset(df) == []
    assert not any(c.startswith("meta_") for c in FEATURE_COLUMNS)


def test_features_ignore_the_response_and_everything_after(mock_run):
    """Rewriting the reply at k, all later turns, and all later labels leaves row k unchanged."""
    plan = _read(mock_run, "plans", ConversationPlan)[0]
    turns = _read(mock_run, "conversations", TurnRecord)
    turns = sorted([t for t in turns if t.conversation_id == plan.conversation_id],
                   key=lambda t: (t.turn_index, t.role != "user"))
    k = [e.index for e in plan.exchanges if e.is_checkpoint][1]
    instr = plan.instructions[0]
    prefix = [t for t in turns if t.turn_index < k or (t.turn_index == k and t.role == "user")]
    hist = InstructionHistory(outcomes=[(k - 3, 1)])
    base = extract_features(plan, k, prefix, instr, hist, 1)

    tampered_plan = plan.model_copy(update={"exchanges": [
        e if e.index <= k else e.model_copy(update={"kind": "distractor", "user_tokens": 9999,
                                                    "conflict_kind": None})
        for e in plan.exchanges]})
    again = extract_features(tampered_plan, k, prefix, instr, hist, 1)
    assert json.dumps(base, default=str) == json.dumps(again, default=str)

    future_reply = [t for t in turns if t.turn_index == k and t.role == "assistant"]
    with pytest.raises(ValueError):
        extract_features(plan, k, prefix + future_reply, instr, hist, 1)
    with pytest.raises(ValueError):
        extract_features(plan, k, prefix, instr, InstructionHistory(outcomes=[(k, 0)]), 1)
    with pytest.raises(ValueError):
        extract_features(plan, k, prefix, instr,
                         InstructionHistory(reinforcements=[(k, None)]), 1)


def test_history_features_use_only_earlier_checkpoints(mock_run):
    df = rows_from_run(mock_run).sort_values(["instruction_id", "exchange_index"])
    for _, g in df.groupby("instruction_id"):
        labels = (1 - g[LABEL]).tolist()
        for i, (_, row) in enumerate(g.iterrows()):
            past = labels[:i]
            assert row["prior_checkpoint_count"] == len(past)
            assert row["prior_failure_count"] == past.count(0)
            if past:
                assert row["prior_compliance_rate"] == pytest.approx(sum(past) / len(past))
            else:
                assert math.isnan(row["prior_compliance_rate"])


def test_history_streaks_and_recency(mock_run):
    plan = _read(mock_run, "plans", ConversationPlan)[0]
    turns = sorted([t for t in _read(mock_run, "conversations", TurnRecord)
                    if t.conversation_id == plan.conversation_id],
                   key=lambda t: (t.turn_index, t.role != "user"))
    k = [e.index for e in plan.exchanges if e.is_checkpoint][-1]
    prefix = [t for t in turns if t.turn_index < k or (t.turn_index == k and t.role == "user")]
    instr = plan.instructions[0]
    f = extract_features(plan, k, prefix, instr,
                         InstructionHistory(outcomes=[(k - 9, 1), (k - 6, 0), (k - 3, 0)]), 1)
    assert (f["consecutive_failure_count"], f["consecutive_success_count"]) == (2, 0)
    assert f["turns_since_last_failure"] == 3 and f["prior_failure_count"] == 2
    f = extract_features(plan, k, prefix, instr,
                         InstructionHistory(outcomes=[(k - 6, 0), (k - 3, 1)]), 1)
    assert (f["consecutive_failure_count"], f["consecutive_success_count"]) == (0, 1)


def _toy(n_groups=20, per_group=5, templates=("A", "B")):
    rows = []
    for t in templates:
        for g in range(n_groups):
            for r in range(per_group):
                rows.append({"template_id": t, "group_key": f"{t}:{g}",
                             "conversation_id": f"{t}{g}", LABEL: r % 2})
    return pd.DataFrame(rows)


def test_group_split_has_no_leakage_and_is_deterministic():
    df = _toy()
    s1 = assign_splits(df, seed=3)
    assert check_split_integrity(df, s1) == []
    assert (s1 == assign_splits(df, seed=3)).all()
    assert not (s1 == assign_splits(df, seed=4)).all()
    for t in ("A", "B"):
        assert set(s1[df["template_id"] == t]) == {"train", "validation", "test"}


def test_template_held_out_split():
    df = _toy(templates=("A", "B", "C"))
    s = assign_splits(df, seed=1, strategy="template", test_templates=["C"])
    assert set(s[df["template_id"] == "C"]) == {"test"}
    assert "test" not in set(s[df["template_id"] != "C"])
    assert check_split_integrity(df, s) == []


def test_split_integrity_detects_leak():
    df = _toy()
    s = assign_splits(df, seed=0)
    first = df.index[df["group_key"] == df["group_key"].iloc[0]]
    s.loc[first[0]] = "train"
    s.loc[first[1]] = "test"
    assert check_split_integrity(df, s)


def test_same_plan_in_two_runs_is_a_duplicate(mock_run):
    df = rows_from_run(mock_run)
    rerun = df.assign(experiment_id="another_run")
    assert any("duplicate" in e for e in validate_dataset(pd.concat([df, rerun])))


def test_missing_model_version_is_rejected(mock_run):
    df = rows_from_run(mock_run)
    df.loc[df.index[0], "model_version"] = None
    assert any("without a model version" in e for e in validate_dataset(df))


def test_dataset_refuses_mock_real_mix():
    df = pd.DataFrame({"experiment_id": ["a", "b"], "checkpoint_id": ["c", "c"],
                       "instruction_id": ["i", "i"], "is_simulated": [True, False],
                       "policy_name": ["B0", "B0"],
                       "model_version": ["m", "m"], LABEL: [0, 1],
                       "feature_schema_version": ["1.0", "1.0"],
                       **{c: [0.0, 0.0] for c in FEATURE_COLUMNS}})
    assert any("mixes simulated" in e for e in validate_dataset(df))


def test_build_dataset_writes_parquet_and_manifest(mock_run, tmp_path):
    out = tmp_path / "ds"
    m = build_dataset([mock_run], out, seed=0)
    assert m["is_simulated"] is True and sum(m["rows"].values()) > 0
    parts = [pd.read_parquet(out / f"{s}.parquet") for s in ("train", "validation", "test")]
    assert sum(len(p) for p in parts) == sum(m["rows"].values())
    groups = [set(p["group_key"]) for p in parts]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    with pytest.raises(FileExistsError):
        build_dataset([mock_run], out, seed=0)

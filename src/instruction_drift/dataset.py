"""Checkpoint-level prediction dataset and leakage-resistant splits.

Rows are rebuilt from run logs by replaying each conversation in exchange order and
calling `extract_features` with only the pre-response prefix. Labels are joined after
features are computed: `next_response_violation = 1 - compliance`.

Split unit: the conversation plan (`group_key` = template + plan seed), so all
checkpoints of one conversation, including reruns of the same plan under other policies
or models, land in the same split. A template-held-out split is available for
generalization experiments (E8).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from instruction_drift.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    InstructionHistory,
    extract_features,
)
from instruction_drift.generation import ConversationPlan
from instruction_drift.registry import InstructionRegistry
from instruction_drift.schemas import (
    B0_POLICY,
    CheckpointRecord,
    ComplianceRecord,
    ResponseRecord,
    TurnRecord,
    utc_now,
)
from instruction_drift.storage import RunStore

LABEL = "next_response_violation"
SPLITS = ("train", "validation", "test")


def group_key(plan: ConversationPlan) -> str:
    return f"{plan.spec.template_id}:{plan.spec.seed}"


def rows_from_run(run_dir: Path) -> pd.DataFrame:
    store = RunStore.open(Path(run_dir))
    manifest = store.manifest()
    plans = {p.conversation_id: p for p in store.read("plans", ConversationPlan)}
    turns: dict[str, list[TurnRecord]] = defaultdict(list)
    # Training rows come from the no-reinforcement arm only: other arms' prefixes contain
    # reminders the B0-trained features never saw.
    for t in store.read("conversations", TurnRecord):
        if t.policy_name == B0_POLICY:
            turns[t.conversation_id].append(t)
    labels: dict[tuple[str, str], ComplianceRecord] = {
        (c.checkpoint_id, c.instruction_id): c for c in store.read("compliance", ComplianceRecord)}
    responses = {r.checkpoint_id: r for r in store.read("responses", ResponseRecord)}
    checkpoints: dict[str, list[CheckpointRecord]] = defaultdict(list)
    for c in store.read("checkpoints", CheckpointRecord):
        if c.policy_name == B0_POLICY:
            checkpoints[c.conversation_id].append(c)

    rows = []
    for cid, cps in checkpoints.items():
        plan = plans[cid]
        reg = InstructionRegistry(plan.instructions, plan.lifecycle_events)
        group_sizes = defaultdict(int)
        for i in plan.instructions:
            group_sizes[i.group_id] += 1
        conv_turns = sorted(turns[cid], key=lambda t: (t.turn_index, t.role != "user"))
        history = {i.instruction_id: InstructionHistory() for i in plan.instructions}
        for cp in sorted(cps, key=lambda c: c.exchange_index):
            k = cp.exchange_index
            prefix = [t for t in conv_turns if t.turn_index < k or
                      (t.turn_index == k and t.role == "user")]
            resp = responses[cp.checkpoint_id]
            for instr in plan.instructions:
                lab = labels.get((cp.checkpoint_id, instr.instruction_id))
                if lab is None or not lab.applicable or lab.compliance is None:
                    continue
                assert reg.is_applicable(instr.instruction_id, k, set(cp.task_tags))
                feats = extract_features(plan, k, prefix, instr, history[instr.instruction_id],
                                         group_sizes[instr.group_id])
                rows.append({
                    "experiment_id": store.run_id,
                    "conversation_id": cid,
                    "group_key": group_key(plan),
                    "template_id": plan.spec.template_id,
                    "checkpoint_id": cp.checkpoint_id,
                    "exchange_index": k,
                    "instruction_id": instr.instruction_id,
                    "instruction_category": instr.category.value,
                    "policy_name": cp.policy_name,
                    "provider": resp.provider,
                    "model_id": resp.model_id,
                    "model_version": resp.model_version,
                    "is_simulated": manifest["is_simulated"],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    **feats,
                    LABEL: 1 - lab.compliance,
                    "label_evaluator_type": lab.evaluator_type.value,
                    "label_evaluator_name": lab.evaluator_name,
                    "label_evaluator_version": lab.evaluator_version,
                    "meta_planned_turn_count": plan.spec.turn_count,
                    "meta_planned_position_ratio": plan.spec.instruction_position_ratio,
                    "meta_planned_distractor_density": plan.spec.distractor_density,
                    "meta_planned_conflict_intensity": plan.spec.conflict_intensity,
                    "meta_planned_total_tokens": plan.achieved.total_tokens,
                })
            # Only after every row at k is built does the outcome at k enter the history.
            for instr in plan.instructions:
                lab = labels.get((cp.checkpoint_id, instr.instruction_id))
                if lab is not None and lab.applicable and lab.compliance is not None:
                    history[instr.instruction_id].outcomes.append((k, lab.compliance))
    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame, allow_mixed_models: bool = False) -> list[str]:
    errors = []
    if df.empty:
        return ["dataset is empty"]
    # No run id in the key: the same plan rerun in another run (e.g. a resumed Kaggle
    # range) is a duplicate observation, not new data.
    key = ["policy_name", "model_version", "checkpoint_id", "instruction_id"]
    if df.duplicated(key).any():
        errors.append(f"{int(df.duplicated(key).sum())} duplicate (policy, model, checkpoint, "
                      "instruction) rows, possibly the same plan in two runs")
    if df["model_version"].isna().any():
        errors.append(f"{int(df['model_version'].isna().sum())} rows without a model version")
    if df["is_simulated"].nunique() > 1:
        errors.append("dataset mixes simulated (mock) and real-provider rows")
    if not allow_mixed_models and df["model_version"].nunique() > 1:
        errors.append(f"dataset mixes model versions: {sorted(df['model_version'].unique())}")
    if not df[LABEL].isin([0, 1]).all():
        errors.append("label must be 0/1")
    if df["feature_schema_version"].nunique() > 1:
        errors.append("mixed feature schema versions")
    if df[[c for c in FEATURE_COLUMNS if c not in _NULLABLE]].isna().any().any():
        errors.append("unexpected missing values in non-nullable features")
    return errors


# Undefined until the event first happens; models must handle NaN explicitly.
_NULLABLE = {"prior_compliance_rate", "turns_since_last_failure",
             "turns_since_last_reinforcement", "prior_reinforcement_success_rate"}


def assign_splits(df: pd.DataFrame, *, seed: int, fractions=(0.6, 0.2, 0.2),
                  strategy: str = "conversation",
                  test_templates: list[str] | None = None) -> pd.Series:
    """Map each row to a split by its group, never by individual row.

    conversation: groups are shuffled within each template so every split has every
      template (stratified group split).
    template: whole templates in `test_templates` form the test split; remaining
      groups are split into train/validation as above.
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = random.Random(seed)
    groups = df.groupby("template_id")["group_key"].unique()
    assignment: dict[str, str] = {}
    held = set(test_templates or [])
    if strategy == "template" and not held:
        raise ValueError("template strategy needs test_templates")
    for template, keys in sorted(groups.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        if strategy == "template" and template in held:
            assignment.update({k: "test" for k in keys})
            continue
        if strategy == "template":
            tr = fractions[0] / (fractions[0] + fractions[1])
            bounds = [round(tr * len(keys)), len(keys)]
            names = ["train", "validation"]
        else:
            c1 = round(fractions[0] * len(keys))
            c2 = round((fractions[0] + fractions[1]) * len(keys))
            bounds, names = [c1, c2, len(keys)], list(SPLITS)
        start = 0
        for end, name in zip(bounds, names, strict=True):
            assignment.update({k: name for k in keys[start:end]})
            start = end
    return df["group_key"].map(assignment)


def check_split_integrity(df: pd.DataFrame, split: pd.Series) -> list[str]:
    errors = []
    for col in ("group_key", "conversation_id"):
        per = df.assign(split=split).groupby(col)["split"].nunique()
        if (per > 1).any():
            errors.append(f"{int((per > 1).sum())} {col} values appear in more than one split")
    if split.isna().any():
        errors.append("rows without a split")
    return errors


def build_dataset(run_dirs: list[Path], out_dir: Path, *, seed: int,
                  strategy: str = "conversation", test_templates: list[str] | None = None,
                  allow_mixed_models: bool = False) -> dict:
    df = pd.concat([rows_from_run(Path(r)) for r in run_dirs], ignore_index=True)
    errors = validate_dataset(df, allow_mixed_models)
    if errors:
        raise ValueError("; ".join(errors))
    split = assign_splits(df, seed=seed, strategy=strategy, test_templates=test_templates)
    errors = check_split_integrity(df, split)
    if errors:
        raise ValueError("; ".join(errors))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    df = df.assign(split=split)
    paths = {}
    for name in SPLITS:
        part = df[df["split"] == name]
        if len(part):
            p = out_dir / f"{name}.parquet"
            part.to_parquet(p, index=False)
            paths[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "created_at": utc_now(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "label": LABEL,
        "source_runs": [str(r) for r in run_dirs],
        "split_strategy": strategy,
        "test_templates": test_templates,
        "split_seed": seed,
        "is_simulated": bool(df["is_simulated"].iloc[0]),
        "model_versions": sorted(df["model_version"].astype(str).unique()),
        "rows": {n: int((df["split"] == n).sum()) for n in SPLITS},
        "positives": {n: int(df.loc[df["split"] == n, LABEL].sum()) for n in SPLITS},
        "groups": {n: int(df.loc[df["split"] == n, "group_key"].nunique()) for n in SPLITS},
        "sha256": paths,
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                   encoding="utf-8")
    return manifest

import json
import logging

import pytest
from pydantic import ValidationError

from instruction_drift.cli import main
from instruction_drift.config import ConfigError, ExperimentConfig, load_config
from instruction_drift.logs import setup_logging
from instruction_drift.schemas import (
    ComplianceRecord,
    EvaluatorType,
    InstructionCategory,
    InstructionRecord,
    Scope,
    Verification,
)
from instruction_drift.storage import RunStore


def _cfg(tmp_path, kind="mock"):
    return ExperimentConfig.model_validate(
        {
            "name": "t",
            "seed": 1,
            "output_root": str(tmp_path),
            "provider": {"kind": kind, "model_id": "m"},
        }
    )


def _instruction(**over):
    base = dict(
        instruction_id="I0001",
        group_id="IG0001",
        conversation_id="C0001",
        original_text="Do not use external libraries.",
        normalized_text="Do not import third-party libraries.",
        compressed_text="Stdlib only.",
        category=InstructionCategory.NEGATIVE_CONSTRAINT,
        priority=1.0,
        scope=Scope(kind="persistent", start_turn=0),
        verification=Verification(
            method="python_ast", evaluator_name="import_checker", evaluator_version="1.0"
        ),
        created_at_turn=0,
        original_token_start=0,
    )
    base.update(over)
    return InstructionRecord(**base)


def test_record_roundtrip_and_extra_fields_rejected():
    rec = _instruction()
    assert InstructionRecord.model_validate_json(rec.model_dump_json()) == rec
    with pytest.raises(ValidationError):
        InstructionRecord.model_validate({**rec.model_dump(), "leaked_label": 1})
    with pytest.raises(ValidationError):
        _instruction(priority=1.5)


def test_config_loads_and_rejects_secrets(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("name: x\nseed: 1\nprovider: {kind: mock, model_id: m}\n", encoding="utf-8")
    assert load_config(good).provider.kind == "mock"

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\nseed: 1\nprovider: {kind: ollama, model_id: m, api_key: sk-123}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(bad)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("name: x\nseed: 1\nprovider: {kind: gpt, model_id: m}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(unknown)


def test_run_store_is_append_only_and_separates_mock(tmp_path):
    cfg = _cfg(tmp_path)
    store = RunStore.create(cfg, run_id="R1")
    assert store.run_dir == tmp_path / "mock" / "R1"
    assert store.manifest()["is_simulated"] is True

    with pytest.raises(FileExistsError):
        RunStore.create(cfg, run_id="R1")

    rec = ComplianceRecord(
        run_id="R1", conversation_id="C1", checkpoint_id="CP1", instruction_id="I1",
        response_id="RESP1", policy_name="B0", applicable=True, compliance=0,
        evaluator_type=EvaluatorType.DETERMINISTIC, evaluator_name="e",
        evaluator_version="1.0", evidence="x",
    )
    store.append("compliance", rec)
    RunStore.open(store.run_dir).append("compliance", rec)
    assert list(store.read("compliance", ComplianceRecord)) == [rec, rec]

    real = RunStore.create(_cfg(tmp_path, kind="ollama"), run_id="R1")
    assert real.run_dir == tmp_path / "ollama" / "R1"
    assert real.manifest()["is_simulated"] is False


def test_json_logging_writes_structured_lines(tmp_path):
    logger = setup_logging(tmp_path)
    logger.info("hello", extra={"run_id": "R1"})
    for h in logger.handlers:
        h.flush()
    line = json.loads((tmp_path / "run.log").read_text(encoding="utf-8").splitlines()[0])
    assert line["msg"] == "hello" and line["run_id"] == "R1" and line["level"] == "INFO"
    setup_logging(None, logging.WARNING)  # release the file handle


def test_cli_config_validate(tmp_path):
    good = tmp_path / "c.yaml"
    good.write_text("name: x\nseed: 1\nprovider: {kind: mock, model_id: m}\n", encoding="utf-8")
    assert main(["config", "validate", "--config", str(good)]) == 0
    good.write_text("name: x\n", encoding="utf-8")
    assert main(["config", "validate", "--config", str(good)]) == 1

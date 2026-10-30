import json

from instruction_drift.config import DesignConfig, ExperimentConfig, ProviderConfig
from instruction_drift.llm import MockProvider
from instruction_drift.runner import run_experiment
from instruction_drift.schemas import (
    CheckpointRecord,
    ComplianceRecord,
    PromptRecord,
    ResponseRecord,
)
from instruction_drift.templates import TEMPLATES


def _cfg(tmp_path):
    return ExperimentConfig(
        name="e2e", seed=5, output_root=tmp_path,
        provider=ProviderConfig(kind="mock", model_id="mock"),
        design=DesignConfig(templates=sorted(TEMPLATES), turn_counts=[20],
                            position_ratios=[0.3], distractor_densities=[0.0],
                            conflict_intensities=[0.2], corpus_path=tmp_path / "none.jsonl"))


def test_mock_end_to_end_is_complete_consistent_and_reproducible(tmp_path):
    cfg = _cfg(tmp_path)
    store, summary = run_experiment(cfg)
    assert summary["is_simulated"] and summary["aborted"] == 0
    assert store.run_dir.parent == tmp_path / "mock"

    prompts = list(store.read("prompts", PromptRecord))
    responses = list(store.read("responses", ResponseRecord))
    checkpoints = list(store.read("checkpoints", CheckpointRecord))
    labels = list(store.read("compliance", ComplianceRecord))
    assert len(prompts) == len(responses) == len(checkpoints) == summary["checkpoints"] > 0
    assert {r.response_id for r in responses} == {c.response_id for c in checkpoints}
    assert {lab.response_id for lab in labels} <= {r.response_id for r in responses}
    assert all(lab.evaluator_version and lab.evaluator_name for lab in labels)

    # Leakage guard: the prompt at exchange k is system + k full exchanges + the current
    # user message, so it cannot contain the reply being predicted or anything later.
    for p in prompts:
        k = int(p.checkpoint_id.rsplit("CP", 1)[1])
        assert len(p.messages) == 2 + 2 * k and p.messages[-1]["role"] == "user"

    # The model's reply, not the scripted one, enters the history.
    turns = [json.loads(line) for line in open(store.path("conversations"), encoding="utf-8")]
    model_turns = [t for t in turns if t["role"] == "assistant" and not t["is_scripted"]]
    assert len(model_turns) == len(responses)

    store2, summary2 = run_experiment(cfg)
    assert store2.run_dir != store.run_dir
    assert summary2["per_template"] == summary["per_template"]


def test_failed_call_aborts_conversation_and_is_recorded(tmp_path):
    cfg = _cfg(tmp_path)
    class Failing(MockProvider):
        def _call(self, messages, params):
            from instruction_drift.llm import ProviderError
            raise ProviderError("boom")

    store, summary = run_experiment(cfg, provider=Failing(cfg.provider, {}, seed=0))
    assert summary["aborted"] == summary["conversations"]
    responses = list(store.read("responses", ResponseRecord))
    assert all(r.error == "boom" for r in responses)
    assert not list(store.read("compliance", ComplianceRecord))

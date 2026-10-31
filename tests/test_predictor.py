import json

import numpy as np
import pandas as pd
import pytest

from instruction_drift.config import DesignConfig, ExperimentConfig, ProviderConfig
from instruction_drift.dataset import LABEL, build_dataset
from instruction_drift.predictor import (
    MODELS,
    Calibrator,
    Predictor,
    choose_threshold,
    evaluate,
    expected_calibration_error,
    grouped_bootstrap_ci,
    load_split,
    metrics,
    train,
)
from instruction_drift.runner import run_experiment
from instruction_drift.templates import TEMPLATES


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Simulated data whose violation risk grows with context length (mock provider)."""
    root = tmp_path_factory.mktemp("pred")
    cfg = ExperimentConfig(
        name="pred", seed=21, output_root=root,
        provider=ProviderConfig(kind="mock", model_id="mock"),
        design=DesignConfig(templates=sorted(TEMPLATES), turn_counts=[20, 60],
                            position_ratios=[0.1, 0.4], distractor_densities=[0.0],
                            conflict_intensities=[0.0, 0.2], replicates=3,
                            corpus_path=root / "none.jsonl"))
    store, _ = run_experiment(cfg)
    ds = root / "ds"
    build_dataset([store.run_dir], ds, seed=0)
    models = root / "models"
    board = train(ds, models, seed=0, n_boot=50)
    return ds, models, board


def test_all_models_train_and_emit_valid_probabilities(trained):
    ds, models, board = trained
    assert set(board) == set(MODELS)
    va = load_split(ds, "validation")
    for name in MODELS:
        p = Predictor(models / name).predict_proba(va)
        assert p.shape == (len(va),) and np.all((p >= 0) & (p <= 1))
        meta = json.loads((models / name / "metadata.json").read_text())
        assert meta["is_simulated"] is True and meta["feature_schema_version"] == "1.0"


def test_learned_models_beat_majority_on_learnable_signal(trained):
    _, _, board = trained
    assert board["majority"]["roc_auc"] == pytest.approx(0.5)
    assert max(board[m]["roc_auc"] for m in ("logreg", "random_forest", "xgboost")) > 0.65


def test_saved_model_reproduces_predictions_and_output_contract(trained):
    ds, models, _ = trained
    va = load_split(ds, "validation")
    a = Predictor(models / "xgboost").predict_proba(va)
    b = Predictor(models / "xgboost").predict_proba(va)
    assert np.array_equal(a, b)
    rec = Predictor(models / "xgboost").predict_records(va.head(1))[0]
    assert set(rec) == {"instruction_id", "checkpoint_id", "p_failure_next_response",
                        "model_name", "model_version", "feature_schema_version"}
    assert (models / "xgboost" / "importance.csv").exists()


def test_test_split_access_is_logged(trained):
    ds, models, _ = trained
    log = models / "logreg" / "test_access_log.jsonl"
    assert not log.exists()
    m = evaluate(models / "logreg", ds, "validation", n_boot=20)
    assert not log.exists() and "per_category" in m
    evaluate(models / "logreg", ds, "test", n_boot=20)
    assert len(log.read_text().splitlines()) == 1


def test_train_refuses_existing_dir_and_single_class(trained, tmp_path):
    ds, models, _ = trained
    with pytest.raises(FileExistsError):
        train(ds, models, models=["majority"], n_boot=10)
    one = tmp_path / "one"
    one.mkdir()
    (one / "dataset_manifest.json").write_text((ds / "dataset_manifest.json").read_text())
    for s in ("train", "validation"):
        load_split(ds, s).assign(**{LABEL: 0}).to_parquet(one / f"{s}.parquet")
    with pytest.raises(ValueError, match="both classes"):
        train(one, tmp_path / "m", models=["logreg"], n_boot=10)


def test_metrics_and_calibration_units():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = metrics(y, p, 0.5)
    assert m["roc_auc"] == pytest.approx(0.75) and m["precision"] == 1.0 and m["recall"] == 0.5
    assert m["false_positive_intervention_rate"] == 0.0
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0
    assert metrics(np.array([0, 0]), np.array([0.2, 0.3]), 0.5)["roc_auc"] is None
    assert choose_threshold([0, 0], [0.1, 0.2]) == (0.5, choose_threshold([0, 0], [0.1])[1])
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 2000)
    yy = (rng.uniform(0, 1, 2000) < raw ** 2).astype(int)  # miscalibrated scores
    for method in ("sigmoid", "isotonic"):
        cal = Calibrator(method).fit(raw, yy)
        assert expected_calibration_error(yy, cal.transform(raw)) < \
            expected_calibration_error(yy, raw)
    assert Calibrator("sigmoid").fit(raw, np.zeros(2000)).method == "none"


def test_grouped_bootstrap_resamples_groups():
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    ci = grouped_bootstrap_ci(y, p, np.repeat(np.arange(10), 10), n_boot=100)
    lo, hi = ci["brier"]
    assert lo <= hi and ci["roc_auc"] is not None
    one_group = grouped_bootstrap_ci(y, p, np.zeros(100), n_boot=20)
    assert one_group["brier"][0] == pytest.approx(one_group["brier"][1])
    assert isinstance(pd.Series(p).mean(), float)


def test_predictor_refuses_other_library_versions(trained, tmp_path):
    import shutil
    _, models, _ = trained
    dst = tmp_path / "m"
    shutil.copytree(models / "logreg", dst)
    meta = json.loads((dst / "metadata.json").read_text())
    meta["library_versions"]["scikit-learn"] = "0.0.1"
    (dst / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="library versions"):
        Predictor(dst)

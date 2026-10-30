"""Next-response failure predictor: baselines, models, calibration, metrics.

Protocol (docs/MODEL_AND_CONTROLLER.md §3):
- fit on train only; calibrate and choose the operating threshold on validation only;
- test is evaluated only on explicit request, and every access is logged;
- CIs come from a bootstrap over conversation groups, never over individual rows,
  because checkpoints within one conversation are correlated.

Output contract: `p_failure_next_response` per (checkpoint, instruction) row.
"""

from __future__ import annotations

import json
import math
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from instruction_drift.dataset import LABEL
from instruction_drift.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from instruction_drift.schemas import utc_now

PREDICTOR_VERSION = "1.0"
MODELS = ("majority", "prior_rate", "rule", "logreg", "random_forest", "xgboost")
EPS = 1e-6


def load_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(Path(dataset_dir) / f"{split}.parquet")


def _X(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].astype(float)


class Majority:
    """Predicts the training base rate for every row."""

    def fit(self, X, y):
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate)


class PriorRate:
    """The instruction's own failure rate so far; training base rate when there is none."""

    def fit(self, X, y):
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return (1 - X["prior_compliance_rate"]).fillna(self.rate).to_numpy()


class RuleScore:
    """Fixed, unfitted heuristic, defined before seeing any data:
    0.35 * conflict in the current turn + 0.25 * failed at the previous checkpoint
    + 0.25 * min(1, tokens since instruction / 8000) + 0.15 * distractor token ratio.
    """

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return (0.35 * X["current_turn_has_conflict"]
                + 0.25 * (X["consecutive_failure_count"] > 0)
                + 0.25 * np.minimum(1.0, X["tokens_since_instruction"] / 8000)
                + 0.15 * X["distractor_token_ratio"]).to_numpy()


class SklearnModel:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)[:, 1]


def make_model(name: str, seed: int):
    if name == "majority":
        return Majority()
    if name == "prior_rate":
        return PriorRate()
    if name == "rule":
        return RuleScore()
    impute = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if name == "logreg":
        return SklearnModel(make_pipeline(impute, StandardScaler(),
                                          LogisticRegression(max_iter=5000)))
    if name == "random_forest":
        return SklearnModel(make_pipeline(impute, RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)))
    if name == "xgboost":
        from xgboost import XGBClassifier
        # Native NaN handling: no imputation.
        return SklearnModel(XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=4, eval_metric="logloss"))
    raise ValueError(name)


class Calibrator:
    """Maps raw scores to probabilities; fitted on validation only."""

    def __init__(self, method: str):
        if method not in ("none", "sigmoid", "isotonic"):
            raise ValueError(method)
        self.method = method
        self.note = None

    def fit(self, p, y):
        if self.method != "none" and len(set(y)) < 2:
            self.note = "validation has one class; calibration skipped"
            self.method = "none"
        if self.method == "sigmoid":
            self.m = LogisticRegression().fit(_logit(p)[:, None], y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "sigmoid":
            return self.m.predict_proba(_logit(p)[:, None])[:, 1]
        if self.method == "isotonic":
            return self.m.predict(p)
        return np.clip(p, 0, 1)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def choose_threshold(y, p) -> tuple[float, str | None]:
    """Threshold maximising F1 on validation; 0.5 when validation has no positives."""
    if sum(y) == 0:
        return 0.5, "validation has no positives; default threshold 0.5"
    candidates = np.unique(np.round(p, 6))
    best = max(candidates, key=lambda t: (f1_score(y, p >= t, zero_division=0), t))
    return float(best), None


def metrics(y, p, threshold: float, budget: float = 0.2) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = p >= threshold
    two = len(set(y)) == 2
    k = max(1, math.ceil(budget * len(p)))
    top = np.argsort(-p, kind="stable")[:k]
    return {
        "n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
        "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if two else None,
        "pr_auc": float(average_precision_score(y, p)) if two else None,
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        f"recall_at_{int(budget * 100)}pct_budget": float(y[top].sum() / y.sum()) if y.sum()
        else None,
        # Share of triggered interventions that were not needed.
        "false_positive_intervention_rate": float((pred & (y == 0)).sum() / pred.sum())
        if pred.sum() else None,
    }


def grouped_bootstrap_ci(y, p, groups, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile CIs, resampling whole conversation groups."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    stats = {"roc_auc": [], "pr_auc": [], "brier": []}
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
        yb, pb = y[idx], p[idx]
        stats["brier"].append(brier_score_loss(yb, pb))
        if len(set(yb)) == 2:
            stats["roc_auc"].append(roc_auc_score(yb, pb))
            stats["pr_auc"].append(average_precision_score(yb, pb))
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None)
            for k, v in stats.items()}


def importance(name: str, model, columns: list[str], X: pd.DataFrame) -> pd.DataFrame | None:
    """LR: standardized coefficients. RF: impurity importance. XGBoost: mean |SHAP|."""
    if name == "xgboost":
        from xgboost import DMatrix
        contribs = model.est.get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]
        return pd.DataFrame({"feature": columns,
                             "mean_abs_shap": np.abs(contribs).mean(axis=0)})
    if name in ("logreg", "random_forest"):
        imputer = model.est.steps[0][1]
        names = list(imputer.get_feature_names_out(columns))
        last = model.est.steps[-1][1]
        vals = last.coef_[0] if name == "logreg" else last.feature_importances_
        key = "coefficient" if name == "logreg" else "impurity_importance"
        return pd.DataFrame({"feature": names, key: vals})
    return None


def _versions() -> dict:
    out = {}
    for pkg in ("scikit-learn", "xgboost", "numpy", "pandas"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def train(dataset_dir: Path, out_dir: Path, *, models=MODELS, seed: int = 0,
          calibration: str = "sigmoid", n_boot: int = 1000) -> dict:
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    ds_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if ds_manifest["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("dataset feature schema does not match this code")
    tr, va = load_split(dataset_dir, "train"), load_split(dataset_dir, "validation")
    if tr[LABEL].nunique() < 2:
        raise ValueError("training split needs both classes")
    out_dir.mkdir(parents=True, exist_ok=False)
    board = {}
    for name in models:
        model = make_model(name, seed).fit(_X(tr), tr[LABEL].to_numpy())
        cal = Calibrator(calibration if name not in ("majority",) else "none")
        p_va_raw = model.predict_proba(_X(va))
        cal.fit(p_va_raw, va[LABEL].to_numpy())
        p_va = cal.transform(p_va_raw)
        threshold, t_note = choose_threshold(va[LABEL].to_numpy(), p_va)
        m_dir = out_dir / name
        m_dir.mkdir()
        joblib.dump({"model": model, "calibrator": cal, "threshold": threshold,
                     "feature_columns": FEATURE_COLUMNS}, m_dir / "model.joblib")
        imp = importance(name, model, FEATURE_COLUMNS, _X(tr))
        if imp is not None:
            imp.sort_values(imp.columns[1], key=abs, ascending=False).to_csv(
                m_dir / "importance.csv", index=False)
        val = metrics(va[LABEL], p_va, threshold)
        val["ci95"] = grouped_bootstrap_ci(va[LABEL], p_va, va["group_key"], n_boot=n_boot,
                                           seed=seed)
        meta = {
            "model_name": name, "model_version": f"{name}_{PREDICTOR_VERSION}",
            "predictor_version": PREDICTOR_VERSION, "created_at": utc_now(), "seed": seed,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": FEATURE_COLUMNS, "dataset_dir": str(dataset_dir),
            "dataset_sha256": ds_manifest["sha256"],
            "is_simulated": ds_manifest["is_simulated"],
            "model_versions_in_data": ds_manifest["model_versions"],
            "calibration": cal.method, "calibration_note": cal.note,
            "threshold": threshold, "threshold_note": t_note,
            "params": getattr(getattr(model, "est", None), "get_params", lambda: {})(),
            "library_versions": _versions(),
            # Calibration and threshold were fitted on validation: these are optimistic.
            "validation_metrics": val,
        }
        (m_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                             encoding="utf-8")
        board[name] = {k: val[k] for k in ("roc_auc", "pr_auc", "brier", "ece", "f1")}
    (out_dir / "leaderboard_validation.json").write_text(json.dumps(board, indent=2),
                                                         encoding="utf-8")
    return board


class Predictor:
    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        art = joblib.load(self.dir / "model.joblib")
        self.model, self.cal, self.threshold = art["model"], art["calibrator"], art["threshold"]
        self.meta = json.loads((self.dir / "metadata.json").read_text(encoding="utf-8"))
        if art["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("model was trained on a different feature list")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.cal.transform(self.model.predict_proba(_X(df)))

    def predict_records(self, df: pd.DataFrame) -> list[dict]:
        p = self.predict_proba(df)
        return [{"instruction_id": r.instruction_id, "checkpoint_id": r.checkpoint_id,
                 "p_failure_next_response": float(pi), "model_name": self.meta["model_name"],
                 "model_version": self.meta["model_version"],
                 "feature_schema_version": self.meta["feature_schema_version"]}
                for r, pi in zip(df.itertuples(), p, strict=True)]


def evaluate(model_dir: Path, dataset_dir: Path, split: str, *, n_boot: int = 1000) -> dict:
    pred = Predictor(model_dir)
    df = load_split(dataset_dir, split)
    if split == "test":
        # Each look at the test set is recorded, so repeated peeking is visible.
        with open(Path(model_dir) / "test_access_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"accessed_at": utc_now(), "dataset_dir": str(dataset_dir)}) + "\n")
    p = pred.predict_proba(df)
    m = metrics(df[LABEL], p, pred.threshold)
    m["ci95"] = grouped_bootstrap_ci(df[LABEL], p, df["group_key"], n_boot=n_boot)
    m["per_category"] = {
        c: metrics(g[LABEL], pred.predict_proba(g), pred.threshold)
        for c, g in df.groupby("instruction_category")}
    return m

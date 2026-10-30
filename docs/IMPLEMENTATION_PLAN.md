# Implementation Plan

## 1. Recommended development schedule

A ten-week research implementation is a planning target, not a guarantee.

### Week 1 — Literature and design
- Verify relevant benchmark papers and repositories.
- Define operational instruction categories.
- Finalize hypotheses and metrics.
- Design pilot conditions.
- Define provider and model budget.

### Week 2 — Registry and evaluation
- Implement instruction data models.
- Implement lifecycle management.
- Implement deterministic evaluators.
- Add unit tests.

### Weeks 3–4 — Generator and inference
- Implement controlled conversation templates.
- Add distractor and conflict injection.
- Implement token counting.
- Implement mock and real LLM adapters.
- Build experiment runner.
- Store raw data and manifests.

### Week 5 — Drift characterization
- Run pilot and initial controlled experiments.
- Validate evaluator accuracy.
- Generate compliance curves.
- Perform preliminary statistical analysis.

### Week 6 — Predictor
- Build leakage-safe feature dataset.
- Implement grouped splits.
- Train Logistic Regression, Random Forest, and XGBoost.
- Calibrate predictions.
- Evaluate and explain model.

### Week 7 — Threshold reinforcement
- Implement all reinforcement actions.
- Add token budget and cooldown.
- Compare fixed baselines with threshold controller.

### Week 8 — Contextual bandit (optional)
- Define state/action/reward.
- Implement controlled learning.
- Evaluate using valid experimental assignment.
- Compare with threshold policy.

### Week 9 — Final evaluation
- Run matched baseline comparisons.
- Conduct ablations.
- Run generalization experiments.
- Generate final statistical outputs.

### Week 10 — Paper and release
- Finalize plots and tables.
- Write methodology/results.
- Document limitations.
- Clean code and reproduce final runs.

## 2. Recommended technology

Core:
- Python 3.11.
- NumPy and Pandas.
- Pydantic or dataclasses.
- scikit-learn.
- XGBoost (optional if installation is problematic).
- Statsmodels and SciPy.
- Matplotlib and Plotly.
- Pytest.
- PyYAML.
- SQLite or JSONL/Parquet storage.

LLM integration:
- Provider SDKs behind a common interface.
- LiteLLM may be considered for multi-provider support.
- Local inference through vLLM or another compatible runtime where hardware allows.
- Mock provider for development and tests.

Optional:
- Sentence Transformers for semantic features.
- SHAP for explainability.
- MLflow for experiment tracking.
- Vowpal Wabbit or a small custom implementation for contextual bandits.

Avoid adding every listed tool automatically. Choose the smallest reliable stack.

## 3. Suggested repository structure

```text
long_context_instruction_drift/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── models.yaml
│   └── experiments/
├── data/
│   ├── raw/
│   ├── labels/
│   ├── processed/
│   └── features/
├── docs/
├── models/
├── notebooks/
├── scripts/
├── src/
│   └── instruction_drift/
│       ├── config/
│       ├── schemas/
│       ├── registry/
│       ├── generation/
│       ├── context/
│       ├── evaluation/
│       ├── llm/
│       ├── prediction/
│       ├── reinforcement/
│       ├── experiments/
│       ├── storage/
│       └── reporting/
├── tests/
├── artifacts/
└── results/
```

## 4. Required command-line workflows

Design a CLI with commands conceptually equivalent to:

```bash
# Validate configuration
python -m instruction_drift.cli config validate --config configs/default.yaml

# Generate controlled conversations
python -m instruction_drift.cli data generate --config configs/experiments/pilot.yaml

# Run experiments (mock provider)
python -m instruction_drift.cli experiment run --config configs/experiments/pilot.yaml --provider mock

# Run real-provider experiments only after credentials and budget are configured
python -m instruction_drift.cli experiment run --config configs/experiments/pilot.yaml --provider configured_provider

# Build prediction dataset
python -m instruction_drift.cli dataset build --run-id RUN_ID

# Train predictor
python -m instruction_drift.cli predictor train --config configs/default.yaml

# Evaluate predictor
python -m instruction_drift.cli predictor evaluate --model-path PATH

# Compare reinforcement policies
python -m instruction_drift.cli experiment compare-policies --config configs/experiments/policy_comparison.yaml

# Generate report
python -m instruction_drift.cli report generate --run-id RUN_ID
```

The exact CLI syntax may change, but commands must be documented and testable.

## 5. Configuration requirements

Use YAML configuration for:
- Provider and model.
- Generation parameters.
- Experiment design.
- Token budget.
- Instruction categories.
- Evaluation settings.
- Feature schema.
- Predictor parameters.
- Controller thresholds.
- Reinforcement cooldown.
- Random seeds.
- Output paths.

Never put API keys in YAML. Use environment variables.

## 6. Minimum tests

- Instruction lifecycle and supersession.
- Instruction extraction schema validation.
- Correct token position calculations.
- Distractor density calculation.
- Conflict authorization handling.
- Deterministic evaluator correctness.
- Feature extraction uses only pre-response data.
- Group split has no conversation/template leakage.
- Predictor output range and calibration interface.
- Controller action selection and cooldown.
- Token accounting.
- Prompt role/insertion position.
- Provider error handling.
- Mock end-to-end experiment.
- Reproducible run manifest.

## 7. Definition of each milestone

A milestone is complete only when:
- Code is implemented.
- Tests pass.
- CLI or documented entry point works.
- Data output is inspected.
- Documentation is updated.
- Limitations are recorded.

Do not move to large-scale experiments while the mock end-to-end pipeline is broken.

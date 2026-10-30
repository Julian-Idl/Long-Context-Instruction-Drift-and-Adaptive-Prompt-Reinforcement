# Drift Predictor and Adaptive Reinforcement Controller

## 1. Prediction target

For every active instruction `i` at checkpoint `t`, predict:

\[
p_{i,t} = P(Y_{i,t+1}=0 \mid X_{i,t})
\]

where:
- `Y = 1` means instruction is followed.
- `Y = 0` means instruction is violated.
- `X` contains only information available before the next response.

For implementation clarity, store the target as:

`next_response_violation = 1` for failure, `0` for compliance.

The model output should therefore be named `p_failure_next_response`.

## 2. Model development sequence

### Baselines

1. Majority-class predictor.
2. Prior instruction failure-rate predictor.
3. Simple rule-based risk score.
4. Logistic Regression.

### Main candidates

- Random Forest.
- XGBoost or LightGBM.
- Optional MLP.
- Optional temporal model only if data volume and evidence justify it.

Do not begin with a large neural model. Structured features are the primary initial representation.

## 3. Training and validation

- Group split by conversation ID and preferably template family.
- Ensure all checkpoints from one conversation/template group remain in one split.
- Use validation data for hyperparameter and threshold selection.
- Keep test data untouched until final evaluation.
- Consider time-based or model-held-out evaluation if appropriate.
- Handle class imbalance using training-only methods.
- Record all preprocessing and model parameters.
- Calibrate probabilities using validation data only (e.g. isotonic regression or Platt scaling, selected appropriately).

Metrics:
- Precision.
- Recall.
- F1.
- ROC-AUC.
- PR-AUC.
- Brier score.
- Expected calibration error or reliability diagram.
- Recall at a fixed intervention budget.
- False-positive intervention rate.

PR-AUC and calibration are important when failures are rare or intervention costs matter.

## 4. Predictor output contract

```json
{
  "instruction_id": "I0001",
  "checkpoint_id": "CP0001",
  "p_failure_next_response": 0.73,
  "model_name": "xgboost",
  "model_version": "drift_model_v1",
  "feature_schema_version": "1.0"
}
```

Illustrative only.

## 5. Reinforcement action set

Required actions:

### `NO_ACTION`
Send the conversation without an additional reminder.

### `FULL_REINFORCEMENT`
Repeat the original instruction text, preserving its meaning exactly.

### `COMPRESSED_REINFORCEMENT`
Produce a concise equivalent of the instruction. Compression must preserve its semantics. If equivalence cannot be verified, fall back to full reinforcement.

### `SELECTIVE_REINFORCEMENT`
Reinforce only selected high-risk/high-priority active instructions.

### `POLICY_CHECKPOINT`
Summarize the currently active instruction set in a compact, structured form. Exclude superseded and expired instructions. Preserve conditional scope and priority.

## 6. Initial threshold controller

Implement a configurable threshold policy. Example starting bands:

- `p < 0.20`: no action.
- `0.20 <= p < 0.40`: monitor.
- `0.40 <= p < 0.60`: compressed reinforcement candidate.
- `0.60 <= p < 0.80`: selective/priority reinforcement.
- `p >= 0.80`: full reinforcement or policy checkpoint.

These are initial heuristics only. Tune thresholds using validation data and report the chosen operating point.

The controller must also consider:
- Instruction priority.
- Lifecycle status.
- Remaining token budget.
- Cooldown since last reinforcement.
- Previous action effectiveness.
- Whether the instruction is already reinforced in the current prompt.

## 7. Optimization objective

A general objective is:

\[
\min_{\pi} \mathbb{E}\left[
\lambda_T C_{\text{tokens}}(\pi)
+\lambda_L C_{\text{latency}}(\pi)
+\lambda_F C_{\text{failure}}(\pi)
\right]
\]

subject to a target compliance requirement where appropriate:

\[
CR(\pi) \geq \tau
\]

where:
- `C_tokens` is added reinforcement token cost.
- `C_latency` is additional end-to-end latency.
- `C_failure` is a weighted instruction failure cost.
- `tau` is the target compliance level.
- `lambda` weights are selected and reported transparently.

Evaluate multiple cost settings or compliance targets rather than presenting one arbitrary weighting as universal.

## 8. Contextual bandit extension

Only implement after the threshold controller and baseline experiments are stable.

State/context may include:
- Predicted failure probability.
- Instruction category and priority.
- Context length and token distance.
- Distractor/conflict features.
- Historical compliance.
- Previous action and its outcome.
- Remaining reinforcement budget.

Actions are the five reinforcement actions above.

Reward can be defined as:

\[
R_t =
\alpha \cdot \text{compliance utility}
-\beta \cdot \text{added token cost}
-\gamma \cdot \text{latency cost}
\]

Define the reward before training and document normalization.

Important:
- A contextual bandit assumes a decision and observable reward at each decision point; delayed effects across later turns may violate a simple one-step formulation.
- If long-term carry-over effects are important, consider a sequential decision process only after the simpler model is understood.
- Avoid invalid offline policy evaluation. A policy cannot be reliably evaluated from logged data if actions were not randomized or action propensities were not recorded.
- Prefer controlled online experiments with matched conversation templates and randomized policy assignment.
- Include exploration limits and token budget constraints.
- Compare against the threshold controller.

## 9. Reinforcement text integrity

- Preserve the original meaning.
- Do not turn a conditional instruction into an unconditional one.
- Do not broaden a narrow constraint.
- Do not omit exceptions.
- Do not reinforce superseded instructions.
- Record the generated reinforcement text and method.
- Where feasible, use deterministic compression templates before an LLM-based compressor.
- Test compressed versions for semantic equivalence using human audit and task-specific checks.

## 10. Avoiding false confidence

A high predicted risk is not proof of future failure. A low predicted risk is not a guarantee of compliance.

Report calibration, confidence intervals, and false-positive/false-negative behaviour. The controller should have a safe fallback when model confidence or feature availability is inadequate.

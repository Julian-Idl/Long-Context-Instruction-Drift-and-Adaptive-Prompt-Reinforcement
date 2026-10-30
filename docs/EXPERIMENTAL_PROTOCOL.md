# Experimental Protocol

## 1. General design

Use controlled, reproducible experiments. Begin with a pilot to validate generation, token accounting, evaluator correctness, and API feasibility. Expand only after the pilot is stable.

## 2. Independent variables

Initial design factors:

### Conversation length
- 5 turns
- 10 turns
- 20 turns
- 35 turns
- 50 turns
- 75 turns
- 100 turns

### Instruction position
- 10%
- 30%
- 50%
- 70%
- 90%

Calculate position from tokenized context, not turn index alone.

### Distractor density
- 0%
- 20%
- 40%
- 60%
- 80%

Define the ratio as distractor tokens divided by total eligible context tokens. Document how the denominator is computed.

### Conflict intensity
Possible levels:
- 0%: no conflicts.
- 10%: low conflict exposure.
- 20%: moderate.
- 40%: high.

Define conflict intensity precisely, such as proportion of eligible turns containing a conflict scenario. Keep authorized updates separate from unauthorized contradictions and untrusted quoted instructions.

### Instruction category
Use the eight categories defined in `PROJECT_SPEC.md`.

### Task transitions
Vary the number and type of task changes independently where possible.

## 3. Do not run the full Cartesian product immediately

The full factor product is expensive. Use:
1. Pilot design.
2. Factorial or fractional-factorial design.
3. Analysis of main effects and selected interactions.
4. Targeted follow-up experiments.
5. Held-out generalization set.

Record the design rationale and all exclusions.

## 4. Dataset resources

Potential public resources to investigate:
- IFEval.
- VerIFY.
- LIFBench.
- MultiTurnInstruct.
- MultiChallenge.

Before use:
- Verify the exact paper and official repository.
- Check license and redistribution terms.
- Check whether the dataset actually contains the required multi-turn/context structure.
- Document any conversion or filtering.
- Do not assume that every benchmark named in planning material is directly downloadable or suitable without inspection.

Use benchmark tasks as supplementary sources or evaluation sets. Synthetic controlled conversations are needed to isolate individual experimental factors.

## 5. Conversation template design

Build templates for:
- Persistent formatting requirements.
- Language restrictions.
- Length limits.
- Negative constraints.
- Structural preservation.
- Behavioral requirements.
- Conditional instructions.
- Multiple simultaneous constraints.

For each template, define:
- Initial instruction.
- Valid follow-up tasks.
- Distractor content.
- Conflict scenarios.
- Expected applicability.
- Deterministic or semantic verification.
- Expected correct response properties.

## 6. Baselines

Run matched comparisons:

- B0: Original instructions only; no reinforcement.
- B1: Repeat all active instructions every turn.
- B2: Repeat all active instructions every 5 turns.
- B3: Repeat all active instructions every 10 turns.
- B4: Fixed-interval policy checkpoint summary.
- B5: Fixed risk-threshold controller.
- P1: Predictive adaptive controller.
- P2: Contextual-bandit controller, optional.

Ensure each method receives the same underlying task and conversation content. Record its actual prompt token count.

## 7. Primary metrics

### Instruction compliance rate

Unweighted:

\[
CR = \frac{\sum_{i,t}Y_{i,t}}{N}
\]

Weighted:

\[
WCR = \frac{\sum_{i,t}w_iY_{i,t}}{\sum_{i,t}w_i}
\]

Report per-instruction and per-category scores as well.

### Degradation slope

Estimate compliance change against:
- Token distance.
- Total context tokens.
- Turn count.

Do not assume a linear relationship; inspect curves and consider nonlinear or mixed-effects models.

### Prediction performance

- Precision, recall, F1.
- ROC-AUC.
- PR-AUC.
- Brier score.
- Calibration.
- Recall at fixed intervention budget.

### Reinforcement recovery

Measure compliance before and after reinforcement and persistence over subsequent turns. Use matched comparisons and account for repeated observations within conversations.

### Token overhead

Report:
- Added reinforcement tokens.
- Total input tokens.
- Output tokens.
- Evaluator tokens separately.
- Controller-related token usage separately.
- Provider-reported versus locally estimated counts.

### Latency

Report:
- Predictor computation latency.
- Controller latency.
- Target LLM latency.
- Evaluator latency.
- End-to-end latency.

### Compliance gain per added token

\[
E = \frac{CR_{\text{adaptive}}-CR_{\text{baseline}}}
{\text{additional reinforcement tokens}}
\]

Report this alongside absolute compliance and total cost.

## 8. Statistical analysis

Recommended:
- Confidence intervals using conversation/template-level bootstrap.
- Mixed-effects logistic regression for repeated binary compliance outcomes, with conversation/template and model effects where supported by the design.
- Paired comparisons for matched experimental conditions.
- Appropriate tests for model metric comparisons.
- Multiple-comparison correction when testing many hypotheses.
- Effect sizes, not p-values alone.

Do not treat each checkpoint row as independent. Observations within a conversation are correlated.

## 9. Leakage controls

- Split by conversation/template groups.
- Fit preprocessing only on training data.
- Tune thresholds on validation data.
- Calibrate on validation data.
- Do not use future compliance history.
- Do not use the response being predicted as an input feature.
- Do not use reinforcement action outcomes that occur after the checkpoint.
- Keep test results hidden until final evaluation.

## 10. Repeated runs and stochasticity

- Fix generation settings where possible.
- Record temperature, seed, and provider support for seed.
- Repeat selected conditions to estimate variability.
- Use matched prompts and paired analysis.
- Record model version and experiment date.
- Do not assume identical model identifiers imply identical snapshots over time.

## 11. Evaluator validation

For deterministic evaluators:
- Unit-test positive and negative examples.
- Test edge cases.
- Record evaluator version.

For semantic evaluators:
- Use a separate judge model where possible.
- Hide controller risk and action metadata from the judge.
- Use a consistent rubric.
- Human-audit a stratified sample.
- Report judge-human agreement and uncertainty.
- Keep judge token cost separate.

## 12. Required experiment groups

### E1 — Drift characterization
Vary context length, instruction position, turn count, and instruction type.

### E2 — Distractor and conflict sensitivity
Vary distractor density, conflict type, and task transitions.

### E3 — Predictor evaluation
Compare baseline classifiers and the primary predictor.

### E4 — Reinforcement comparison
Compare all fixed and adaptive policies.

### E5 — Recovery analysis
Measure immediate and sustained compliance after interventions.

### E6 — Cost-efficiency analysis
Compare compliance, tokens, and latency.

### E7 — Ablation
Remove historical, semantic, priority, calibration, or adaptive components one at a time.

### E8 — Generalization
Evaluate on unseen templates and, if feasible, a held-out target model.

## 13. Instruction compliance half-life

An optional exploratory metric is the token distance at which compliance falls to half its baseline level under a specified experimental condition.

Do not call it a universal property of a model. Estimate it only when the observed curve supports such a threshold. If compliance never reaches the threshold, report it as not estimable within the tested range.

Compare estimated values by instruction type and experimental condition, with uncertainty intervals.

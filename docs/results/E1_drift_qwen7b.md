# E1: drift characterization and predictor, qwen2.5:7b (B0, no reinforcement)

Real-model results. Not simulated.

## Run

| Item | Value |
|---|---|
| Config | `configs/experiments/e1_drift_kaggle.yaml`, seed 20261031 |
| Code | commit `c8a8786`, clean tree |
| Model | `qwen2.5:7b`, digest `845dbda0…697e`, 7.6B Q4_K_M, Ollama 0.40.13 |
| Generation | temperature 0, seed 1234, max output 400 tokens, context window 32768 |
| Hardware | Kaggle T4 x2, one Ollama server per GPU, one session, 5 h 43 min |
| Plans | 576 designed, 565 valid, 11 excluded by the context pre-check |
| Calls | 5956 checkpoints, 0 errors, 0 retries, 0 aborted conversations |
| Largest prompt | 23212 provider tokens (no truncation) |
| Runs | `e1_drift_qwen7b_20261030T191115Z_ea0db4`, `e1_drift_qwen7b_20261030T191116Z_927ed6` |

Labels: 6373 applicable instruction checks, 1681 violations (26.4%).

**Truncation check.** 433 replies (7.3%) stopped at the 400-token limit. No JSON reply was
truncated. Truncated replies in the word-limit template exceed 60 words regardless of
the cut, and truncated code replies fail python checks less often than complete ones
(0.11 vs 0.26). Truncation is therefore not a source of false violations here.

## Violation rate by template (all conversations)

| Template | Checks | Violations | Rate |
|---|---|---|---|
| FMT_JSON | 785 | 324 | 0.413 |
| LANG_PY | 719 | 237 | 0.330 |
| STRUCT_NAMES | 587 | 187 | 0.319 |
| BEHAV_ASSUME | 790 | 209 | 0.265 |
| COND_CODE | 384 | 97 | 0.253 |
| NEG_STDLIB | 742 | 168 | 0.226 |
| LEN_60 | 774 | 168 | 0.217 |
| MULTI_LEN_ASSUME | 1592 | 291 | 0.183 |

## Main finding: conflicts, not context length, drive violations

| Conflict intensity | Violation rate |
|---|---|
| 0.0 (no injected conflicts) | 0.018 (45 / 2505 checks) |
| 0.2 | 0.423 |

In conflict-free conversations, by current context size:

| Context tokens | Checks | Violations | Rate |
|---|---|---|---|
| up to 2k | 719 | 7 | 0.010 |
| 2k to 5k | 911 | 14 | 0.015 |
| 5k to 10k | 618 | 5 | 0.008 |
| 10k to 30k | 257 | 19 | 0.074 |

Conflict-free violations are concentrated in LANG_PY (33 of 45). Distractor density and
planned turn count show no consistent effect without conflicts. These are descriptive
counts on pooled splits, without a model or a significance test. The rise above 10k
tokens rests on 19 events and needs confirmation.

Interpretation limits: violations are output non-compliance. They do not show what the
model internally retained. The conflict effect measures susceptibility to in-context
contradictions (unauthorized requests and quoted untrusted text), which is a different
mechanism from forgetting over distance.

## Predictor (grouped split by conversation plan)

Rows: train 3803, validation 1416, test 1154 (339 / 114 / 112 conversation groups).
Fit on train; Platt calibration and the F1 threshold on validation. Test evaluated once
per model, logged in each model's `test_access_log.jsonl`. 95% CIs from 1000 bootstrap
resamples of conversation groups.

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier | ECE | Precision | Recall | Recall at 20% budget | FP intervention rate |
|---|---|---|---|---|---|---|---|---|
| majority | 0.500 | 0.279 [0.223, 0.338] | 0.202 | 0.031 | 0.28 | 1.00 | 0.25 | 0.72 |
| prior_rate | 0.802 [0.760, 0.837] | 0.631 [0.519, 0.710] | 0.159 | 0.079 | 0.65 | 0.70 | 0.49 | 0.35 |
| rule | 0.880 [0.853, 0.905] | 0.712 [0.638, 0.787] | 0.127 | 0.054 | 0.65 | 0.75 | 0.51 | 0.35 |
| logreg | 0.938 [0.920, 0.954] | 0.850 [0.794, 0.893] | 0.093 | 0.034 | 0.78 | 0.70 | 0.60 | 0.22 |
| random_forest | 0.962 [0.947, 0.975] | 0.909 [0.862, 0.944] | 0.068 | 0.021 | 0.84 | 0.84 | 0.66 | 0.16 |
| **xgboost** (primary) | **0.965 [0.952, 0.977]** | **0.917 [0.876, 0.946]** | 0.067 | 0.024 | 0.81 | 0.84 | 0.65 | 0.19 |

Test: 1154 rows, 322 positives, XGBoost threshold 0.401. Validation AUC was 0.961, so
model selection did not overfit validation.

Per category (XGBoost, test): AUC 0.95 to 0.98 for every category except language
(0.858, 99 rows, 29 positives).

**What the predictor learns.** Top features by mean |SHAP|: current user message
length, unauthorized conflict count, recent conflict count, prior compliance rate,
conflict in the current turn. On validation it predicts onset (previous check passed)
with AUC 0.972 and persistence (previous check failed) with 0.841. So it is not only
copying the last outcome. Most of its signal is conflict exposure, which is known
before the response by design. Within conflict-free conversations, its validation AUC
is 0.957 over just 566 rows and 8 positives, which is too few to support a claim.

## Consequences for E3

- Reinforcement can only reduce violations where they occur, which is mostly conflict
  conversations. Conflict-free arms mostly measure overhead.
- The E3 design and its primary comparisons stay as pre-registered.

## Limitations

- One model, one quantization, one seed per plan, temperature 0.
- Synthetic templates and scripted conflicts; the results describe this design.
- Proxy evaluators are documented in `docs/IMPLEMENTATION_NOTES.md` section 5. No
  semantic judge was used, so no human audit sample was required for these labels.
- Raw outputs live under `results/kaggle/results_1793387465/` (not committed, 233 MB).

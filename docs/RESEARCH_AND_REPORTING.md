# Research Questions, Reporting, and Paper Plan

## 1. Central research question

Can instruction-level compliance failure be predicted from measurable conversation features, and can adaptive reinforcement improve compliance while reducing token and latency overhead relative to fixed repetition strategies?

## 2. Research questions

- RQ1: How does instruction compliance vary with token distance, context length, and turn count?
- RQ2: How do instruction category, distractor density, conflicts, and task transitions relate to failure risk?
- RQ3: Can structured context and history features predict next-response instruction violation?
- RQ4: Does selective predictive reinforcement outperform fixed-interval repetition?
- RQ5: Which reinforcement action is most effective for different instruction types and risk conditions?
- RQ6: What is the compliance-versus-token/latency trade-off?
- RQ7: Do findings generalize to unseen templates and different target LLMs?

## 3. Proposed contribution claims

Claims must be supported by results. Potential contributions:

1. A controlled dataset and protocol for instruction-level compliance drift.
2. An empirical characterization of contextual factors associated with instruction violations.
3. A calibrated ML model predicting next-response instruction failure.
4. A risk-aware controller selecting among different reinforcement actions.
5. A cost-aware evaluation of compliance, token overhead, and latency.
6. An analysis of instruction-specific compliance half-life where estimable.

Do not claim novelty until the literature review establishes what is and is not already known.

## 4. Required plots

1. Compliance rate versus total context tokens.
2. Compliance rate versus turns.
3. Compliance by instruction category.
4. Compliance versus instruction position.
5. Compliance versus distractor density.
6. Compliance versus conflict intensity.
7. Predicted risk calibration/reliability plot.
8. ROC and precision-recall curves.
9. Feature importance and SHAP summary.
10. Compliance before/after reinforcement.
11. Sustained recovery over subsequent turns.
12. Compliance versus additional reinforcement tokens.
13. Compliance versus end-to-end latency.
14. Pareto frontier for compliance and token cost.
15. Ablation comparison.

Every plot must state sample size, model, experimental condition, and whether values are observed or illustrative.

## 5. Required tables

- Dataset composition.
- Target model and inference settings.
- Instruction categories and verification methods.
- Experimental factors and levels.
- Predictor metrics.
- Calibration metrics.
- Baseline policy comparison.
- Token and latency overhead.
- Recovery results.
- Ablation results.
- Generalization results.
- Statistical test/effect-size summary.

## 6. Suggested paper structure

1. Introduction.
2. Related Work.
3. Problem Formulation.
4. Proposed Framework.
5. Experimental Methodology.
6. Instruction Drift Characterization.
7. Predictive Model.
8. Adaptive Reinforcement Controller.
9. Results and Discussion.
10. Ablation and Generalization.
11. Limitations.
12. Conclusions.

## 7. Limitations to discuss

- Output violations do not prove internal forgetting.
- Evaluator quality affects measured compliance.
- Synthetic conversations may not represent natural usage.
- Provider model snapshots may change.
- Tokenization and role semantics differ by provider.
- Reinforcement can increase prompt length and latency.
- Compressed reminders may alter instruction meaning.
- A predictor trained on one model may not transfer to another.
- Contextual bandit results depend on action assignment and reward design.
- Some long-context APIs may truncate or transform conversation history.

## 8. Reproducibility checklist

For each reported result, preserve:
- Code commit/version.
- Config file.
- Random seeds.
- Dataset and template versions.
- Target model/provider identifiers.
- Inference parameters.
- Evaluator version.
- Predictor version.
- Controller version.
- Token accounting method.
- Raw response records.
- Statistical analysis script.
- Run manifest.

## 9. Literature review protocol

Investigate and verify:
- IFEval.
- VerIFY.
- LIFBench.
- MultiTurnInstruct.
- MultiChallenge.
- Long-context position sensitivity and instruction-following research.
- Prompt repetition and summarization approaches.
- Adaptive prompting and context management.
- Contextual bandits for adaptive intervention, if used.

For every paper, record:
- Exact title.
- Authors.
- Year.
- Venue/preprint status.
- DOI or official URL.
- Research question.
- Dataset.
- Method.
- Baselines.
- Findings.
- Limitations.
- How the present project differs.

Do not rely on paper names alone. Verify official papers and repositories before citing them.

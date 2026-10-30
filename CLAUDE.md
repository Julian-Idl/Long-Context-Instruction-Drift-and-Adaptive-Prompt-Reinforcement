# Claude Code Agent Instructions

You are implementing a research-grade Python framework for **Long-Context Instruction Drift and Adaptive Prompt Reinforcement**.

Read `README.md` and every document in `docs/` before making architectural decisions. These documents define the intended research scope and implementation requirements.

## 1. Your role

Act as a careful research software engineer. Build a modular, testable, reproducible experimental system. Prioritize correctness, traceability, and scientific validity over unnecessary complexity.

Do not immediately generate the entire repository in one unreviewed pass. Work in milestones, run tests after each milestone, and summarize changed files and test results.

## 2. Non-negotiable research requirements

1. The prediction target is **next-response failure for an individual active instruction**:
   `P(instruction i is violated in the next response | information available before that response)`.
2. Features must be computed only from information available before the response being predicted. No future response, evaluator label, or future conversation content may leak into features.
3. Split training, validation, and test data by conversation/template group, not randomly by individual checkpoint rows.
4. Store both per-instruction compliance and aggregate compliance. Never rely on aggregate scores alone.
5. Distinguish instruction violation from its possible causes. A violation does not prove internal forgetting.
6. Keep target LLM execution separate from the drift predictor and reinforcement controller.
7. Log exact model/provider identifiers, model versions where available, generation parameters, prompt version, token usage, and timestamps.
8. Preserve raw prompts and responses. Do not silently overwrite experiment outputs.
9. Use deterministic programmatic evaluators whenever feasible. LLM-as-judge may supplement, not replace, deterministic checks.
10. Never claim an experiment succeeded unless it was actually run and its output inspected.

## 3. Development sequence

Implement in this order:

### Milestone 0 — Repository inspection and plan
- Inspect existing files before modifying them.
- Identify the Python version and environment.
- Propose a concise implementation plan.
- Do not overwrite existing user files or environments.
- If a provider/API key is unavailable, implement a mock provider and a clearly documented adapter interface.

### Milestone 1 — Foundations
- Project configuration and dependency management.
- Typed configuration loading and validation.
- Data models and JSONL persistence.
- Structured logging and experiment IDs.
- Unit tests.

### Milestone 2 — Instruction registry and evaluators
- Instruction schema and lifecycle.
- Instruction extraction interface.
- Deterministic evaluators for supported constraints.
- Compliance records and evaluator tests.

### Milestone 3 — Conversation generation and LLM adapters
- Controlled conversation templates.
- Distractor/conflict injection.
- Tokenization abstraction.
- Mock provider and at least one real provider adapter if credentials/configuration are available.
- Reproducible experiment runner.

### Milestone 4 — Feature extraction and drift dataset
- Causal, pre-response feature extraction.
- Instruction-level checkpoint rows.
- Dataset validation and leakage-resistant splits.

### Milestone 5 — Predictor
- Logistic Regression baseline.
- Random Forest baseline.
- XGBoost primary model (if dependency can be installed).
- Calibration, metrics, and explainability.
- Persist model artifacts and metadata.

### Milestone 6 — Reinforcement
- No-op, full, compressed, selective/priority, and policy-checkpoint actions.
- Threshold policy.
- Token-budget and cooldown constraints.
- Matched baseline experiment runner.

### Milestone 7 — Optional contextual bandit
- Implement only after Milestones 1–6 pass.
- Start with a contextual bandit, not a DQN.
- Define action, reward, exploration, safety constraints, and offline evaluation limitations.
- Never claim causal policy improvement from an invalid offline replay.

### Milestone 8 — Reporting
- Reproducible plots, tables, statistical tests, and experiment summaries.
- Research limitations and run manifests.

## 4. Engineering standards

- Python 3.11 target unless repository constraints require otherwise.
- Use type hints for public functions and dataclasses or Pydantic models for structured records.
- Prefer small modules with clear responsibilities.
- Use `pathlib`, structured logging, and explicit configuration.
- Avoid global mutable state.
- Add tests for feature leakage, evaluator correctness, action selection, token accounting, and split integrity.
- Set random seeds where applicable and record them.
- Make external API calls configurable and rate-limited.
- Implement retries with bounded backoff; never retry indefinitely.
- Handle rate limits, timeouts, malformed outputs, and missing usage metadata.
- Never put secrets in source files, YAML, logs, or committed artifacts. Use environment variables or a local `.env` excluded from version control.
- Pin or bound dependencies in `pyproject.toml`; document any compatibility uncertainty.
- Do not add heavyweight dependencies without a concrete need.

## 5. Scientific integrity

- Use synthetic data for pipeline tests, not as evidence of real LLM behaviour.
- Clearly label simulated/mock results.
- Do not mix mock-provider results with real-provider results.
- Do not infer a model's internal state from output compliance.
- Record evaluator type and evaluator version for every label.
- Manually audit a stratified sample of semantic-judge labels.
- Report uncertainty and confidence intervals where appropriate.
- Report both absolute compliance and overhead. Efficiency metrics do not replace either.
- Avoid p-hacking: predefine primary comparisons and document exploratory analyses.

## 6. Required developer workflow

For each milestone:

1. State the milestone objective.
2. Inspect relevant code and specs.
3. Implement the smallest complete slice.
4. Add or update tests.
5. Run tests and linters available in the repository.
6. Fix failures before proceeding.
7. Update documentation.
8. Summarize files changed, commands run, test outcomes, and unresolved issues.

Ask the user before:
- Making paid API calls or launching expensive batch inference.
- Downloading very large models or datasets.
- Deleting or overwriting existing data.
- Changing the agreed research scope.
- Running experiments likely to incur substantial cost.

## 7. Initial task

Start with Milestone 0. Inspect the repository and environment, read the project specification, then return a concise plan. Do not start large-scale inference or create paid API usage without explicit approval.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.

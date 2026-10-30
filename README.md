# Long-Context Instruction Drift and Adaptive Prompt Reinforcement

**Repository specification for Claude Code**

This repository is the implementation specification for a research project that studies instruction-following degradation in long-context, multi-turn LLM conversations and develops a predictive, cost-aware reinforcement controller.

## Research objective

Build an experimental framework that:

1. Generates controlled multi-turn conversations with persistent instructions.
2. Measures instruction compliance at defined checkpoints.
3. Extracts context, instruction, conflict, and historical-compliance features.
4. Predicts the probability that each active instruction will be violated in the next response.
5. Selects whether and how to reinforce vulnerable instructions.
6. Compares adaptive reinforcement against fixed and non-adaptive baselines.
7. Produces reproducible datasets, metrics, plots, and research-ready reports.

The central research question is:

> Can instruction-level failure risk be predicted from measurable conversation features, and can selective, adaptive reinforcement maintain a target compliance rate with less token and latency overhead than fixed repetition strategies?

## Read this first

Claude Code must read these files in order:

1. [`CLAUDE.md`](CLAUDE.md) — agent operating instructions and non-negotiable constraints.
2. [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) — full project context and scope.
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components and data flow.
4. [`docs/PIPELINE.md`](docs/PIPELINE.md) — end-to-end implementation pipeline.
5. [`docs/DATA_AND_SCHEMAS.md`](docs/DATA_AND_SCHEMAS.md) — data model and file formats.
6. [`docs/MODEL_AND_CONTROLLER.md`](docs/MODEL_AND_CONTROLLER.md) — prediction and reinforcement policy.
7. [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md) — controlled experiments and baselines.
8. [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — staged development plan.
9. [`docs/RESEARCH_AND_REPORTING.md`](docs/RESEARCH_AND_REPORTING.md) — research questions, metrics, plots, and paper structure.

## Expected top-level repository

```text
long_context_instruction_drift/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── artifacts/
└── results/
```

## Setup and usage

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Local runs need [Ollama](https://ollama.com).

```bash
uv sync                                   # create .venv with pinned dependencies
uv run python scripts/fetch_corpus.py     # distractor corpus (WikiText-103 test, CC BY-SA)
uv run pytest                             # test suite (no network, no model)

# Plan and validate conversations without calling any model
uv run python -m instruction_drift.cli data generate --config configs/experiments/pilot.yaml

# End-to-end run on the simulated provider (results/runs/mock/, never evidence)
uv run python -m instruction_drift.cli experiment run --config configs/experiments/pilot.yaml --provider mock

# Real local run (ollama pull qwen2.5:7b first)
uv run python -m instruction_drift.cli experiment run --config configs/experiments/pilot.yaml
```

Hosted models use `provider.kind: openai_compat` with `base_url` and `api_key_env`; the
key goes in an environment variable (see `.env.example`), never in YAML.

Implementation decisions, definitions, and known limits: [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md).

## Important scope boundaries

- This is a research framework, not a production prompt-management product.
- Do not fine-tune or train the target LLM as part of the core project.
- Train only the auxiliary drift predictor and, optionally, a contextual-bandit policy.
- Begin with deterministic evaluators and a threshold controller. Add a contextual bandit only after the complete baseline pipeline works.
- Do not fabricate benchmark results, model capabilities, citations, or performance claims.
- Treat instruction violation as an observable outcome. Do not claim to have measured the model's internal memory or attention state.
- Preserve all raw model responses, prompts, model identifiers, inference parameters, token usage, timestamps, and evaluator outputs for reproducibility.

## Definition of done

The project is not complete merely because an ML classifier trains. The complete system must demonstrate:

- Controlled data generation.
- Validated compliance labels.
- Leakage-resistant predictor evaluation.
- Working reinforcement interventions.
- Matched baseline comparisons.
- Token and latency accounting.
- Statistical analysis and ablations.
- Reproducible results and documentation.

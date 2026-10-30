# Data Model, Schemas, and Storage

## 1. Storage principles

- Use JSONL for append-only raw events and experiment records.
- Use Parquet or CSV for derived feature matrices.
- Use SQLite initially for indexing and run metadata if needed.
- Preserve raw records separately from derived datasets.
- Include schema versions.
- Use stable IDs across records.
- Store timestamps in UTC ISO-8601.
- Never store API secrets.

## 2. Instruction record

Illustrative JSON:

```json
{
  "schema_version": "1.0",
  "instruction_id": "I0001",
  "group_id": "IG0001",
  "original_text": "Do not use external libraries.",
  "normalized_text": "Do not import or depend on third-party libraries.",
  "category": "negative_constraint",
  "priority": 1.0,
  "scope": {
    "kind": "persistent",
    "start_turn": 0,
    "end_turn": null,
    "condition": null
  },
  "verification": {
    "method": "python_ast",
    "evaluator_name": "import_checker",
    "evaluator_version": "1.0"
  },
  "status": "active",
  "created_at_turn": 0,
  "original_token_start": 0,
  "reinforcement_history": []
}
```

## 3. Conversation turn record

```json
{
  "schema_version": "1.0",
  "conversation_id": "C0001",
  "turn_id": "C0001_T0004",
  "turn_index": 4,
  "role": "user",
  "content": "Please refactor this function.",
  "token_count": 42,
  "content_type": "task",
  "is_distractor": false,
  "contains_conflict": false,
  "conflict_authorized": null,
  "source_template_id": "TEMPLATE_07"
}
```

`content_type` may include `initial_instruction`, `task`, `relevant_context`, `distractor`, `conflict`, `untrusted_quote`, or `assistant_response`.

## 4. Experiment manifest

```json
{
  "schema_version": "1.0",
  "experiment_id": "EXP_000001",
  "conversation_id": "C0001",
  "template_id": "TEMPLATE_07",
  "model_provider": "provider_name",
  "model_id": "exact_model_identifier",
  "model_version": null,
  "generation": {
    "temperature": 0.0,
    "max_output_tokens": 512,
    "seed": 1234
  },
  "scenario": {
    "turn_count": 30,
    "target_context_tokens": 12000,
    "instruction_position_ratio": 0.1,
    "distractor_density": 0.4,
    "conflict_intensity": 0.1,
    "task_transition_count": 4
  },
  "policy": {
    "name": "no_reinforcement",
    "version": "1.0"
  },
  "created_at": "2026-10-30T00:00:00Z"
}
```

These values are illustrative. Do not use them as observed experimental results.

## 5. Feature row

One row represents one instruction at one pre-response checkpoint.

Required identifiers:
- `experiment_id`
- `conversation_id`
- `template_id`
- `checkpoint_id`
- `instruction_id`
- `model_id`
- `feature_schema_version`

### Positional features

- `instruction_position_ratio`
- `tokens_since_instruction`
- `turns_since_instruction`
- `instruction_token_start`
- `current_context_tokens`
- `distance_to_current_task_tokens`

### Context features

- `turn_count`
- `mean_tokens_per_turn`
- `task_transition_count`
- `topic_count` (only if method is documented)
- `distractor_token_ratio`
- `repeated_content_ratio`
- `code_token_ratio`

### Instruction features

- `instruction_category`
- `instruction_priority`
- `is_negative_constraint`
- `is_conditional`
- `constraint_count`
- `instruction_complexity_score` (define and document how calculated)

### Conflict features

- `conflict_count`
- `conflict_token_ratio`
- `recent_conflict_count`
- `authorized_conflict_count`
- `unauthorized_conflict_count`
- `untrusted_instruction_like_content_count`

### Historical features

- `prior_compliance_rate`
- `prior_failure_count`
- `consecutive_success_count`
- `consecutive_failure_count`
- `turns_since_last_failure`
- `reinforcement_count`
- `turns_since_last_reinforcement`
- `prior_reinforcement_success_rate`

### Target label

- `next_response_violation` — 1 for violation, 0 for compliance.
- `label_evaluator_type`
- `label_evaluator_version`
- `label_confidence` where applicable.

Do not populate historical fields using events after the current checkpoint.

## 6. Compliance record

```json
{
  "schema_version": "1.0",
  "experiment_id": "EXP_000001",
  "checkpoint_id": "CP_000021",
  "instruction_id": "I0001",
  "response_id": "RESP_000021",
  "compliance": 0,
  "severity": 1.0,
  "failure_type": "third_party_import",
  "evaluator_type": "deterministic",
  "evaluator_name": "python_ast_import_checker",
  "evaluator_version": "1.0",
  "evidence": "Detected import numpy",
  "evaluated_at": "2026-10-30T00:00:00Z"
}
```

## 7. Controller decision record

```json
{
  "schema_version": "1.0",
  "experiment_id": "EXP_000001",
  "checkpoint_id": "CP_000021",
  "instruction_id": "I0001",
  "predicted_failure_probability": 0.82,
  "instruction_priority": 1.0,
  "selected_action": "COMPRESSED_REINFORCEMENT",
  "reason": "Risk threshold exceeded",
  "reinforcement_text": "Use only the Python standard library.",
  "reinforcement_token_count": 9,
  "cooldown_applied": false,
  "policy_name": "threshold_policy",
  "policy_version": "1.0"
}
```

The probability above is an illustrative example, not a real model output.

## 8. LLM response and usage record

Record:
- Exact response text.
- Input token usage reported by provider.
- Output token usage reported by provider.
- Locally estimated token count if provider usage is unavailable.
- Tokenizer name/version.
- End-to-end latency.
- Provider/model ID.
- Generation parameters.
- Request ID.
- Retry count.
- Error status.

Keep provider-reported usage distinct from local estimates.

## 9. Suggested storage layout

```text
data/
├── raw/
│   ├── conversations.jsonl
│   ├── instructions.jsonl
│   ├── prompts.jsonl
│   ├── responses.jsonl
│   └── provider_usage.jsonl
├── labels/
│   ├── compliance.jsonl
│   └── human_audit.jsonl
├── processed/
│   ├── experiment_manifest.jsonl
│   └── checkpoints.parquet
└── features/
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

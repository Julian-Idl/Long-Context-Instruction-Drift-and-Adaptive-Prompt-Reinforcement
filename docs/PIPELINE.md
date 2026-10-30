# End-to-End Implementation Pipeline

## 1. Offline preparation

### Step 1 — Define experiment configuration

Each experiment declares:
- Target model/provider.
- Instruction templates and categories.
- Conversation template IDs.
- Number of turns.
- Token budget/context target.
- Instruction placement.
- Distractor ratio.
- Conflict scenario and authorization status.
- Task transition schedule.
- Checkpoint schedule.
- Generation parameters.
- Evaluation methods.
- Reinforcement policy and configuration.
- Random seeds.

### Step 2 — Create instructions

Create atomic instructions with:
- Stable ID.
- Original text.
- Category.
- Priority.
- Scope.
- Verification method.
- Lifecycle rule.
- Expected compliance criteria.

### Step 3 — Generate conversation

Generate a sequence of turns using controlled templates. Preserve:
- Turn order.
- Speaker role.
- Text.
- Token count.
- Whether content is relevant or distractor.
- Whether it contains a conflict.
- Whether the conflict is authorized.
- Template and random seed.

### Step 4 — Validate scenario

Before sending to an LLM, verify:
- Intended instruction position.
- Intended turn count.
- Token count within target range.
- Distractor ratio within tolerance.
- Conflict labels are correct.
- No accidental instruction supersession.
- All active instructions have evaluation rules.

## 2. Online checkpoint execution

For each checkpoint:

### Step 5 — Build pre-response state

Construct the conversation prefix and current task. Do not include the response that will be generated.

### Step 6 — Resolve active instructions

Exclude superseded, expired, and currently inapplicable instructions.

### Step 7 — Extract features

Produce one feature record per active instruction. Include:
- Position and distance.
- Context length and turn count.
- Distractor and conflict variables.
- Instruction attributes.
- Prior compliance.
- Prior reinforcement outcomes.

### Step 8 — Predict risk

The predictor returns a calibrated probability for each active instruction:

`p_failure_next_response`.

If no trained predictor is available, use a declared baseline policy. Do not silently substitute heuristic risk for an ML prediction.

### Step 9 — Select reinforcement action

The controller selects among:
- `NO_ACTION`
- `FULL_REINFORCEMENT`
- `COMPRESSED_REINFORCEMENT`
- `SELECTIVE_REINFORCEMENT`
- `POLICY_CHECKPOINT`

Apply token budget and cooldown constraints.

### Step 10 — Assemble prompt

Insert the selected reinforcement into the configured role/location. Log:
- Original prompt token count.
- Reinforcement token count.
- Final prompt token count.
- Exact reinforcement text.
- Role and insertion position.

### Step 11 — Generate response

Call the target model and capture:
- Response text.
- Model/provider identifier.
- Generation settings.
- Input/output token usage.
- Latency.
- Provider request ID if available.
- Errors/retries.

### Step 12 — Evaluate compliance

Evaluate all instructions applicable to the response. Store one result per instruction.

### Step 13 — Update history

Update:
- Compliance history.
- Failure count.
- Consecutive successes/failures.
- Reinforcement action history.
- Estimated intervention effectiveness.
- Token and latency costs.

### Step 14 — Persist results

Append raw and derived records to versioned JSONL or database storage. Never overwrite raw runs.

## 3. Offline model development

### Step 15 — Build prediction dataset

Create checkpoint rows with features known before the response and a target label from the next response.

Target:

`y = 1` if instruction is violated in the next response, otherwise `0`.

### Step 16 — Split without leakage

Group by conversation template/conversation ID. Keep all checkpoints from a group in one split. If evaluating cross-model transfer, hold out a model or model family as a separate test.

### Step 17 — Train models

Train:
- Logistic Regression.
- Random Forest.
- XGBoost (primary candidate).

Compare with simple baselines such as prior failure rate and a rule-based risk score.

### Step 18 — Calibrate and evaluate

Evaluate discrimination, calibration, and operating-threshold performance. Select thresholds on validation data only.

### Step 19 — Train/assess controller

Start with a threshold controller. Only after baseline comparisons work, evaluate a contextual bandit using online experiments or a valid controlled interaction protocol.

## 4. Reporting pipeline

Generate:
- Dataset summary.
- Compliance decay plots.
- Feature association plots.
- Predictor metrics and calibration plots.
- Baseline comparison tables.
- Reinforcement recovery plots.
- Token/latency overhead plots.
- Ablation tables.
- Confidence intervals and statistical test results.
- Run manifest and limitations.

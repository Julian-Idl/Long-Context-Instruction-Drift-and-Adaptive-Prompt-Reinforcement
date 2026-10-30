"""Versioned record schemas. Every persisted record is one of these models.

Field names follow docs/DATA_AND_SCHEMAS.md. Labels (compliance) and features are
separate record types so a feature row can never carry its own label by accident.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
B0_POLICY = "B0_no_reinforcement"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = SCHEMA_VERSION


class InstructionCategory(StrEnum):
    FORMATTING = "formatting"
    LANGUAGE = "language"
    LENGTH = "length"
    NEGATIVE_CONSTRAINT = "negative_constraint"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CONDITIONAL = "conditional"
    MULTI_CONSTRAINT = "multi_constraint"


class LifecycleStatus(StrEnum):
    ACTIVE = "active"
    CONDITIONAL = "conditional"
    TEMPORARY = "temporary"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    AMBIGUOUS = "ambiguous"


class ContentType(StrEnum):
    INITIAL_INSTRUCTION = "initial_instruction"
    TASK = "task"
    RELEVANT_CONTEXT = "relevant_context"
    DISTRACTOR = "distractor"
    CONFLICT = "conflict"
    UNTRUSTED_QUOTE = "untrusted_quote"
    ASSISTANT_RESPONSE = "assistant_response"
    REINFORCEMENT = "reinforcement"


class ConflictKind(StrEnum):
    AUTHORIZED_UPDATE = "authorized_update"
    UNAUTHORIZED_CONTRADICTION = "unauthorized_contradiction"
    UNTRUSTED_QUOTE = "untrusted_quote"


class EvaluatorType(StrEnum):
    DETERMINISTIC = "deterministic"
    SEMANTIC_JUDGE = "semantic_judge"
    HUMAN = "human"


class Action(StrEnum):
    NO_ACTION = "NO_ACTION"
    FULL_REINFORCEMENT = "FULL_REINFORCEMENT"
    COMPRESSED_REINFORCEMENT = "COMPRESSED_REINFORCEMENT"
    SELECTIVE_REINFORCEMENT = "SELECTIVE_REINFORCEMENT"
    POLICY_CHECKPOINT = "POLICY_CHECKPOINT"


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str = Field(pattern="^(persistent|temporary|conditional)$")
    start_turn: int = Field(ge=0)
    end_turn: int | None = Field(default=None, ge=0)
    condition: str | None = None


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method: str
    evaluator_name: str
    evaluator_version: str
    params: dict = Field(default_factory=dict)


class InstructionRecord(Record):
    instruction_id: str
    group_id: str
    conversation_id: str
    original_text: str
    normalized_text: str
    compressed_text: str
    category: InstructionCategory
    priority: float = Field(ge=0.0, le=1.0)
    scope: Scope
    verification: Verification
    created_at_turn: int = Field(ge=0)
    original_token_start: int = Field(ge=0)


class LifecycleEvent(Record):
    """A status change caused by conversation content, e.g. an authorized update."""

    conversation_id: str
    instruction_id: str
    turn_index: int = Field(ge=0)
    new_status: LifecycleStatus
    source_turn_id: str
    reason: str


class TurnRecord(Record):
    conversation_id: str
    turn_id: str
    turn_index: int = Field(ge=0)
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str
    token_count: int = Field(ge=0)
    content_type: ContentType
    is_distractor: bool = False
    contains_conflict: bool = False
    conflict_kind: ConflictKind | None = None
    conflict_target_instruction_id: str | None = None
    is_checkpoint: bool = False
    is_scripted: bool = True
    source_template_id: str
    policy_name: str = B0_POLICY


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, gt=0)
    seed: int | None = 1234
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    context_window: int | None = Field(default=None, gt=0)


class Usage(BaseModel):
    """Provider-reported usage is kept apart from local estimates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    local_input_tokens: int | None = None
    local_output_tokens: int | None = None
    tokenizer: str | None = None


class PromptRecord(Record):
    run_id: str
    conversation_id: str
    checkpoint_id: str
    policy_name: str
    messages: list[dict]
    original_token_count: int = Field(ge=0)
    reinforcement_token_count: int = Field(ge=0)
    final_token_count: int = Field(ge=0)
    reinforcement_role: str | None = None
    reinforcement_insert_index: int | None = None
    created_at: str = Field(default_factory=utc_now)


class ResponseRecord(Record):
    run_id: str
    conversation_id: str
    checkpoint_id: str
    response_id: str
    policy_name: str
    provider: str
    model_id: str
    model_version: str | None
    generation: GenerationParams
    text: str
    usage: Usage
    latency_ms: float = Field(ge=0.0)
    request_id: str | None = None
    retry_count: int = Field(default=0, ge=0)
    error: str | None = None
    finish_reason: str | None = None
    created_at: str = Field(default_factory=utc_now)


class CheckpointRecord(Record):
    """One target-model call: the point where risk would be predicted and labels observed."""

    run_id: str
    conversation_id: str
    checkpoint_id: str
    exchange_index: int = Field(ge=0)
    policy_name: str
    task_id: str | None
    topic: str | None
    task_tags: list[str]
    conflict_kind: ConflictKind | None
    prompt_token_count: int = Field(ge=0)
    response_id: str
    error: str | None = None


class ExclusionRecord(Record):
    """A planned conversation that failed pre-flight validation and was not run."""

    conversation_id: str
    spec: dict
    errors: list[str]


class ComplianceRecord(Record):
    run_id: str
    conversation_id: str
    checkpoint_id: str
    instruction_id: str
    response_id: str
    policy_name: str
    applicable: bool
    compliance: int | None = Field(default=None, ge=0, le=1)
    severity: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_type: str | None = None
    evaluator_type: EvaluatorType
    evaluator_name: str
    evaluator_version: str
    evidence: str
    evaluated_at: str = Field(default_factory=utc_now)


class ControllerDecision(Record):
    run_id: str
    conversation_id: str
    exchange_index: int = Field(ge=0)
    checkpoint_id: str  # "<conversation>_X<exchange>" when not a checkpoint
    instruction_id: str
    predicted_failure_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    instruction_priority: float
    selected_action: Action
    reason: str
    reinforcement_text: str | None = None
    reinforcement_token_count: int = Field(default=0, ge=0)
    cooldown_applied: bool = False
    policy_name: str
    policy_version: str

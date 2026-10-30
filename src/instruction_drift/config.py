"""Typed experiment configuration loaded from YAML.

Secrets never live in YAML: any key that looks like a credential is rejected, and
providers read keys from the environment variable named in `api_key_env`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from instruction_drift.schemas import B0_POLICY, Action, ConflictKind, GenerationParams

_SECRET_KEY = re.compile(r"(api[_-]?key|secret|token|password)$", re.IGNORECASE)
# These names describe token counts, not credentials.
_ALLOWED_TOKEN_KEYS = {"max_output_tokens", "target_context_tokens", "max_reinforcement_tokens"}

ProviderKind = Literal["mock", "ollama", "openai_compat"]


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: ProviderKind
    model_id: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = Field(default=300.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    requests_per_minute: float | None = Field(default=None, gt=0)


class DesignConfig(BaseModel):
    """Factor levels; the run uses their product (optionally a seeded random subset)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    templates: list[str] = Field(min_length=1)
    turn_counts: list[int] = Field(min_length=1)
    position_ratios: list[float] = Field(min_length=1)
    distractor_densities: list[float] = Field(min_length=1)
    conflict_intensities: list[float] = [0.0]
    conflict_kinds: list[ConflictKind] = [
        ConflictKind.UNAUTHORIZED_CONTRADICTION, ConflictKind.UNTRUSTED_QUOTE]
    task_transition_counts: list[int] = [0]
    distractor_tokens_per_exchange: int = Field(default=300, gt=0)
    checkpoint_interval: int = Field(default=5, gt=0)
    replicates: int = Field(default=1, gt=0)
    max_conversations: int | None = Field(default=None, gt=0)
    corpus_path: Path = Path("data/corpus/wikitext103_test.jsonl")
    corpus_sha256: str | None = None  # set to pin the exact corpus across machines
    position_tolerance: float = 0.05
    density_tolerance: float = 0.05


class PolicyConfig(BaseModel):
    """A reinforcement policy. See docs/IMPLEMENTATION_NOTES.md §11."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    kind: Literal["none", "fixed_interval", "threshold"]
    interval: int | None = Field(default=None, gt=0)
    action: Action = Action.FULL_REINFORCEMENT
    risk_source: Literal["rule", "model"] = "rule"
    model_path: Path | None = None
    # Risk band edges: below [0] and [1] no action; [1]-[2] compressed; [2]-[3] selective;
    # above [3] top_action. A single threshold uses `single_threshold` instead.
    bands: list[float] = [0.2, 0.4, 0.6, 0.8]
    single_threshold: float | None = Field(default=None, ge=0, le=1)
    top_action: Action = Action.FULL_REINFORCEMENT
    cooldown_exchanges: int = Field(default=0, ge=0)
    max_reinforcement_tokens: int | None = Field(default=None, gt=0)
    min_priority: float = Field(default=0.0, ge=0, le=1)


B0 = PolicyConfig(name=B0_POLICY, kind="none")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    seed: int
    output_root: Path = Path("results/runs")
    provider: ProviderConfig
    generation: GenerationParams = GenerationParams()
    design: DesignConfig | None = None
    system_prompt: str = "You are a helpful assistant."
    policies: list[PolicyConfig] = [B0]


class ConfigError(ValueError):
    pass


def _reject_secrets(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            key_s = str(key)
            if _SECRET_KEY.search(key_s) and key_s not in _ALLOWED_TOKEN_KEYS:
                raise ConfigError(
                    f"Config key '{path}{key_s}' looks like a secret. "
                    "Put credentials in environment variables, not YAML."
                )
            _reject_secrets(value, f"{path}{key_s}.")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _reject_secrets(item, f"{path}{i}.")


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    _reject_secrets(raw)
    return ExperimentConfig.model_validate(raw)

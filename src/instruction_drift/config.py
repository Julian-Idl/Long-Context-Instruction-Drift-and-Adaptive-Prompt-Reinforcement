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

from instruction_drift.schemas import GenerationParams

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


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    seed: int
    output_root: Path = Path("results/runs")
    provider: ProviderConfig
    generation: GenerationParams = GenerationParams()
    design: dict[str, Any] = Field(default_factory=dict)


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

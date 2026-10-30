"""Append-only run storage.

Layout: <output_root>/<provider_kind>/<run_id>/ so mock and real-provider runs can
never land in the same directory. A run directory is created exactly once; reopening
an existing run for writing only ever appends.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from instruction_drift import __version__
from instruction_drift.config import ExperimentConfig
from instruction_drift.schemas import utc_now

M = TypeVar("M", bound=BaseModel)

STREAMS = {
    "conversations": "raw/conversations.jsonl",
    "instructions": "raw/instructions.jsonl",
    "lifecycle": "raw/lifecycle_events.jsonl",
    "prompts": "raw/prompts.jsonl",
    "responses": "raw/responses.jsonl",
    "compliance": "labels/compliance.jsonl",
    "human_audit": "labels/human_audit.jsonl",
    "decisions": "processed/decisions.jsonl",
}

_PACKAGES = ["pydantic", "pyyaml", "httpx", "tiktoken", "numpy", "pandas", "scikit-learn"]


def new_run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}_{stamp}_{uuid.uuid4().hex[:6]}"


def _git_state(cwd: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


class RunStore:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = run_dir
        self.run_id = run_id

    @classmethod
    def create(cls, config: ExperimentConfig, run_id: str | None = None) -> RunStore:
        run_id = run_id or new_run_id(config.name)
        run_dir = Path(config.output_root) / config.provider.kind / run_id
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)  # never reuse or overwrite a run
        store = cls(run_dir, run_id)
        manifest = {
            "run_id": run_id,
            "created_at": utc_now(),
            "is_simulated": config.provider.kind == "mock",
            "package_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_state(Path.cwd()),
            "dependencies": _package_versions(),
            "config": config.model_dump(mode="json"),
        }
        with open(run_dir / "run_manifest.json", "x", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return store

    @classmethod
    def open(cls, run_dir: Path) -> RunStore:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        return cls(run_dir, manifest["run_id"])

    def manifest(self) -> dict:
        return json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    def path(self, stream: str) -> Path:
        return self.run_dir / STREAMS[stream]

    def append(self, stream: str, record: BaseModel) -> None:
        self.append_many(stream, [record])

    def append_many(self, stream: str, records: list[BaseModel]) -> None:
        p = self.path(stream)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
            f.flush()

    def read(self, stream: str, model: type[M]) -> Iterator[M]:
        p = self.path(stream)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield model.model_validate_json(line)

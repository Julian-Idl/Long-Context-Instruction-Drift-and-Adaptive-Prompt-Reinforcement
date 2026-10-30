"""Command-line entry point: python -m instruction_drift.cli <group> <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from instruction_drift.config import ConfigError, load_config


def _config_validate(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (ConfigError, ValidationError, OSError) as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print(f"OK: {cfg.name} (provider={cfg.provider.kind}, model={cfg.provider.model_id})")
    return 0


def _data_generate(args: argparse.Namespace) -> int:
    from instruction_drift.runner import build_plans

    cfg = load_config(args.config)
    plans, excluded = build_plans(cfg)
    print(f"valid plans: {len(plans)}  excluded: {len(excluded)}")
    for e in excluded:
        print(f"  EXCLUDED {e.conversation_id}: {'; '.join(e.errors)}")
    for p in plans[: args.show]:
        a = p.achieved
        print(f"  {p.conversation_id}: tokens={a.total_tokens}"
              f" pos={a.instruction_position_ratio:.2f}"
              f" density={a.distractor_density:.2f} conflicts={sum(a.conflict_counts.values())}"
              f" transitions={a.task_transition_count} checkpoints={a.checkpoint_count}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        with open(out / "plans.jsonl", "x", encoding="utf-8", newline="\n") as f:
            f.writelines(p.model_dump_json() + "\n" for p in plans)
        with open(out / "exclusions.jsonl", "x", encoding="utf-8", newline="\n") as f:
            f.writelines(e.model_dump_json() + "\n" for e in excluded)
        print(f"wrote {out}")
    return 0


def _experiment_run(args: argparse.Namespace) -> int:
    from instruction_drift.runner import run_experiment

    cfg = load_config(args.config)
    if args.provider:
        provider = cfg.provider.model_copy(update={"kind": args.provider})
        if args.provider == "mock":
            provider = provider.model_copy(update={"model_id": "mock-scripted-v1"})
        cfg = cfg.model_copy(update={"provider": provider})
    store, summary = run_experiment(cfg, max_conversations=args.max_conversations)
    print(json.dumps(summary, indent=2))
    print(f"run directory: {store.run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="instruction_drift")
    groups = parser.add_subparsers(dest="group", required=True)

    config = groups.add_parser("config").add_subparsers(dest="command", required=True)
    validate = config.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_config_validate)

    data = groups.add_parser("data").add_subparsers(dest="command", required=True)
    gen = data.add_parser("generate", help="plan and validate conversations (no LLM calls)")
    gen.add_argument("--config", required=True)
    gen.add_argument("--out", help="new directory to write plans.jsonl and exclusions.jsonl")
    gen.add_argument("--show", type=int, default=10)
    gen.set_defaults(func=_data_generate)

    exp = groups.add_parser("experiment").add_subparsers(dest="command", required=True)
    run = exp.add_parser("run", help="run B0 (no reinforcement) on the configured provider")
    run.add_argument("--config", required=True)
    run.add_argument("--provider", choices=["mock", "ollama", "openai_compat"])
    run.add_argument("--max-conversations", type=int)
    run.set_defaults(func=_experiment_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Command-line entry point: python -m instruction_drift.cli <group> <command> ..."""

from __future__ import annotations

import argparse
import sys

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="instruction_drift")
    groups = parser.add_subparsers(dest="group", required=True)

    config = groups.add_parser("config").add_subparsers(dest="command", required=True)
    validate = config.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_config_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

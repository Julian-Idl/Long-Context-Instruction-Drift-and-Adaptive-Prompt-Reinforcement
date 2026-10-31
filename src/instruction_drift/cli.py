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
    if args.base_url:
        cfg = cfg.model_copy(update={"provider": cfg.provider.model_copy(
            update={"base_url": args.base_url})})
    if args.output_root:
        cfg = cfg.model_copy(update={"output_root": Path(args.output_root)})
    budget = args.time_budget_hours * 3600 if args.time_budget_hours else None
    store, summary = run_experiment(cfg, max_conversations=args.max_conversations,
                                    offset=args.offset, time_budget_s=budget)
    print(json.dumps(summary, indent=2))
    print(f"run directory: {store.run_dir}")
    return 0


def _dataset_build(args: argparse.Namespace) -> int:
    from instruction_drift.dataset import build_dataset

    tests = args.test_templates.split(",") if args.test_templates else None
    manifest = build_dataset([Path(r) for r in args.runs], Path(args.out), seed=args.seed,
                             strategy=args.strategy, test_templates=tests,
                             allow_mixed_models=args.allow_mixed_models)
    print(json.dumps({k: manifest[k] for k in ("rows", "positives", "groups", "is_simulated",
                                               "model_versions")}, indent=2))
    return 0


def _predictor_train(args: argparse.Namespace) -> int:
    from instruction_drift.predictor import MODELS, train

    models = args.models.split(",") if args.models else MODELS
    board = train(Path(args.dataset), Path(args.out), models=models, seed=args.seed,
                  calibration=args.calibration, n_boot=args.n_boot)
    print("validation (calibration and threshold fitted here, so optimistic):")
    print(json.dumps(board, indent=2))
    return 0


def _predictor_evaluate(args: argparse.Namespace) -> int:
    from instruction_drift.predictor import evaluate

    if args.split == "test" and not args.final:
        print("Refusing to touch the test split without --final.", file=sys.stderr)
        return 1
    print(json.dumps(evaluate(Path(args.model), Path(args.dataset), args.split,
                              n_boot=args.n_boot), indent=2))
    return 0


def _demo(args: argparse.Namespace) -> int:
    from instruction_drift.demo import run_demo

    cfg = load_config(args.config)
    if args.provider:
        update = {"kind": args.provider}
        if args.provider == "mock":
            update["model_id"] = "mock-scripted-v1"
        cfg = cfg.model_copy(update={"provider": cfg.provider.model_copy(update=update)})
    policy = args.policy or ("B5_rule_threshold" if cfg.provider.kind == "mock"
                             else "P1_predictive")
    run_demo(cfg, policy, args.plan)
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
    run = exp.add_parser("run", help="run every configured policy on the same plans")
    run.add_argument("--config", required=True)
    run.add_argument("--provider", choices=["mock", "ollama", "openai_compat"])
    run.add_argument("--max-conversations", type=int)
    run.add_argument("--offset", type=int, default=0,
                     help="skip the first N valid plans (resume a design across sessions)")
    run.add_argument("--time-budget-hours", type=float,
                     help="start no new conversation after this much wall-clock time")
    run.add_argument("--output-root", help="override config output_root")
    run.add_argument("--base-url", help="override provider base_url (e.g. a second server)")
    run.set_defaults(func=_experiment_run)

    ds = groups.add_parser("dataset").add_subparsers(dest="command", required=True)
    b = ds.add_parser("build", help="checkpoint-level feature rows with grouped splits")
    b.add_argument("--runs", nargs="+", required=True, help="run directories")
    b.add_argument("--out", required=True, help="new output directory")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--strategy", choices=["conversation", "template"], default="conversation")
    b.add_argument("--test-templates", help="comma-separated, for --strategy template")
    b.add_argument("--allow-mixed-models", action="store_true")
    b.set_defaults(func=_dataset_build)

    pr = groups.add_parser("predictor").add_subparsers(dest="command", required=True)
    t = pr.add_parser("train", help="fit on train, calibrate and pick threshold on validation")
    t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True, help="new directory, one subfolder per model")
    t.add_argument("--models", help="comma-separated subset of the default model list")
    t.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n-boot", type=int, default=1000)
    t.set_defaults(func=_predictor_train)
    e = pr.add_parser("evaluate")
    e.add_argument("--model", required=True, help="one model subfolder")
    e.add_argument("--dataset", required=True)
    e.add_argument("--split", choices=["validation", "test"], default="validation")
    e.add_argument("--final", action="store_true", help="required to evaluate on test")
    e.add_argument("--n-boot", type=int, default=1000)
    e.set_defaults(func=_predictor_evaluate)

    dm = groups.add_parser("demo", help="print one conversation live, step by step")
    dm.add_argument("--config", default="configs/demo.yaml")
    dm.add_argument("--provider", choices=["mock", "ollama", "openai_compat"])
    dm.add_argument("--policy", help="default: P1_predictive (real) or B5_rule_threshold (mock)")
    dm.add_argument("--plan", type=int, help="plan index (default: first with a conflict)")
    dm.set_defaults(func=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

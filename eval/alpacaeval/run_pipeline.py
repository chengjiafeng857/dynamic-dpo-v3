"""Staged AlpacaEval pipeline for prepare/generate/judge/report."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_model import evaluate_outputs
from pipeline_lib import (
    DEFAULT_DATASET_REPO,
    build_report,
    format_report,
    generate_run,
    load_manifest,
    prepare_run,
    utc_now_iso,
    write_manifest,
)


def _run_prepare(args: argparse.Namespace) -> None:
    run_dir = prepare_run(
        preset_name=args.preset,
        run_name=args.run_name,
        output_root=args.output_root,
        dataset_repo=args.dataset_repo,
        data_file=args.data_file,
        max_instances=args.max_instances,
        force=args.force,
    )
    print(run_dir)


def _run_generate(args: argparse.Namespace) -> None:
    outputs_file = generate_run(
        args.run_dir,
        device=args.device,
        seed=args.seed,
        max_input_tokens=args.max_input_tokens,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    print(outputs_file)


def _run_judge(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    manifest = load_manifest(run_dir)
    evaluate_outputs(
        model_name=str(manifest.get("resolved_model_name", manifest["preset"]["model_name"])),
        output_dir=str(run_dir),
        outputs_file=str(run_dir / "outputs" / "model_outputs.json"),
        run_dir=str(run_dir),
        name=str(manifest["preset"]["pretty_name"]),
        annotators_config=args.annotators_config,
        reference_outputs=args.reference_outputs,
        reference_repo=args.reference_repo,
        max_instances=args.max_instances,
        skip_generation=True,
        skip_eval=False,
        skip_analysis=args.skip_analysis,
        dataset_repo=args.dataset_repo,
        data_file=None,
        batch_size=1,
        device=args.device,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=1,
    )
    manifest["status"]["judged_at"] = utc_now_iso()
    write_manifest(run_dir, manifest)
    print(run_dir / "results")


def _run_report(args: argparse.Namespace) -> None:
    report = build_report(args.run_dir)
    print(format_report(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged AlpacaEval pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze the dataset and preset into a run dir")
    prepare.add_argument("--preset", required=True, help="Preset name or YAML path")
    prepare.add_argument("--run_name", required=True, help="Run directory name")
    prepare.add_argument("--output_root", default=None, help="Optional root directory for runs")
    prepare.add_argument("--dataset_repo", default=DEFAULT_DATASET_REPO, help="Dataset repo id")
    prepare.add_argument("--data_file", default=None, help="Optional local dataset file")
    prepare.add_argument("--max_instances", type=int, default=None, help="Optional dataset row limit")
    prepare.add_argument("--force", action="store_true", help="Allow reuse of a non-empty run dir")
    prepare.set_defaults(func=_run_prepare)

    generate = subparsers.add_parser("generate", help="Render prompts and generate outputs")
    generate.add_argument("--run_dir", required=True, help="Prepared run directory")
    generate.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--max_input_tokens", type=int, default=2048)
    generate.add_argument("--model_name", default=None, help="Override preset model name")
    generate.add_argument("--batch_size", type=int, default=None, help="Override preset batch size")
    generate.set_defaults(func=_run_generate)

    judge = subparsers.add_parser("judge", help="Run AlpacaEval on pre-generated outputs")
    judge.add_argument("--run_dir", required=True, help="Prepared run directory")
    judge.add_argument("--annotators_config", default="weighted_alpaca_eval_gpt4_turbo")
    judge.add_argument("--reference_outputs", default="gpt4_turbo")
    judge.add_argument("--reference_repo", default=DEFAULT_DATASET_REPO)
    judge.add_argument("--dataset_repo", default=DEFAULT_DATASET_REPO)
    judge.add_argument("--max_instances", type=int, default=None)
    judge.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    judge.add_argument("--skip_analysis", action="store_true")
    judge.set_defaults(func=_run_judge)

    report = subparsers.add_parser("report", help="Summarize an AlpacaEval run")
    report.add_argument("--run_dir", required=True, help="Prepared run directory")
    report.set_defaults(func=_run_report)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

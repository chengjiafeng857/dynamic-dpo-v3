"""
Evaluate a model with AlpacaEval 2.0.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

from pipeline_lib import SIMPO_PARITY_VERSION, get_installed_alpaca_eval_version, sanitize_generator_name


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _tokenize(value: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", value.lower()) if part]


def _sanitize_generators(outputs_df):
    for column in ("generator", "generator_1", "generator_2"):
        if column in outputs_df.columns:
            outputs_df[column] = outputs_df[column].apply(
                lambda value: sanitize_generator_name(str(value))
            )
    return outputs_df


def _load_and_sanitize_outputs(path: str | Path):
    from alpaca_eval import utils as alpaca_utils

    outputs_df = alpaca_utils.load_or_convert_to_dataframe(path)
    return _sanitize_generators(outputs_df)


def _load_alpaca_eval_main():
    return importlib.import_module("alpaca_eval.main")


def preflight_evaluation_env() -> None:
    version = get_installed_alpaca_eval_version()
    if version is None:
        raise RuntimeError("alpaca-eval is not installed in the current environment.")
    if version != SIMPO_PARITY_VERSION:
        print(
            f"Warning: installed alpaca-eval version is {version}; "
            f"SimPO parity recommends {SIMPO_PARITY_VERSION}.",
            file=sys.stderr,
        )

    try:
        import pkg_resources  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "alpaca_eval requires pkg_resources during import. Install or sync setuptools first."
        ) from exc

    try:
        _load_alpaca_eval_main()
    except Exception as exc:
        raise RuntimeError("Failed to import alpaca_eval. Check the local evaluation environment.") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for AlpacaEval judging.")


def _resolve_io_paths(
    *,
    output_dir: str,
    outputs_file: str | None,
    run_dir: str | None,
):
    if run_dir:
        resolved_run_dir = Path(run_dir).resolve()
        resolved_output_dir = resolved_run_dir
        resolved_outputs_file = Path(outputs_file).resolve() if outputs_file else resolved_run_dir / "outputs" / "model_outputs.json"
    else:
        resolved_output_dir = Path(output_dir).resolve()
        resolved_outputs_file = Path(outputs_file).resolve() if outputs_file else resolved_output_dir / "outputs" / "model_outputs.json"

    outputs_dir = resolved_outputs_file.parent
    results_dir = (resolved_output_dir if run_dir else resolved_output_dir) / "results"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return resolved_output_dir, outputs_dir, results_dir, resolved_outputs_file


def evaluate_outputs(
    *,
    model_name: str,
    output_dir: str,
    name: str | None = None,
    annotators_config: str = "weighted_alpaca_eval_gpt4_turbo",
    reference_outputs: str = "gpt4_turbo",
    skip_generation: bool = False,
    skip_eval: bool = False,
    skip_analysis: bool = False,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    device: str = _default_device(),
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_instances: int | None = None,
    dataset_repo: str = "tatsu-lab/alpaca_eval",
    data_file: str | None = None,
    reference_repo: str = "tatsu-lab/alpaca_eval",
    outputs_file: str | None = None,
    run_dir: str | None = None,
) -> None:
    resolved_output_dir, outputs_dir, results_dir, resolved_outputs_file = _resolve_io_paths(
        output_dir=output_dir,
        outputs_file=outputs_file,
        run_dir=run_dir,
    )

    script_dir = Path(__file__).resolve().parent
    generate_script = script_dir / "generate_outputs.py"
    analyze_script = script_dir / "analyze_results.py"
    leaderboard_name = name or model_name.split("/")[-1]

    if not skip_generation:
        cmd = [
            sys.executable,
            str(generate_script),
            "--model_name",
            model_name,
            "--output_file",
            str(resolved_outputs_file),
            "--max_new_tokens",
            str(max_new_tokens),
            "--batch_size",
            str(batch_size),
            "--device",
            device,
            "--temperature",
            str(temperature),
            "--top_p",
            str(top_p),
            "--dataset_repo",
            dataset_repo,
        ]
        if max_instances is not None:
            cmd.extend(["--max_instances", str(max_instances)])
        if data_file:
            cmd.extend(["--data_file", data_file])
        _run(cmd)
    elif not resolved_outputs_file.exists():
        raise FileNotFoundError(
            f"Outputs not found at {resolved_outputs_file}. Run without --skip_generation."
        )

    if not skip_eval:
        preflight_evaluation_env()
        alpaca_main = _load_alpaca_eval_main()
        reference_outputs_path = _resolve_reference_outputs(reference_outputs, reference_repo)
        model_outputs_df = _load_and_sanitize_outputs(resolved_outputs_file)
        reference_outputs_df = _load_and_sanitize_outputs(reference_outputs_path)
        weights_dir = outputs_dir / "weights"

        alpaca_main.evaluate(
            model_outputs=model_outputs_df,
            reference_outputs=reference_outputs_df,
            annotators_config=annotators_config,
            output_path=str(results_dir),
            name=leaderboard_name,
            max_instances=max_instances,
            metric_kwargs={"save_weights_dir": str(weights_dir)},
        )

    if not skip_analysis:
        cmd = [
            sys.executable,
            str(analyze_script),
            "--results_dir",
            str(results_dir),
            "--model_name",
            leaderboard_name,
        ]
        _run(cmd)


def _select_reference_file(repo_id: str, alias: str) -> str:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates = [
        file
        for file in files
        if file.lower().endswith((".json", ".jsonl", ".parquet", ".csv", ".tsv"))
    ]
    if not candidates:
        raise RuntimeError("No reference output files found in the dataset repo.")

    tokens = _tokenize(alias)
    if tokens:
        matched = [
            file
            for file in candidates
            if all(token in file.lower() for token in tokens)
        ]
        if matched:
            return sorted(matched)[0]

    for file in candidates:
        lowered = file.lower()
        if "gpt4" in lowered and "baseline" in lowered:
            return file

    return sorted(candidates)[0]


def _resolve_reference_outputs(reference_outputs: str, reference_repo: str) -> str:
    path = Path(reference_outputs)
    if path.suffix in {".json", ".jsonl", ".parquet", ".csv", ".tsv"}:
        if path.exists():
            return str(path)
        return hf_hub_download(
            repo_id=reference_repo, repo_type="dataset", filename=path.name
        )

    alias_map = {
        "gpt4_turbo": "alpaca_eval_gpt4_baseline.json",
        "gpt4": "alpaca_eval_gpt4_baseline.json",
        "gpt4_baseline": "alpaca_eval_gpt4_baseline.json",
        "alpaca_eval_gpt4_baseline": "alpaca_eval_gpt4_baseline.json",
    }
    alias_key = reference_outputs.lower().strip()
    if alias_key in alias_map:
        filename = alias_map[alias_key]
    else:
        filename = _select_reference_file(reference_repo, reference_outputs)

    return hf_hub_download(
        repo_id=reference_repo, repo_type="dataset", filename=filename
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model with AlpacaEval 2.0")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test/alpacaeval",
        help="Output directory for outputs/results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Leaderboard name override",
    )
    parser.add_argument(
        "--annotators_config",
        type=str,
        default="weighted_alpaca_eval_gpt4_turbo",
        help="AlpacaEval annotators config",
    )
    parser.add_argument(
        "--reference_outputs",
        type=str,
        default="gpt4_turbo",
        help="Reference outputs name or file path",
    )
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation step")
    parser.add_argument("--skip_eval", action="store_true", help="Skip AlpacaEval step")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip analysis step")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        choices=["cpu", "cuda", "mps"],
        help="Device for generation",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max_instances", type=int, default=None, help="Limit number of prompts")
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="tatsu-lab/alpaca_eval",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Local dataset file path (JSON/JSONL/Parquet)",
    )
    parser.add_argument(
        "--reference_repo",
        type=str,
        default="tatsu-lab/alpaca_eval",
        help="HuggingFace dataset repo ID for reference outputs",
    )
    parser.add_argument(
        "--outputs_file",
        type=str,
        default=None,
        help="Existing AlpacaEval-compatible outputs JSON file",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing outputs/model_outputs.json and results/",
    )

    args = parser.parse_args()
    evaluate_outputs(
        model_name=args.model_name,
        output_dir=args.output_dir,
        name=args.name,
        annotators_config=args.annotators_config,
        reference_outputs=args.reference_outputs,
        skip_generation=args.skip_generation,
        skip_eval=args.skip_eval,
        skip_analysis=args.skip_analysis,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_instances=args.max_instances,
        dataset_repo=args.dataset_repo,
        data_file=args.data_file,
        reference_repo=args.reference_repo,
        outputs_file=args.outputs_file,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()

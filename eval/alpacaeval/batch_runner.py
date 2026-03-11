"""Batch orchestration for AlpacaEval inference and evaluation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

from src.config.loader import load_yaml

from .alpacaeval_common import get_output_dir, sanitize_name
from .alpacaeval_eval import run_alpacaeval_evaluation
from .alpacaeval_infer import run_alpacaeval_inference


def _resolve_config_path(config_path: str, base_dir: Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _apply_model_family_defaults(
    config: Dict[str, Any],
    *,
    model_name_or_path: str,
    pretty_name: str,
) -> None:
    alpacaeval_cfg = config.setdefault("alpacaeval", {})
    generation_cfg = alpacaeval_cfg.setdefault("generation", {})
    model_name = model_name_or_path.lower()
    if "qwen3" in model_name:
        alpacaeval_cfg["use_custom_chat_template"] = False
        alpacaeval_cfg.pop("prompt_template", None)
        generation_cfg["stop_token_ids"] = [151645]
    elif "llama3" in model_name or "llama-3" in model_name:
        alpacaeval_cfg["use_custom_chat_template"] = True
        alpacaeval_cfg["prompt_template"] = (
            f"templates/{sanitize_name(pretty_name)}.txt"
        )
        generation_cfg["stop_token_ids"] = [128001, 128009]


def _build_model_config(
    base_config: Dict[str, Any],
    *,
    batch_config: Dict[str, Any],
    model_entry: Dict[str, Any],
    config_path: Path,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    batch_overrides = batch_config.get("overrides", {})
    if batch_overrides:
        if not isinstance(batch_overrides, dict):
            raise ValueError("Batch overrides must be a mapping.")
        _deep_update(config, batch_overrides)

    model_name_or_path = str(model_entry["model_name_or_path"])
    pretty_name = str(model_entry.get("pretty_name", model_name_or_path))

    config["policy_name"] = model_name_or_path
    alpacaeval_cfg = config.setdefault("alpacaeval", {})
    alpacaeval_cfg["model_name_or_path"] = model_name_or_path
    alpacaeval_cfg["pretty_name"] = pretty_name
    alpacaeval_cfg["output_dir"] = str(
        Path("../../outputs/alpacaeval") / sanitize_name(pretty_name)
    )

    _apply_model_family_defaults(
        config,
        model_name_or_path=model_name_or_path,
        pretty_name=pretty_name,
    )

    model_overrides = model_entry.get("overrides", {})
    if model_overrides:
        if not isinstance(model_overrides, dict):
            raise ValueError("Per-model overrides must be a mapping.")
        _deep_update(config, model_overrides)

    config["_config_path"] = str(config_path)
    return config


def build_run_matrix(batch_config: Dict[str, Any], *, config_path: Path) -> List[Dict[str, Any]]:
    base_config_value = batch_config.get("base_config")
    if not base_config_value:
        raise ValueError("base_config is required.")

    base_config_path = _resolve_config_path(str(base_config_value), config_path.parent)
    base_config = load_yaml(str(base_config_path))

    model_entries = batch_config.get("models", [])
    if not isinstance(model_entries, list) or not model_entries:
        raise ValueError("models must be a non-empty list.")

    run_plans: List[Dict[str, Any]] = []
    for model_entry in model_entries:
        if not isinstance(model_entry, dict):
            raise ValueError("Each model entry must be a mapping.")
        config = _build_model_config(
            base_config,
            batch_config=batch_config,
            model_entry=model_entry,
            config_path=config_path,
        )
        run_plans.append(
            {
                "model_name_or_path": str(config["alpacaeval"]["model_name_or_path"]),
                "pretty_name": str(config["alpacaeval"]["pretty_name"]),
                "output_dir": str(get_output_dir(config)),
                "config": config,
            }
        )
    return run_plans


def _results_exist(config: Dict[str, Any]) -> bool:
    results_dir = get_output_dir(config) / "results"
    return results_dir.exists() and any(results_dir.iterdir())


def run_alpacaeval_batch(
    batch_config: Dict[str, Any],
    *,
    config_path: str,
    run_inference: bool | None = None,
    run_evaluation: bool | None = None,
    use_model_configs: bool | None = None,
) -> int:
    config_file = Path(config_path).resolve()
    run_plans = build_run_matrix(batch_config, config_path=config_file)
    do_inference = (
        bool(batch_config.get("run_inference", True))
        if run_inference is None
        else run_inference
    )
    do_evaluation = (
        bool(batch_config.get("run_evaluation", True))
        if run_evaluation is None
        else run_evaluation
    )
    if not do_inference and not do_evaluation:
        raise ValueError("At least one of run_inference or run_evaluation must be enabled.")

    skip_existing = bool(batch_config.get("skip_existing", True))
    eval_with_model_configs = (
        bool(batch_config.get("use_model_configs", False))
        if use_model_configs is None
        else use_model_configs
    )
    continue_on_error = bool(batch_config.get("continue_on_error", False))

    failures: list[str] = []
    for run_index, run_plan in enumerate(run_plans, start=1):
        config = run_plan["config"]
        pretty_name = run_plan["pretty_name"]
        output_dir = Path(run_plan["output_dir"])
        model_outputs_path = output_dir / "model_outputs.json"

        print(
            f"[AlpacaEval-BATCH] ({run_index}/{len(run_plans)}) "
            f"model={pretty_name}"
        )
        try:
            if do_inference:
                if skip_existing and model_outputs_path.exists():
                    print(
                        "[AlpacaEval-BATCH] skipping inference; "
                        f"existing_outputs={model_outputs_path}"
                    )
                else:
                    run_alpacaeval_inference(config)

            if do_evaluation:
                if skip_existing and not eval_with_model_configs and _results_exist(config):
                    print(
                        "[AlpacaEval-BATCH] skipping evaluation; "
                        f"existing_results={output_dir / 'results'}"
                    )
                else:
                    resolved_outputs = str(model_outputs_path) if model_outputs_path.exists() else None
                    run_alpacaeval_evaluation(
                        config,
                        model_outputs_path=resolved_outputs,
                        use_model_configs=eval_with_model_configs,
                    )
        except Exception as exc:
            message = f"{pretty_name}: {exc}"
            print(f"[AlpacaEval-BATCH] failed: {message}")
            failures.append(message)
            if not continue_on_error:
                break

    if failures:
        print("[AlpacaEval-BATCH] failed_models=")
        for message in failures:
            print(f"  - {message}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AlpacaEval for a model batch")
    parser.add_argument(
        "--config",
        type=str,
        default="eval/alpacaeval/config_alpacaeval_batch.yaml",
    )
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-model-configs", action="store_true")
    args = parser.parse_args(argv)

    if args.inference_only and args.eval_only:
        raise ValueError("Choose at most one of --inference-only or --eval-only.")

    batch_config = load_yaml(args.config)
    run_inference = None if not args.eval_only else False
    run_evaluation = None if not args.inference_only else False
    return run_alpacaeval_batch(
        batch_config,
        config_path=args.config,
        run_inference=run_inference,
        run_evaluation=run_evaluation,
        use_model_configs=True if args.use_model_configs else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())

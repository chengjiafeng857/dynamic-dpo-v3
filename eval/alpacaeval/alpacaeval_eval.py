"""Evaluation wrapper for AlpacaEval."""

from __future__ import annotations

import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import yaml

from src.config.loader import resolve_torch_dtype
from .alpacaeval_common import (
    DEFAULT_ALPACA_EVAL_ANNOTATOR,
    get_alpacaeval_config,
    get_generation_config,
    get_model_name_or_path,
    get_output_dir,
    get_pretty_name,
    load_prompt_template,
    resolve_path,
    sanitize_name,
    use_custom_chat_template,
)


SIMPO_ALPACA_EVAL_VERSION = "0.6.2"


def _torch_dtype_name(config: Dict[str, Any]) -> str | None:
    torch_dtype = resolve_torch_dtype(config.get("precision", "fp32"))
    if torch_dtype is None:
        return None
    return str(torch_dtype).replace("torch.", "")


def _ensure_alpacaeval_version(config: Dict[str, Any]) -> str:
    try:
        version = metadata.version("alpaca-eval")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "alpaca-eval is not installed in the current environment."
        ) from exc

    if bool(get_alpacaeval_config(config).get("simpo_compat", False)) and version != SIMPO_ALPACA_EVAL_VERSION:
        raise RuntimeError(
            "SimPO-compatible AlpacaEval evaluation requires alpaca-eval==0.6.2; "
            f"found {version}."
        )
    return version


def _alpacaeval_command(config: Dict[str, Any]) -> list[str]:
    command = get_alpacaeval_config(config).get("command")
    if command is None:
        return [sys.executable, "-m", "alpaca_eval.main"]
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, list):
        return [str(part) for part in command]
    raise ValueError("alpacaeval.command must be a string or list.")


def _build_runtime_model_config(config: Dict[str, Any]) -> Path:
    if not use_custom_chat_template(config):
        raise ValueError(
            "alpacaeval.use_custom_chat_template=false is not supported with "
            "evaluate_from_model. Use alpacaeval-eval on saved model outputs instead."
        )
    alpacaeval_cfg = get_alpacaeval_config(config)
    output_dir = get_output_dir(config)
    prompt_template_path, _ = load_prompt_template(config)
    generation_cfg = get_generation_config(config)
    backend = str(alpacaeval_cfg.get("backend", "transformers")).lower()

    if backend == "vllm":
        fn_completions = "vllm_local_completions"
    elif backend == "transformers":
        fn_completions = "huggingface_local_completions"
    else:
        raise ValueError("alpacaeval.backend must be 'vllm' or 'transformers'.")

    completions_kwargs: Dict[str, Any] = {
        "model_name": get_model_name_or_path(config),
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 1024)),
        "temperature": float(generation_cfg.get("temperature", 0.0)),
        "top_p": float(generation_cfg.get("top_p", 1.0)),
        "do_sample": bool(generation_cfg.get("do_sample", False)),
        "batch_size": int(generation_cfg.get("batch_size", 1)),
    }
    stop_token_ids = generation_cfg.get("stop_token_ids")
    if stop_token_ids:
        completions_kwargs["stop_token_ids"] = [int(token_id) for token_id in stop_token_ids]

    torch_dtype_name = _torch_dtype_name(config)
    if torch_dtype_name is not None:
        completions_kwargs["model_kwargs"] = {"torch_dtype": torch_dtype_name}

    model_key = sanitize_name(get_pretty_name(config))
    payload = {
        model_key: {
            "prompt_template": str(prompt_template_path),
            "fn_completions": fn_completions,
            "completions_kwargs": completions_kwargs,
            "pretty_name": get_pretty_name(config),
        }
    }

    config_path = output_dir / "alpacaeval_model_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _resolve_model_outputs_path(
    config: Dict[str, Any],
    model_outputs_path: str | None,
) -> Path:
    if model_outputs_path is not None:
        return resolve_path(config, model_outputs_path)

    default_path = get_output_dir(config) / "model_outputs.json"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Could not find model outputs. Run alpacaeval-infer first or pass --model-outputs."
    )


def run_alpacaeval_evaluation(
    config: Dict[str, Any],
    *,
    model_outputs_path: str | None = None,
    use_model_configs: bool = False,
) -> Path:
    version = _ensure_alpacaeval_version(config)
    alpacaeval_cfg = get_alpacaeval_config(config)
    output_dir = get_output_dir(config)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    annotators_config = str(
        alpacaeval_cfg.get("annotators_config", DEFAULT_ALPACA_EVAL_ANNOTATOR)
    )

    command = _alpacaeval_command(config)
    if use_model_configs or str(alpacaeval_cfg.get("evaluation_mode", "outputs")).lower() == "model_configs":
        runtime_model_config = _build_runtime_model_config(config)
        command.extend(
            [
                "evaluate_from_model",
                "--model_configs",
                str(runtime_model_config),
                "--annotators_config",
                annotators_config,
                "--output_path",
                str(results_dir),
            ]
        )
        reference_model_configs = alpacaeval_cfg.get("reference_model_configs")
        if reference_model_configs:
            command.extend(
                [
                    "--reference_model_configs",
                    str(resolve_path(config, reference_model_configs)),
                ]
            )
    else:
        resolved_model_outputs = _resolve_model_outputs_path(config, model_outputs_path)
        command.extend(
            [
                "evaluate",
                "--model_outputs",
                str(resolved_model_outputs),
                "--annotators_config",
                annotators_config,
                "--output_path",
                str(results_dir),
            ]
        )
        reference_outputs = alpacaeval_cfg.get("reference_outputs")
        if reference_outputs:
            command.extend(
                ["--reference_outputs", str(resolve_path(config, reference_outputs))]
            )

    print(f"[AlpacaEval] alpaca_eval_version={version}")
    print(f"[AlpacaEval] command={' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, check=True)
    print(f"[AlpacaEval] results_dir={results_dir}")
    return results_dir

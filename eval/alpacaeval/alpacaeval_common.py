"""Shared helpers for AlpacaEval inference and evaluation."""

from __future__ import annotations

import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any, Dict


DEFAULT_ALPACA_EVAL_DATASET = "tatsu-lab/alpaca_eval"
DEFAULT_ALPACA_EVAL_CONFIG = "alpaca_eval"
DEFAULT_ALPACA_EVAL_SPLIT = "eval"
DEFAULT_ALPACA_EVAL_ANNOTATOR = "weighted_alpaca_eval_gpt4_turbo"


def get_alpacaeval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    alpacaeval_cfg = config.get("alpacaeval", {})
    if not isinstance(alpacaeval_cfg, dict):
        raise ValueError("alpacaeval config must be a mapping.")
    return alpacaeval_cfg


def get_model_name_or_path(config: Dict[str, Any]) -> str:
    alpacaeval_cfg = get_alpacaeval_config(config)
    model_name = alpacaeval_cfg.get("model_name_or_path") or config.get("policy_name")
    if not model_name:
        raise ValueError("alpacaeval.model_name_or_path or policy_name is required.")
    return str(model_name)


def get_pretty_name(config: Dict[str, Any]) -> str:
    alpacaeval_cfg = get_alpacaeval_config(config)
    pretty_name = alpacaeval_cfg.get("pretty_name") or get_model_name_or_path(config)
    return str(pretty_name)


def resolve_path(config: Dict[str, Any], path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path

    config_path = config.get("_config_path")
    if config_path:
        return (Path(str(config_path)).expanduser().resolve().parent / path).resolve()
    return path.resolve()


def get_output_dir(config: Dict[str, Any]) -> Path:
    alpacaeval_cfg = get_alpacaeval_config(config)
    output_dir = alpacaeval_cfg.get("output_dir")
    if output_dir is None:
        output_dir = Path("outputs") / "alpacaeval" / sanitize_name(get_pretty_name(config))
    resolved = resolve_path(config, output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    alpacaeval_cfg = get_alpacaeval_config(config)
    generation_cfg = alpacaeval_cfg.get("generation", {})
    if not isinstance(generation_cfg, dict):
        raise ValueError("alpacaeval.generation must be a mapping.")
    return generation_cfg


def use_custom_chat_template(config: Dict[str, Any]) -> bool:
    alpacaeval_cfg = get_alpacaeval_config(config)
    return bool(alpacaeval_cfg.get("use_custom_chat_template", True))


def load_prompt_template(config: Dict[str, Any]) -> tuple[Path, str]:
    if not use_custom_chat_template(config):
        raise ValueError(
            "alpacaeval.prompt_template is unavailable when "
            "alpacaeval.use_custom_chat_template=false."
        )
    alpacaeval_cfg = get_alpacaeval_config(config)
    template_path = alpacaeval_cfg.get("prompt_template")
    if not template_path:
        raise ValueError("alpacaeval.prompt_template is required.")

    resolved_path = resolve_path(config, template_path)
    return resolved_path, resolved_path.read_text(encoding="utf-8")


def render_prompt(template_text: str, instruction: str) -> str:
    return template_text.format(instruction=instruction)


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "alpacaeval"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in ("alpaca-eval", "datasets", "torch", "transformers", "vllm"):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return versions

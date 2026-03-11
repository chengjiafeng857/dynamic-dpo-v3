"""Shared helpers for benchmark-specific evaluation packages."""

from __future__ import annotations

import json
import re
import shlex
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List


def get_block_config(config: Dict[str, Any], block_name: str) -> Dict[str, Any]:
    block_cfg = config.get(block_name, {})
    if not isinstance(block_cfg, dict):
        raise ValueError(f"{block_name} config must be a mapping.")
    return block_cfg


def get_model_name_or_path(config: Dict[str, Any], block_name: str) -> str:
    block_cfg = get_block_config(config, block_name)
    model_name = block_cfg.get("model_name_or_path") or config.get("policy_name")
    if not model_name:
        raise ValueError(f"{block_name}.model_name_or_path or policy_name is required.")
    return str(model_name)


def get_pretty_name(config: Dict[str, Any], block_name: str) -> str:
    block_cfg = get_block_config(config, block_name)
    pretty_name = block_cfg.get("pretty_name") or get_model_name_or_path(config, block_name)
    return str(pretty_name)


def resolve_path(config: Dict[str, Any], path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path

    config_path = config.get("_config_path")
    if config_path:
        return (Path(str(config_path)).expanduser().resolve().parent / path).resolve()
    return path.resolve()


def resolve_existing_path(
    config: Dict[str, Any],
    path_value: Any,
    *,
    package_dir: Path,
) -> Path:
    resolved = resolve_path(config, path_value)
    if resolved.exists():
        return resolved

    package_relative = (package_dir / str(path_value)).resolve()
    if package_relative.exists():
        return package_relative
    return resolved


def resolve_existing_or_download_default_path(
    config: Dict[str, Any],
    path_value: Any,
    *,
    package_dir: Path,
    default_filename: str,
    download_url: str,
) -> Path:
    resolved = resolve_existing_path(
        config,
        path_value,
        package_dir=package_dir,
    )
    if resolved.exists():
        return resolved

    if str(path_value) != default_filename:
        return resolved

    package_target = (package_dir / default_filename).resolve()
    package_target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(download_url, timeout=60) as response:
        package_target.write_bytes(response.read())
    return package_target


def get_output_dir(config: Dict[str, Any], block_name: str) -> Path:
    block_cfg = get_block_config(config, block_name)
    output_dir = block_cfg.get("output_dir")
    if output_dir is None:
        output_dir = Path("outputs") / block_name / sanitize_name(get_pretty_name(config, block_name))
    resolved = resolve_path(config, output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_generation_config(config: Dict[str, Any], block_name: str) -> Dict[str, Any]:
    block_cfg = get_block_config(config, block_name)
    generation_cfg = block_cfg.get("generation", {})
    if not isinstance(generation_cfg, dict):
        raise ValueError(f"{block_name}.generation must be a mapping.")
    return generation_cfg


def use_custom_chat_template(config: Dict[str, Any], block_name: str) -> bool:
    block_cfg = get_block_config(config, block_name)
    return bool(block_cfg.get("use_custom_chat_template", True))


def load_prompt_template(
    config: Dict[str, Any],
    block_name: str,
    *,
    package_dir: Path,
) -> tuple[Path, str]:
    if not use_custom_chat_template(config, block_name):
        raise ValueError(
            f"{block_name}.prompt_template is unavailable when "
            f"{block_name}.use_custom_chat_template=false."
        )

    block_cfg = get_block_config(config, block_name)
    template_path = block_cfg.get("prompt_template")
    if not template_path:
        raise ValueError(f"{block_name}.prompt_template is required.")

    resolved_path = resolve_existing_path(
        config,
        template_path,
        package_dir=package_dir,
    )
    return resolved_path, resolved_path.read_text(encoding="utf-8")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "benchmark"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, payload: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True) for row in payload]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object rows in {path}.")
        rows.append(payload)
    return rows


def get_package_versions(package_names: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[str(package_name)] = metadata.version(str(package_name))
        except metadata.PackageNotFoundError:
            continue
    return versions


def format_command(
    command_value: Any,
    *,
    replacements: Dict[str, Any],
    field_name: str,
) -> List[str]:
    if command_value is None:
        raise ValueError(f"{field_name} is required.")

    if isinstance(command_value, str):
        return [
            part.format(**replacements)
            for part in shlex.split(command_value)
        ]
    if isinstance(command_value, list):
        return [str(part).format(**replacements) for part in command_value]
    raise ValueError(f"{field_name} must be a string or list.")

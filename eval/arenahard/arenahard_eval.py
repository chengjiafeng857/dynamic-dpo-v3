"""Evaluation wrapper for Arena-Hard."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict

from . import __file__ as _PACKAGE_FILE
from eval.benchmark_common import (
    format_command,
    get_block_config,
    get_model_name_or_path,
    get_output_dir,
    get_pretty_name,
    resolve_existing_or_download_default_path,
    resolve_existing_path,
)


PACKAGE_DIR = Path(_PACKAGE_FILE).resolve().parent
DEFAULT_ARENAHARD_QUESTION_FILE = "questions.jsonl"
DEFAULT_ARENAHARD_QUESTION_URL = (
    "https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/resolve/main/"
    "data/arena-hard-v0.1/question.jsonl"
)


def _resolve_model_answer_path(
    config: Dict[str, Any],
    model_answer_path: str | None,
) -> Path:
    if model_answer_path is not None:
        return resolve_existing_path(config, model_answer_path, package_dir=PACKAGE_DIR)

    default_path = get_output_dir(config, "arenahard") / "model_answer.jsonl"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Could not find Arena-Hard model answers. Run arenahard-infer first or "
        "pass --model-answer."
    )


def run_arenahard_evaluation(
    config: Dict[str, Any],
    *,
    model_answer_path: str | None = None,
) -> Path:
    block_cfg = get_block_config(config, "arenahard")
    output_dir = get_output_dir(config, "arenahard")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    resolved_answer_path = _resolve_model_answer_path(config, model_answer_path)
    judge_config = resolve_existing_path(
        config,
        block_cfg.get("judge_config", "judge_config.yaml"),
        package_dir=PACKAGE_DIR,
    )
    api_config = resolve_existing_path(
        config,
        block_cfg.get("api_config", "api_config.yaml"),
        package_dir=PACKAGE_DIR,
    )
    question_file = resolve_existing_or_download_default_path(
        config,
        block_cfg.get("question_file", DEFAULT_ARENAHARD_QUESTION_FILE),
        package_dir=PACKAGE_DIR,
        default_filename=DEFAULT_ARENAHARD_QUESTION_FILE,
        download_url=DEFAULT_ARENAHARD_QUESTION_URL,
    )
    judgment_file = results_dir / "arena_hard_judgments.jsonl"

    replacements = {
        "answer_file": str(resolved_answer_path),
        "api_config": str(api_config),
        "judge_config": str(judge_config),
        "question_file": str(question_file),
        "results_dir": str(results_dir),
        "judgment_file": str(judgment_file),
        "pretty_name": get_pretty_name(config, "arenahard"),
        "model_name_or_path": get_model_name_or_path(config, "arenahard"),
        "judge_model": str(block_cfg.get("judge_model", "")),
        "baseline_model": str(block_cfg.get("baseline_model", "")),
    }
    command = format_command(
        block_cfg.get("judge_command"),
        replacements=replacements,
        field_name="arenahard.judge_command",
    )

    print(f"[ArenaHard] command={' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, check=True)

    show_result_command = block_cfg.get("show_result_command")
    if show_result_command is not None:
        resolved_show_command = format_command(
            show_result_command,
            replacements=replacements,
            field_name="arenahard.show_result_command",
        )
        print(
            "[ArenaHard] show_result_command="
            f"{' '.join(shlex.quote(part) for part in resolved_show_command)}"
        )
        subprocess.run(resolved_show_command, check=True)

    print(f"[ArenaHard] results_dir={results_dir}")
    return results_dir

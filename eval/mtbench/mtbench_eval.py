"""Evaluation wrapper for MT-Bench."""

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
DEFAULT_MTBENCH_QUESTION_FILE = "questions.jsonl"
DEFAULT_MTBENCH_QUESTION_URL = (
    "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"
    "data/mt_bench/question.jsonl"
)


def _resolve_model_answer_path(
    config: Dict[str, Any],
    model_answer_path: str | None,
) -> Path:
    if model_answer_path is not None:
        return resolve_existing_path(config, model_answer_path, package_dir=PACKAGE_DIR)

    default_path = get_output_dir(config, "mtbench") / "model_answer.jsonl"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Could not find MT-Bench model answers. Run mtbench-infer first or "
        "pass --model-answer."
    )


def run_mtbench_evaluation(
    config: Dict[str, Any],
    *,
    model_answer_path: str | None = None,
) -> Path:
    block_cfg = get_block_config(config, "mtbench")
    output_dir = get_output_dir(config, "mtbench")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    resolved_answer_path = _resolve_model_answer_path(config, model_answer_path)
    question_file = resolve_existing_or_download_default_path(
        config,
        block_cfg.get("question_file", DEFAULT_MTBENCH_QUESTION_FILE),
        package_dir=PACKAGE_DIR,
        default_filename=DEFAULT_MTBENCH_QUESTION_FILE,
        download_url=DEFAULT_MTBENCH_QUESTION_URL,
    )
    reference_answer_file = resolve_existing_path(
        config,
        block_cfg.get(
            "reference_answer_file",
            "reference_answer/gpt-4-1106-preview.jsonl",
        ),
        package_dir=PACKAGE_DIR,
    )
    judgment_file = results_dir / "mtbench_judgments.jsonl"

    replacements = {
        "answer_file": str(resolved_answer_path),
        "question_file": str(question_file),
        "reference_answer_file": str(reference_answer_file),
        "results_dir": str(results_dir),
        "judgment_file": str(judgment_file),
        "pretty_name": get_pretty_name(config, "mtbench"),
        "model_name_or_path": get_model_name_or_path(config, "mtbench"),
        "judge_model": str(block_cfg.get("judge_model", "")),
    }
    command = format_command(
        block_cfg.get("judge_command"),
        replacements=replacements,
        field_name="mtbench.judge_command",
    )

    print(f"[MTBench] command={' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, check=True)

    show_result_command = block_cfg.get("show_result_command")
    if show_result_command is not None:
        resolved_show_command = format_command(
            show_result_command,
            replacements=replacements,
            field_name="mtbench.show_result_command",
        )
        print(
            "[MTBench] show_result_command="
            f"{' '.join(shlex.quote(part) for part in resolved_show_command)}"
        )
        subprocess.run(resolved_show_command, check=True)

    print(f"[MTBench] results_dir={results_dir}")
    return results_dir

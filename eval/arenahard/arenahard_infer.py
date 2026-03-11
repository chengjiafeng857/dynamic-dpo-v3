"""Inference pipeline for Arena-Hard."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from . import __file__ as _PACKAGE_FILE
from eval.benchmark_common import (
    get_block_config,
    get_generation_config,
    get_model_name_or_path,
    get_output_dir,
    get_package_versions,
    get_pretty_name,
    read_jsonl,
    resolve_existing_path,
    write_json,
    write_jsonl,
)
from eval.model_generation import (
    generate_with_transformers,
    generate_with_vllm,
    load_render_tokenizer,
    render_chat_prompts,
)


PACKAGE_DIR = Path(_PACKAGE_FILE).resolve().parent
DEFAULT_ARENAHARD_QUESTION_FILE = "questions.jsonl"


def _summarize_text(text: str, *, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _normalize_question_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    question_id = row.get("question_id", row.get("id"))
    if question_id is None:
        return None

    turns_value = row.get("turns")
    if isinstance(turns_value, list) and turns_value:
        turns = [str(turn).strip() for turn in turns_value if str(turn).strip()]
    else:
        prompt = str(row.get("question", row.get("prompt", ""))).strip()
        turns = [prompt] if prompt else []

    if not turns:
        return None

    return {
        "question_id": question_id,
        "category": row.get("category"),
        "turns": turns,
    }


def _load_arenahard_questions(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    block_cfg = get_block_config(config, "arenahard")
    question_file = block_cfg.get("question_file", DEFAULT_ARENAHARD_QUESTION_FILE)
    question_path = resolve_existing_path(
        config,
        question_file,
        package_dir=PACKAGE_DIR,
    )
    if not question_path.exists():
        raise FileNotFoundError(
            "Could not find Arena-Hard question file. Set arenahard.question_file "
            "to a valid JSONL path."
        )

    rows = read_jsonl(question_path)
    normalized = [row for row in (_normalize_question_row(item) for item in rows) if row]
    if not normalized:
        raise ValueError("Arena-Hard question file is empty after normalization.")

    max_instances = block_cfg.get("max_instances")
    if max_instances is not None:
        normalized = normalized[: int(max_instances)]
    return normalized


def _generate_outputs(
    config: Dict[str, Any],
    questions: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], str]:
    backend = str(get_block_config(config, "arenahard").get("backend", "vllm")).lower()
    generation_cfg = get_generation_config(config, "arenahard")
    tokenizer = load_render_tokenizer(config, "arenahard")

    states = [
        {
            "question_id": question["question_id"],
            "category": question.get("category"),
            "messages": [],
            "outputs": [],
            "turns": list(question["turns"]),
            "next_turn": 0,
        }
        for question in questions
    ]

    prompt_ref = ""
    while True:
        active_states = [state for state in states if state["next_turn"] < len(state["turns"])]
        if not active_states:
            break

        conversations = []
        for state in active_states:
            user_turn = str(state["turns"][state["next_turn"]])
            conversations.append(
                list(state["messages"]) + [{"role": "user", "content": user_turn}]
            )

        prompts, current_prompt_ref = render_chat_prompts(
            config,
            "arenahard",
            tokenizer=tokenizer,
            conversations=conversations,
            package_dir=PACKAGE_DIR,
        )
        if not prompt_ref:
            prompt_ref = current_prompt_ref

        if backend == "vllm":
            outputs = generate_with_vllm(config, "arenahard", prompts, generation_cfg)
        elif backend == "transformers":
            outputs = generate_with_transformers(config, "arenahard", prompts, generation_cfg)
        else:
            raise ValueError("arenahard.backend must be 'transformers' or 'vllm'.")

        for state, conversation, output_text in zip(active_states, conversations, outputs, strict=True):
            state["outputs"].append(output_text)
            state["messages"] = list(conversation) + [{"role": "assistant", "content": output_text}]
            state["next_turn"] += 1

    model_id = get_pretty_name(config, "arenahard")
    payload = [
        {
            "question_id": state["question_id"],
            "answer_id": uuid.uuid4().hex,
            "model_id": model_id,
            "choices": [{"index": 0, "turns": list(state["outputs"])}],
            "tstamp": time.time(),
            "category": state.get("category"),
        }
        for state in states
    ]
    return payload, prompt_ref


def run_arenahard_inference(config: Dict[str, Any]) -> Path:
    questions = _load_arenahard_questions(config)
    output_dir = get_output_dir(config, "arenahard")
    model_answer_path = output_dir / "model_answer.jsonl"
    metadata_path = output_dir / "metadata.json"

    print(f"[ArenaHard] model={get_model_name_or_path(config, 'arenahard')}")
    print(f"[ArenaHard] num_questions={len(questions)}")
    print(f"[ArenaHard] sample_question={_summarize_text(str(questions[0]['turns'][0]))}")

    payload, prompt_ref = _generate_outputs(config, questions)
    write_jsonl(model_answer_path, payload)
    write_json(
        metadata_path,
        {
            "model_name_or_path": get_model_name_or_path(config, "arenahard"),
            "pretty_name": get_pretty_name(config, "arenahard"),
            "backend": str(get_block_config(config, "arenahard").get("backend", "vllm")).lower(),
            "prompt_template": prompt_ref,
            "question_file": str(
                resolve_existing_path(
                    config,
                    get_block_config(config, "arenahard").get(
                        "question_file",
                        DEFAULT_ARENAHARD_QUESTION_FILE,
                    ),
                    package_dir=PACKAGE_DIR,
                )
            ),
            "generation": get_generation_config(config, "arenahard"),
            "package_versions": get_package_versions(
                ("torch", "transformers", "vllm")
            ),
        },
    )
    print(f"[ArenaHard] wrote_model_answer={model_answer_path}")
    print(f"[ArenaHard] wrote_metadata={metadata_path}")
    return model_answer_path

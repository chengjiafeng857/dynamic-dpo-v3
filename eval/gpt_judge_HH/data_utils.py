from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset

TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")


def _strip_one_leading_newline(text: str) -> str:
    """Remove a single leading newline from an HH content block."""
    return text[1:] if text.startswith("\n") else text


def parse_hh_to_messages(text: str) -> list[dict[str, str]]:
    """Parse raw HH transcript text into normalized user/assistant messages."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("\n\nHuman:") and not text.startswith("\n\nAssistant:"):
        text = "\n\n" + text

    parts = TAG_RE.split(text)
    messages: list[dict[str, str]] = []
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        content = _strip_one_leading_newline(content).strip()
        if not content:
            continue
        role = "user" if role_tag == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def render_hh_messages(messages: list[dict[str, str]]) -> str:
    """Render normalized messages back into HH transcript format."""
    parts: list[str] = []
    for message in messages:
        role_tag = "Human" if message["role"] == "user" else "Assistant"
        parts.append(f"\n\n{role_tag}: {message['content']}")
    return "".join(parts).strip()


def render_hh_prompt(messages: list[dict[str, str]]) -> str:
    """Render HH prompt history and append the final assistant cue."""
    rendered_messages = render_hh_messages(messages)
    if not rendered_messages:
        return "Assistant:"
    return f"{rendered_messages}\n\nAssistant:"


def extract_prompt_example(
    text: str,
    *,
    single_turn_only: bool,
) -> dict[str, Any] | None:
    """Extract one generation example from a chosen HH transcript."""
    messages = parse_hh_to_messages(text)
    if len(messages) < 2:
        return None
    if messages[-1]["role"] != "assistant":
        return None
    if single_turn_only and len(messages) != 2:
        return None

    prompt_messages = messages[:-1]
    if not prompt_messages or prompt_messages[0]["role"] != "user":
        return None

    if single_turn_only:
        instruction = prompt_messages[0]["content"]
    else:
        instruction = render_hh_prompt(prompt_messages)

    return {
        "prompt_messages": prompt_messages,
        "instruction": instruction,
        "output": messages[-1]["content"],
    }


def load_hh_eval_examples(
    repo_id: str = "Anthropic/hh-rlhf",
    split: str = "test",
    data_dir: str | None = None,
    single_turn_only: bool = True,
) -> list[dict[str, Any]]:
    """Load HH rows and convert them into generation-ready prompt examples."""
    dataset = load_dataset(repo_id, data_dir=data_dir, split=split)
    examples: list[dict[str, Any]] = []
    for row in dataset:
        text = row.get("chosen") or row.get("prompt") or row.get("text")
        if text is None:
            continue
        example = extract_prompt_example(
            text,
            single_turn_only=single_turn_only,
        )
        if example is None:
            continue
        examples.append(example)
    return examples


def load_hh_chosen_outputs(
    repo_id: str = "Anthropic/hh-rlhf",
    split: str = "test",
    data_dir: str | None = None,
    single_turn_only: bool = True,
) -> list[dict[str, str]]:
    """Load HH chosen responses and emit them in judge input format."""
    dataset = load_dataset(repo_id, data_dir=data_dir, split=split)
    rows: list[dict[str, str]] = []
    for row in dataset:
        chosen = row.get("chosen")
        if chosen is None:
            continue
        example = extract_prompt_example(
            chosen,
            single_turn_only=single_turn_only,
        )
        if example is None:
            continue
        rows.append(
            {
                "instruction": str(example["instruction"]),
                "output": str(example["output"]),
                "generator": f"{repo_id}:{split}:chosen",
            }
        )
    return rows

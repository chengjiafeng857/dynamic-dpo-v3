"""UltraFeedback preference dataset processing utilities."""

from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset

from .templates import (
    ensure_tokenizer_chat_template,
    get_assistant_generation_suffix,
)


def _normalize_text(value: Any) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _coerce_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list) or not messages:
        return None

    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            return None
        content = _normalize_text(message.get("content", "")).strip()
        if not content:
            return None
        normalized.append({"role": role.strip(), "content": content})

    return normalized if normalized else None


def _normalize_ultrafeedback_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt_text = _normalize_text(row.get("prompt", "")).strip()
    if not prompt_text:
        return None

    chosen_messages = _coerce_messages(row.get("chosen"))
    rejected_messages = _coerce_messages(row.get("rejected"))
    if chosen_messages is None or rejected_messages is None:
        return None

    if chosen_messages[-1]["role"] != "assistant":
        return None
    if rejected_messages[-1]["role"] != "assistant":
        return None

    prompt_messages = chosen_messages[:-1]
    if not prompt_messages:
        return None
    if rejected_messages[:-1] != prompt_messages:
        return None

    chosen_response = chosen_messages[-1]["content"].strip()
    rejected_response = rejected_messages[-1]["content"].strip()
    if not chosen_response or not rejected_response:
        return None

    return {
        "prompt_messages": prompt_messages,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response,
    }


def _ensure_generation_prompt(
    prompt_text: str, *, assistant_generation_suffix: Optional[str]
) -> str:
    if not assistant_generation_suffix:
        return prompt_text
    trimmed = prompt_text.rstrip()
    if trimmed.endswith(assistant_generation_suffix.rstrip()):
        return prompt_text
    return f"{prompt_text}{assistant_generation_suffix}"


def _render_completion_from_messages(
    prompt_messages: List[Dict[str, str]],
    response: str,
    *,
    tokenizer: Any,
    prompt_text: str,
) -> Optional[str]:
    full_messages = prompt_messages + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if full_text.startswith(prompt_text):
        completion = full_text[len(prompt_text):]
    else:
        completion = response

    completion = completion.strip()
    return completion if completion else None


def build_ultrafeedback_preference_dataset(
    ds: Iterable[Dict[str, Any]],
    tokenizer: Any,
    *,
    model_name: Optional[str] = None,
    chat_template_name: Optional[str] = None,
) -> Dataset:
    """Convert UltraFeedback rows into {prompt, chosen, rejected} for DPO."""
    resolved_template_name = ensure_tokenizer_chat_template(
        tokenizer,
        model_name=model_name or "",
        configured_name=chat_template_name,
    )

    assistant_generation_suffix = None
    if resolved_template_name is not None:
        assistant_generation_suffix = get_assistant_generation_suffix(
            resolved_template_name
        )

    rows: List[Dict[str, str]] = []
    for row in ds:
        normalized = _normalize_ultrafeedback_row(row)
        if normalized is None:
            continue

        prompt_messages = normalized["prompt_messages"]
        chosen_response = normalized["chosen_response"]
        rejected_response = normalized["rejected_response"]

        prompt_rendered = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not prompt_rendered:
            continue

        prompt_rendered = _ensure_generation_prompt(
            prompt_rendered,
            assistant_generation_suffix=assistant_generation_suffix,
        )

        chosen_rendered = _render_completion_from_messages(
            prompt_messages,
            chosen_response,
            tokenizer=tokenizer,
            prompt_text=prompt_rendered,
        )
        rejected_rendered = _render_completion_from_messages(
            prompt_messages,
            rejected_response,
            tokenizer=tokenizer,
            prompt_text=prompt_rendered,
        )
        if not chosen_rendered or not rejected_rendered:
            continue

        rows.append(
            {
                "prompt": prompt_rendered,
                "chosen": chosen_rendered,
                "rejected": rejected_rendered,
            }
        )

    return Dataset.from_list(rows)

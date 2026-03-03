"""UltraChat dataset processing utilities for SFT."""

from typing import Any, Dict, List, Optional

from datasets import Dataset


def _normalize_text(text: Any) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _coerce_chatml_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list):
        return None

    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        content = _normalize_text(msg.get("content", "")).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    if not cleaned:
        return None
    if not any(message["role"] == "assistant" for message in cleaned):
        return None
    return cleaned


def build_ultrachat_sft_dataset(ds, *, completion_only_loss: bool = False) -> Dataset:
    """Normalize UltraChat rows to conversational SFT format."""
    rows = []
    for row in ds:
        messages = row.get("messages") if isinstance(row, dict) else None
        if messages is None:
            messages = row["messages"] if "messages" in row else None
        chat = _coerce_chatml_messages(messages)
        if chat is None:
            continue

        if not completion_only_loss:
            rows.append({"messages": chat})
            continue

        for idx, message in enumerate(chat):
            if message["role"] != "assistant":
                continue

            prompt_messages = chat[:idx]
            if not prompt_messages or prompt_messages[-1]["role"] != "user":
                continue

            rows.append(
                {
                    "prompt": prompt_messages,
                    "completion": [message],
                }
            )
    return Dataset.from_list(rows)

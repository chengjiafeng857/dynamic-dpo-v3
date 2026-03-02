"""SFT dataset processing utilities."""

from typing import Any, Dict, List, Optional

from datasets import Dataset
from transformers import AutoTokenizer

from .templates import LLAMA3_CHAT_TEMPLATE, parse_hh_to_messages


def load_tokenizer(
    model_name: str,
    *,
    padding_side: str = "left",
    add_chat_template: bool = True,
    use_fast: bool = True,
) -> AutoTokenizer:
    """Load tokenizer with padding and template defaults for chat models.
    
    Args:
        model_name: HuggingFace model name or path.
        padding_side: Side to pad sequences ('left' or 'right').
        add_chat_template: Whether to add Llama3 chat template if missing.
        use_fast: Whether to use the fast tokenizer.
        
    Returns:
        Configured AutoTokenizer instance.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if add_chat_template and not tok.chat_template:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    return tok


def build_sft_dataset(ds, tokenizer=None) -> Dataset:
    """Convert HH dataset rows into prompt-completion format for SFT.

    For each HH `chosen` conversation, we train on the final assistant turn:
    - prompt: full history up to (but excluding) the final assistant turn
    - completion: the final assistant turn

    Args:
        ds: Input dataset with 'chosen' field in HH format.
        tokenizer: Optional tokenizer (unused, kept for API compatibility).

    Returns:
        Dataset with 'prompt' and 'completion' fields for completion-only SFT.
    """
    rows = []
    for row in ds:
        text = row.get("chosen") if isinstance(row, dict) else None
        if text is None:
            text = row["chosen"] if "chosen" in row else None
        if text is None:
            continue

        messages = parse_hh_to_messages(text)
        if len(messages) < 2:
            continue
        if messages[-1]["role"] != "assistant":
            continue

        prompt_messages = messages[:-1]
        completion_messages = [messages[-1]]

        if not prompt_messages or prompt_messages[-1]["role"] != "user":
            continue

        rows.append(
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
            }
        )
    return Dataset.from_list(rows)


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
    if not any(m["role"] == "assistant" for m in cleaned):
        return None
    return cleaned


def build_ultrachat_sft_dataset(ds, *, completion_only_loss: bool = False) -> Dataset:
    """Normalize UltraChat rows to conversational SFT format.

    Args:
        ds: Dataset containing a `messages` column in ChatML format.
        completion_only_loss: Whether to emit prompt/completion rows for each
            assistant turn instead of full `messages` rows.

    Returns:
        Dataset with either:
        - `messages` rows for full-sequence CLM SFT, or
        - `prompt`/`completion` rows for completion-only SFT.
    """
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

        # For completion-only mode, train on each assistant turn conditioned on
        # all prior messages in the same conversation.
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

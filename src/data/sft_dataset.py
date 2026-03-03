"""SFT dataset processing utilities."""

from datasets import Dataset
from transformers import AutoTokenizer

from .templates import ensure_tokenizer_chat_template, parse_hh_to_messages


def load_tokenizer(
    model_name: str,
    *,
    padding_side: str = "left",
    chat_template_name: str | None = None,
    add_chat_template: bool = True,
    use_fast: bool = True,
) -> AutoTokenizer:
    """Load tokenizer with padding and template defaults for chat models.
    
    Args:
        model_name: HuggingFace model name or path.
        padding_side: Side to pad sequences ('left' or 'right').
        chat_template_name: Optional configured chat template name.
        add_chat_template: Whether to add a local chat template fallback if needed.
        use_fast: Whether to use the fast tokenizer.
        
    Returns:
        Configured AutoTokenizer instance.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if add_chat_template:
        ensure_tokenizer_chat_template(
            tok,
            model_name=model_name,
            configured_name=chat_template_name,
        )
    return tok


def build_hh_sft_dataset(ds, tokenizer=None) -> Dataset:
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

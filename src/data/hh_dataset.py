"""HH (Anthropic Helpful-Harmless) dataset processing utilities."""

from datasets import load_dataset, Dataset
from typing import Any, Dict, Iterable, List, Optional

from .templates import LLAMA3_CHAT_TEMPLATE, parse_hh_to_messages


ASSISTANT_TAG = "\n\nAssistant:"
HUMAN_TAG = "\n\nHuman:"
LLAMA3_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def strip_one_leading_newline(s: str) -> str:
    """Remove a single leading newline to normalize HH blocks."""
    return s[1:] if s.startswith("\n") else s


def split_prompt_and_response(input_text: str) -> tuple[str, str]:
    """Split HH format text into prompt and response.
    
    HH format: multi-turn text containing many "\\n\\nAssistant:".
    We take the LAST Assistant tag as the start of the final assistant response.

    Args:
        input_text: Raw HH format text.
        
    Returns:
        Tuple of (prompt, response) where prompt includes the final Assistant tag.
        
    Raises:
        ValueError: If no Assistant tag is found.
    """
    input_text = str(input_text).replace("\r\n", "\n").replace("\r", "\n")
    index = input_text.rfind(ASSISTANT_TAG)
    if index < 0:
        raise ValueError("No '\\n\\nAssistant:' tag found in HH input.")
    prompt = input_text[:index + len(ASSISTANT_TAG)]
    response = input_text[index + len(ASSISTANT_TAG):]
    response = strip_one_leading_newline(response)
    return prompt, response


def convert_to_triples(
    chosen_text: str, rejected_text: str
) -> Optional[Dict[str, str]]:
    """Convert one HH row into an explicit triplet: {prompt, chosen, rejected}."""
    chosen_prompt, chosen_response = split_prompt_and_response(chosen_text)

    if not rejected_text.startswith(chosen_prompt):
        return None

    rejected_response = strip_one_leading_newline(rejected_text[len(chosen_prompt):])

    if len(chosen_prompt.strip()) == 0:
        return None
    if len(chosen_response.strip()) == 0 or len(rejected_response.strip()) == 0:
        return None

    return {
        "prompt": chosen_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def build_HH_dataset(ds) -> Dataset:
    """Process entire dataset into HH triplets format."""
    hh_ds_raw = []
    for idx, row in enumerate(ds):
        output = convert_to_triples(
            chosen_text=row["chosen"], rejected_text=row["rejected"]
        )
        if output is not None:
            hh_ds_raw.append(output)
    return Dataset.from_list(hh_ds_raw)


def _normalize_text(text: Any) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _coerce_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list):
        return None
    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _normalize_text(msg.get("content", "")).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned if cleaned else None


def _messages_to_hh_prompt(messages: List[Dict[str, str]]) -> Optional[str]:
    if not messages or messages[-1]["role"] != "user":
        return None
    parts: List[str] = []
    for msg in messages:
        tag = HUMAN_TAG if msg["role"] == "user" else ASSISTANT_TAG
        parts.append(f"{tag} {msg['content']}")
    prompt = "".join(parts)
    if not prompt.endswith(ASSISTANT_TAG):
        prompt = f"{prompt}{ASSISTANT_TAG}"
    return prompt


def _extract_response_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = _normalize_text(value).strip()
        return text if text else None
    if isinstance(value, dict):
        content = _normalize_text(value.get("content", "")).strip()
        return content if content else None
    if isinstance(value, list):
        parts: List[str] = []
        for msg in value:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role is not None and role != "assistant":
                continue
            content = _normalize_text(msg.get("content", "")).strip()
            if content:
                parts.append(content)
        if parts:
            return "\n\n".join(parts)
    return None


def build_rollout_dataset(ds: Iterable[Dict[str, Any]]) -> Dataset:
    """Build dataset from rollout generation outputs."""
    rollout_ds_raw: List[Dict[str, str]] = []
    for row in ds:
        prompt_messages = _coerce_messages(row.get("prompt_messages"))
        if prompt_messages is None:
            continue
        prompt_text = _messages_to_hh_prompt(prompt_messages)
        if not prompt_text:
            continue
        chosen_text = _extract_response_text(row.get("chosen"))
        rejected_text = _extract_response_text(row.get("rejected"))
        if not chosen_text or not rejected_text:
            continue
        rollout_ds_raw.append(
            {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        )
    return Dataset.from_list(rollout_ds_raw)


def load_generated_hf_dataset(dataset_name: str, *, subset: str = "train") -> Dataset:
    """Load a generated dataset from HuggingFace."""
    raw_ds = load_dataset(dataset_name, split=subset)
    return build_rollout_dataset(raw_ds)


def load_generated_dataset_from_config(config: Dict[str, Any]) -> Dataset:
    """Load generated dataset using configuration dictionary."""
    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("Missing dataset.dataset_name in config.")
    subset = dataset_cfg.get("subset", "train")
    return load_generated_hf_dataset(dataset_name, subset=subset)


def _ensure_chat_template(tokenizer: Any) -> None:
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE


def _ensure_generation_prompt(prompt_text: str, tokenizer: Any) -> str:
    trimmed = prompt_text.rstrip()
    if trimmed.endswith(LLAMA3_ASSISTANT_HEADER.rstrip()):
        return prompt_text
    template = getattr(tokenizer, "chat_template", "") or ""
    if "<|start_header_id|>" in prompt_text or "start_header_id" in template:
        return f"{prompt_text}{LLAMA3_ASSISTANT_HEADER}"
    return prompt_text


def _render_response_with_chat_template(
    messages: List[Dict[str, str]],
    response: str,
    *,
    tokenizer: Any,
    prompt_text: str,
) -> Optional[str]:
    response = _normalize_text(response).strip()
    if not response:
        return None
    full_messages = messages + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    if full_text.startswith(prompt_text):
        rendered = full_text[len(prompt_text):]
    else:
        rendered = response
    rendered = rendered.strip()
    return rendered if rendered else None


def apply_chat_template_to_dataset(ds: Dataset, tokenizer: Any) -> Dataset:
    """Apply chat template to dataset prompts and responses."""
    _ensure_chat_template(tokenizer)
    rows: List[Dict[str, str]] = []
    for row in ds:
        prompt_text = _normalize_text(row.get("prompt", "")).strip()
        chosen_text = row.get("chosen", "")
        rejected_text = row.get("rejected", "")
        if not prompt_text:
            continue

        messages = parse_hh_to_messages(prompt_text)
        if not messages or messages[-1]["role"] != "user":
            continue

        prompt_rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not prompt_rendered:
            continue
        prompt_rendered = _ensure_generation_prompt(prompt_rendered, tokenizer)

        chosen_rendered = _render_response_with_chat_template(
            messages, str(chosen_text), tokenizer=tokenizer, prompt_text=prompt_rendered
        )
        rejected_rendered = _render_response_with_chat_template(
            messages, str(rejected_text), tokenizer=tokenizer, prompt_text=prompt_rendered
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

"""Chat templates and HH format parsing utilities."""

import re
from typing import Dict, List


TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")

# Llama 3 chat template
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = message['content'] %}"
    "{% if loop.index0 == 0 %}"
    "{{ '<|begin_of_text|>' }}"
    "{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + content | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def strip_one_leading_newline(text: str) -> str:
    """Remove a single leading newline to normalize HH blocks."""
    return text[1:] if text.startswith("\n") else text


def parse_hh_to_messages(text: str) -> List[Dict[str, str]]:
    """Parse Anthropic HH multi-turn text into [{role, content}, ...].
    
    Ensures content is trimmed and skips empty blocks.
    
    Args:
        text: Raw HH format text with Human/Assistant turns.
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("\n\nHuman:") and not text.startswith("\n\nAssistant:"):
        text = "\n\n" + text

    parts = TAG_RE.split(text)
    messages = []
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        content = strip_one_leading_newline(content).strip()
        if not content:
            continue
        role = "user" if role_tag == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages

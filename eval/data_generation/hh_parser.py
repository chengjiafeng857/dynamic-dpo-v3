from __future__ import annotations

from typing import List, Optional, Tuple

import re

from util import parse_hh_to_messages

RAW_ROLE_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):")


def clean_content(text: str) -> str:
    """Normalize line endings and trim surrounding whitespace."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def extract_prompt_and_reference(text: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    Split HH text into a prompt (ending on a user turn) and a final assistant
    reference string. Returns (prompt_messages, reference_response).
    """
    messages = parse_hh_to_messages(text)
    if not messages:
        return None, None

    reference_response = None
    if messages[-1]["role"] == "assistant":
        reference_response = messages[-1]["content"]
        messages = messages[:-1]

    if not messages or messages[-1]["role"] != "user":
        return None, reference_response

    return messages, reference_response


def messages_have_raw_role_tags(messages: List[dict]) -> bool:
    """Return True if any content still contains raw HH role headers."""
    for msg in messages:
        content = msg.get("content", "")
        if RAW_ROLE_RE.search(content):
            return True
    return False

"""Chat templates and HH format parsing utilities."""

import re
from typing import Any, Dict, List, Optional, TypedDict


TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")

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

QWEN3_CHAT_TEMPLATE = r'''{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}'''


class ChatTemplateSpec(TypedDict):
    template: str
    assistant_generation_suffix: str


CHAT_TEMPLATES: Dict[str, ChatTemplateSpec] = {
    "llama3": {
        "template": LLAMA3_CHAT_TEMPLATE,
        "assistant_generation_suffix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen3": {
        "template": QWEN3_CHAT_TEMPLATE,
        "assistant_generation_suffix": "<|im_start|>assistant\n",
    },
}


def _normalize_template_name(template_name: Optional[str]) -> Optional[str]:
    if template_name is None:
        return None
    normalized = str(template_name).strip().lower()
    if not normalized:
        return None
    if normalized not in CHAT_TEMPLATES:
        supported = ", ".join(sorted(CHAT_TEMPLATES))
        raise ValueError(
            f"Unsupported chat template '{template_name}'. Expected one of: {supported}."
        )
    return normalized


def resolve_chat_template_name(
    model_name: str, configured_name: Optional[str] = None
) -> Optional[str]:
    """Resolve the configured or inferred chat template name."""
    normalized = _normalize_template_name(configured_name)
    if normalized is not None:
        return normalized

    lowered = str(model_name).strip().lower()
    if not lowered:
        return None
    if "qwen/qwen3" in lowered or "qwen3" in lowered:
        return "qwen3"
    if "llama-3" in lowered or "llama3" in lowered:
        return "llama3"
    return None


def get_chat_template(template_name: str) -> str:
    """Return the template string for a supported chat template name."""
    normalized = _normalize_template_name(template_name)
    if normalized is None:
        raise ValueError("chat template name must be provided.")
    return CHAT_TEMPLATES[normalized]["template"]


def get_assistant_generation_suffix(template_name: str) -> str:
    """Return the assistant generation suffix for a supported template name."""
    normalized = _normalize_template_name(template_name)
    if normalized is None:
        raise ValueError("chat template name must be provided.")
    return CHAT_TEMPLATES[normalized]["assistant_generation_suffix"]


def ensure_tokenizer_chat_template(
    tokenizer: Any, *, model_name: str, configured_name: Optional[str] = None
) -> Optional[str]:
    """Ensure the tokenizer has a chat template, preferring any native template."""
    if getattr(tokenizer, "chat_template", None):
        return None

    template_name = resolve_chat_template_name(model_name, configured_name)
    if template_name is not None:
        tokenizer.chat_template = get_chat_template(template_name)
        return template_name

    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    return "llama3"


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

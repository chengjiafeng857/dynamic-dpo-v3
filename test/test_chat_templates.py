"""Unit tests for chat template selection helpers."""

import unittest

from src.data.templates import (
    ensure_tokenizer_chat_template,
    get_chat_template,
    resolve_chat_template_name,
)


class _DummyTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


class ChatTemplateSelectionTest(unittest.TestCase):
    def test_resolve_chat_template_name_honors_explicit_config(self):
        self.assertEqual(resolve_chat_template_name("custom/model", "llama3"), "llama3")
        self.assertEqual(resolve_chat_template_name("custom/model", "qwen3"), "qwen3")

    def test_resolve_chat_template_name_infers_from_model_name(self):
        self.assertEqual(
            resolve_chat_template_name("meta-llama/Llama-3.2-1B-Instruct"),
            "llama3",
        )
        self.assertEqual(resolve_chat_template_name("Qwen/Qwen3-8B"), "qwen3")

    def test_ensure_tokenizer_chat_template_preserves_native_template(self):
        tokenizer = _DummyTokenizer(chat_template="native-template")

        resolved = ensure_tokenizer_chat_template(
            tokenizer,
            model_name="custom/model",
        )

        self.assertIsNone(resolved)
        self.assertEqual(tokenizer.chat_template, "native-template")

    def test_ensure_tokenizer_chat_template_uses_fallback_when_needed(self):
        tokenizer = _DummyTokenizer()

        resolved = ensure_tokenizer_chat_template(
            tokenizer,
            model_name="custom/model",
        )

        self.assertEqual(resolved, "llama3")
        self.assertEqual(tokenizer.chat_template, get_chat_template("llama3"))

    def test_ensure_tokenizer_chat_template_sets_qwen_template(self):
        tokenizer = _DummyTokenizer()

        resolved = ensure_tokenizer_chat_template(
            tokenizer,
            model_name="Qwen/Qwen3-8B",
        )

        self.assertEqual(resolved, "qwen3")
        self.assertEqual(tokenizer.chat_template, get_chat_template("qwen3"))


if __name__ == "__main__":
    unittest.main()

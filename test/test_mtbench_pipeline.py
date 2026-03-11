"""Unit tests for MT-Bench inference and evaluation helpers."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import main_mtbench_eval, main_mtbench_infer
from eval.mtbench.mtbench_eval import run_mtbench_evaluation
from eval.mtbench.mtbench_infer import _load_mtbench_questions, run_mtbench_inference
from eval.model_generation import _load_tokenizer, _prepare_tokenizer_path, load_render_tokenizer


def _base_config(output_dir: str, question_file: str, reference_answer_file: str) -> dict:
    return {
        "policy_name": "dummy/local-model",
        "precision": "bf16",
        "mtbench": {
            "pretty_name": "dummy-mtbench-model",
            "model_name_or_path": "dummy/local-model",
            "backend": "transformers",
            "use_custom_chat_template": False,
            "output_dir": output_dir,
            "question_file": question_file,
            "reference_answer_file": reference_answer_file,
            "judge_model": "gpt-4o-mini",
            "judge_command": [
                "mtbench-judge",
                "--answers",
                "{answer_file}",
                "--questions",
                "{question_file}",
                "--reference",
                "{reference_answer_file}",
                "--output",
                "{judgment_file}",
            ],
            "show_result_command": [
                "mtbench-show",
                "--input",
                "{judgment_file}",
                "--model",
                "{pretty_name}",
            ],
            "generation": {
                "batch_size": 2,
                "do_sample": False,
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop_token_ids": [128001, 128009],
            },
            "transformers": {
                "device": "cpu",
                "trust_remote_code": False,
            },
        },
    }


class _DummyTokenizer:
    chat_template = "dummy"
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise AssertionError("Expected tokenize=False for prompt rendering.")
        if not add_generation_prompt:
            raise AssertionError("Expected add_generation_prompt=True.")
        return " || ".join(f"{item['role']}:{item['content']}" for item in messages)


class MTBenchPipelineTest(unittest.TestCase):
    def test_prepare_tokenizer_path_rewrites_tokenizers_backend_class(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "source"
            source_dir.mkdir()
            (source_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (source_dir / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "TokenizersBackend",
                        "bos_token": "<|begin_of_text|>",
                        "eos_token": "<|end_of_text|>",
                    }
                ),
                encoding="utf-8",
            )

            with patch("eval.model_generation.Path.home", return_value=Path(tmp_dir)):
                sanitized_path = Path(_prepare_tokenizer_path(str(source_dir)))

            payload = json.loads((sanitized_path / "tokenizer_config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["tokenizer_class"], "PreTrainedTokenizerFast")

    def test_load_tokenizer_retries_from_sanitized_path_for_tokenizers_backend(self):
        tokenizer = object()
        with (
            patch(
                "eval.model_generation.AutoTokenizer.from_pretrained",
                side_effect=[
                    ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported."),
                    tokenizer,
                ],
            ) as mock_from_pretrained,
            patch(
                "eval.model_generation._prepare_tokenizer_path",
                return_value="/tmp/sanitized-tokenizer",
            ) as mock_prepare,
        ):
            resolved = _load_tokenizer("W-61/ultrafeedback-llama3-8b-margin-dpo", trust_remote_code=False)

        self.assertIs(resolved, tokenizer)
        mock_prepare.assert_called_once_with("W-61/ultrafeedback-llama3-8b-margin-dpo")
        self.assertEqual(mock_from_pretrained.call_args_list[1].args[0], "/tmp/sanitized-tokenizer")

    def test_load_render_tokenizer_uses_shared_tokenizer_loader(self):
        config = _base_config("/tmp/out", "questions.jsonl", "reference.jsonl")
        with patch(
            "eval.model_generation._load_tokenizer",
            return_value=_DummyTokenizer(),
        ) as mock_load:
            tokenizer = load_render_tokenizer(config, "mtbench")

        self.assertIsInstance(tokenizer, _DummyTokenizer)
        mock_load.assert_called_once_with("dummy/local-model", trust_remote_code=False)

    def test_load_questions_downloads_default_question_file_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir, "questions.jsonl", "reference.jsonl")
            question_payload = (
                '{"question_id":"q1","category":"writing","turns":["Draft a poem."]}\n'
            )

            with patch("eval.mtbench.mtbench_infer.PACKAGE_DIR", Path(tmp_dir)), patch(
                "eval.benchmark_common.urllib.request.urlopen",
                return_value=io.BytesIO(question_payload.encode("utf-8")),
            ):
                questions = _load_mtbench_questions(config)

            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0]["question_id"], "q1")
            self.assertTrue((Path(tmp_dir) / "questions.jsonl").exists())

    def test_inference_writes_model_answer_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            question_file = Path(tmp_dir) / "questions.jsonl"
            question_file.write_text("{}\n", encoding="utf-8")
            reference_answer_file = Path(tmp_dir) / "reference.jsonl"
            reference_answer_file.write_text("{}\n", encoding="utf-8")
            config = _base_config(tmp_dir, str(question_file), str(reference_answer_file))
            questions = [
                {"question_id": "q1", "category": "coding", "turns": ["Explain recursion"]},
                {"question_id": "q2", "category": "math", "turns": ["Step one", "Step two"]},
            ]

            with (
                patch(
                    "eval.mtbench.mtbench_infer._load_mtbench_questions",
                    return_value=questions,
                ),
                patch(
                    "eval.mtbench.mtbench_infer.load_render_tokenizer",
                    return_value=_DummyTokenizer(),
                ),
                patch(
                    "eval.mtbench.mtbench_infer.generate_with_transformers",
                    side_effect=[["answer-1", "answer-2"], ["answer-3"]],
                ),
            ):
                model_answer_path = run_mtbench_inference(config)

            rows = model_answer_path.read_text(encoding="utf-8").splitlines()
            payload = [json.loads(line) for line in rows]
            metadata = json.loads(
                (Path(tmp_dir) / "metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["choices"][0]["turns"], ["answer-1"])
            self.assertEqual(payload[1]["choices"][0]["turns"], ["answer-2", "answer-3"])
            self.assertEqual(metadata["prompt_template"], "model_default")
            self.assertEqual(metadata["backend"], "transformers")

    def test_evaluation_formats_judge_and_show_commands(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            question_file = Path(tmp_dir) / "questions.jsonl"
            question_file.write_text("{}\n", encoding="utf-8")
            reference_answer_file = Path(tmp_dir) / "reference.jsonl"
            reference_answer_file.write_text("{}\n", encoding="utf-8")
            output_dir = Path(tmp_dir) / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            answer_file = output_dir / "model_answer.jsonl"
            answer_file.write_text("{}\n", encoding="utf-8")

            config = _base_config(
                str(output_dir),
                str(question_file),
                str(reference_answer_file),
            )

            with patch("eval.mtbench.mtbench_eval.subprocess.run") as mock_run:
                results_dir = run_mtbench_evaluation(
                    config,
                    model_answer_path=str(answer_file),
                )

            self.assertEqual(results_dir, output_dir / "results")
            self.assertEqual(mock_run.call_count, 2)
            first_command = mock_run.call_args_list[0].args[0]
            second_command = mock_run.call_args_list[1].args[0]
            self.assertEqual(first_command[0], "mtbench-judge")
            self.assertIn(str(answer_file), first_command)
            self.assertIn(str(question_file), first_command)
            self.assertIn(str(reference_answer_file), first_command)
            self.assertEqual(second_command[0], "mtbench-show")

    def test_cli_entrypoints_forward_config(self):
        with (
            patch("src.cli.load_yaml", return_value={"mtbench": {}}),
            patch(
                "eval.mtbench.mtbench_infer.run_mtbench_inference",
                return_value=Path("/tmp/model_answer.jsonl"),
            ) as mock_infer,
            patch.object(
                sys,
                "argv",
                ["mtbench-infer", "--config", "eval/mtbench/config_mtbench.yaml"],
            ),
        ):
            main_mtbench_infer()

        infer_config = mock_infer.call_args.args[0]
        self.assertEqual(infer_config["_config_path"], "eval/mtbench/config_mtbench.yaml")

        with (
            patch("src.cli.load_yaml", return_value={"mtbench": {}}),
            patch(
                "eval.mtbench.mtbench_eval.run_mtbench_evaluation",
                return_value=Path("/tmp/results"),
            ) as mock_eval,
            patch.object(
                sys,
                "argv",
                [
                    "mtbench-eval",
                    "--config",
                    "eval/mtbench/config_mtbench.yaml",
                    "--model-answer",
                    "/tmp/model_answer.jsonl",
                ],
            ),
        ):
            main_mtbench_eval()

        eval_config = mock_eval.call_args.args[0]
        self.assertEqual(eval_config["_config_path"], "eval/mtbench/config_mtbench.yaml")
        self.assertEqual(
            mock_eval.call_args.kwargs["model_answer_path"],
            "/tmp/model_answer.jsonl",
        )

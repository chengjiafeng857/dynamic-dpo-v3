"""Unit tests for MT-Bench inference and evaluation helpers."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import main_mtbench_eval, main_mtbench_infer
from eval.mtbench.mtbench_eval import run_mtbench_evaluation
from eval.mtbench.mtbench_infer import run_mtbench_inference


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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise AssertionError("Expected tokenize=False for prompt rendering.")
        if not add_generation_prompt:
            raise AssertionError("Expected add_generation_prompt=True.")
        return " || ".join(f"{item['role']}:{item['content']}" for item in messages)


class MTBenchPipelineTest(unittest.TestCase):
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

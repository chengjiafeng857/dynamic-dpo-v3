"""Unit tests for Arena-Hard inference and evaluation helpers."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import main_arenahard_eval, main_arenahard_infer
from eval.arenahard.arenahard_eval import run_arenahard_evaluation
from eval.arenahard.arenahard_infer import _load_arenahard_questions, run_arenahard_inference


def _base_config(output_dir: str, question_file: str) -> dict:
    return {
        "policy_name": "dummy/local-model",
        "precision": "bf16",
        "arenahard": {
            "pretty_name": "dummy-arenahard-model",
            "model_name_or_path": "dummy/local-model",
            "backend": "transformers",
            "use_custom_chat_template": False,
            "output_dir": output_dir,
            "question_file": question_file,
            "judge_model": "judge-model",
            "baseline_model": "baseline-model",
            "judge_config": "judge_config.yaml",
            "api_config": "api_config.yaml",
            "judge_command": [
                "arena-hard",
                "--answers",
                "{answer_file}",
                "--judge-config",
                "{judge_config}",
                "--api-config",
                "{api_config}",
                "--questions",
                "{question_file}",
                "--output",
                "{judgment_file}",
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


class ArenaHardPipelineTest(unittest.TestCase):
    def test_load_questions_downloads_default_question_file_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir, "questions.jsonl")
            question_payload = (
                '{"uid":"q1","category":"arena-hard-v0.1","prompt":"Explain caching."}\n'
            )

            with patch("eval.arenahard.arenahard_infer.PACKAGE_DIR", Path(tmp_dir)), patch(
                "eval.benchmark_common.urllib.request.urlopen",
                return_value=io.BytesIO(question_payload.encode("utf-8")),
            ):
                questions = _load_arenahard_questions(config)

            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0]["question_id"], "q1")
            self.assertEqual(questions[0]["turns"], ["Explain caching."])
            self.assertTrue((Path(tmp_dir) / "questions.jsonl").exists())

    def test_inference_writes_model_answer_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            question_file = Path(tmp_dir) / "questions.jsonl"
            question_file.write_text("{}\n", encoding="utf-8")
            config = _base_config(tmp_dir, str(question_file))
            questions = [
                {"question_id": "q1", "category": "general", "turns": ["Hello there"]},
                {"question_id": "q2", "category": "math", "turns": ["First turn", "Second turn"]},
            ]

            with (
                patch(
                    "eval.arenahard.arenahard_infer._load_arenahard_questions",
                    return_value=questions,
                ),
                patch(
                    "eval.arenahard.arenahard_infer.load_render_tokenizer",
                    return_value=_DummyTokenizer(),
                ),
                patch(
                    "eval.arenahard.arenahard_infer.generate_with_transformers",
                    side_effect=[["answer-1", "answer-2"], ["answer-3"]],
                ),
            ):
                model_answer_path = run_arenahard_inference(config)

            rows = model_answer_path.read_text(encoding="utf-8").splitlines()
            payload = [json.loads(line) for line in rows]
            metadata = json.loads(
                (Path(tmp_dir) / "metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["question_id"], "q1")
            self.assertEqual(payload[0]["choices"][0]["turns"], ["answer-1"])
            self.assertEqual(payload[1]["choices"][0]["turns"], ["answer-2", "answer-3"])
            self.assertEqual(metadata["backend"], "transformers")
            self.assertEqual(metadata["prompt_template"], "model_default")

    def test_evaluation_formats_judge_command(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            question_file = Path(tmp_dir) / "questions.jsonl"
            question_file.write_text("{}\n", encoding="utf-8")
            judge_config = Path(tmp_dir) / "judge_config.yaml"
            judge_config.write_text("judge: test\n", encoding="utf-8")
            api_config = Path(tmp_dir) / "api_config.yaml"
            api_config.write_text("api: test\n", encoding="utf-8")
            output_dir = Path(tmp_dir) / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            answer_file = output_dir / "model_answer.jsonl"
            answer_file.write_text("{}\n", encoding="utf-8")

            config = _base_config(str(output_dir), str(question_file))
            config["arenahard"]["judge_config"] = str(judge_config)
            config["arenahard"]["api_config"] = str(api_config)

            with patch("eval.arenahard.arenahard_eval.subprocess.run") as mock_run:
                results_dir = run_arenahard_evaluation(
                    config,
                    model_answer_path=str(answer_file),
                )

            self.assertEqual(results_dir, output_dir / "results")
            mock_run.assert_called_once()
            command = mock_run.call_args.args[0]
            self.assertEqual(command[0], "arena-hard")
            self.assertIn(str(answer_file), command)
            self.assertIn(str(judge_config), command)
            self.assertIn(str(api_config), command)
            self.assertIn(str(question_file), command)

    def test_cli_entrypoints_forward_config(self):
        with (
            patch("src.cli.load_yaml", return_value={"arenahard": {}}),
            patch(
                "eval.arenahard.arenahard_infer.run_arenahard_inference",
                return_value=Path("/tmp/model_answer.jsonl"),
            ) as mock_infer,
            patch(
                "eval.arenahard.arenahard_eval.run_arenahard_evaluation",
                return_value=Path("/tmp/results"),
            ) as mock_eval,
            patch.object(
                sys,
                "argv",
                ["arenahard-infer", "--config", "eval/arenahard/config_arenahard.yaml"],
            ),
        ):
            main_arenahard_infer()

        infer_config = mock_infer.call_args.args[0]
        self.assertEqual(infer_config["_config_path"], "eval/arenahard/config_arenahard.yaml")

        with (
            patch("src.cli.load_yaml", return_value={"arenahard": {}}),
            patch(
                "eval.arenahard.arenahard_eval.run_arenahard_evaluation",
                return_value=Path("/tmp/results"),
            ) as mock_eval,
            patch.object(
                sys,
                "argv",
                [
                    "arenahard-eval",
                    "--config",
                    "eval/arenahard/config_arenahard.yaml",
                    "--model-answer",
                    "/tmp/model_answer.jsonl",
                ],
            ),
        ):
            main_arenahard_eval()

        eval_config = mock_eval.call_args.args[0]
        self.assertEqual(eval_config["_config_path"], "eval/arenahard/config_arenahard.yaml")
        self.assertEqual(
            mock_eval.call_args.kwargs["model_answer_path"],
            "/tmp/model_answer.jsonl",
        )

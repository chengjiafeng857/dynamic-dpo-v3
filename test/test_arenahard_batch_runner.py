"""Unit tests for Arena-Hard batch orchestration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.arenahard.batch_runner import build_run_matrix, main, run_arenahard_batch
from eval.benchmark_common import get_output_dir


def _base_config() -> dict:
    return {
        "policy_name": "placeholder/model",
        "precision": "bf16",
        "arenahard": {
            "backend": "vllm",
            "use_custom_chat_template": True,
            "prompt_template": "templates/placeholder.jinja",
            "output_dir": "../../outputs/arenahard/placeholder",
            "judge_command": ["arena-hard", "--answers", "{answer_file}"],
            "question_file": "questions.jsonl",
            "generation": {
                "batch_size": 32,
                "do_sample": False,
                "max_new_tokens": 1024,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        },
    }


def _batch_config() -> dict:
    return {
        "base_config": "config_arenahard.yaml",
        "run_inference": True,
        "run_evaluation": True,
        "skip_existing": True,
        "continue_on_error": False,
        "models": [
            {
                "model_name_or_path": "W-61/ultrafeedback-qwen3-8b-margin-dpo",
                "pretty_name": "ultrafeedback-qwen3-8b-margin-dpo",
            },
            {
                "model_name_or_path": "W-61/ultrafeedback-llama3-8b-beta-dpo",
                "pretty_name": "ultrafeedback-llama3-8b-beta-dpo",
            },
        ],
    }


class ArenaHardBatchRunnerTest(unittest.TestCase):
    def test_build_run_matrix_applies_model_family_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval" / "arenahard" / "config_arenahard_batch.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}", encoding="utf-8")

            with patch("eval.arenahard.batch_runner.load_yaml", return_value=_base_config()):
                run_plans = build_run_matrix(_batch_config(), config_path=config_path)

        self.assertEqual(len(run_plans), 2)

        qwen_cfg = run_plans[0]["config"]["arenahard"]
        self.assertFalse(qwen_cfg["use_custom_chat_template"])
        self.assertNotIn("prompt_template", qwen_cfg)
        self.assertEqual(qwen_cfg["generation"]["stop_token_ids"], [151645])

        llama_cfg = run_plans[1]["config"]["arenahard"]
        self.assertTrue(llama_cfg["use_custom_chat_template"])
        self.assertEqual(
            llama_cfg["prompt_template"],
            "templates/ultrafeedback-llama3-8b-beta-dpo.jinja",
        )
        self.assertEqual(llama_cfg["generation"]["stop_token_ids"], [128001, 128009])

    def test_run_batch_respects_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval" / "arenahard" / "config_arenahard_batch.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}", encoding="utf-8")

            with patch("eval.arenahard.batch_runner.load_yaml", return_value=_base_config()):
                run_plans = build_run_matrix(_batch_config(), config_path=config_path)

            first_config = run_plans[0]["config"]
            output_dir = get_output_dir(first_config, "arenahard")
            (output_dir / "model_answer.jsonl").write_text("{}\n", encoding="utf-8")
            results_dir = output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "arena_hard_judgments.jsonl").write_text("{}\n", encoding="utf-8")

            infer_calls: list[str] = []
            eval_calls: list[str] = []

            def _fake_infer(config):
                infer_calls.append(config["arenahard"]["pretty_name"])
                return get_output_dir(config, "arenahard") / "model_answer.jsonl"

            def _fake_eval(config, *, model_answer_path=None):
                eval_calls.append(config["arenahard"]["pretty_name"])
                return get_output_dir(config, "arenahard") / "results"

            with (
                patch("eval.arenahard.batch_runner.load_yaml", return_value=_base_config()),
                patch(
                    "eval.arenahard.batch_runner.run_arenahard_inference",
                    side_effect=_fake_infer,
                ),
                patch(
                    "eval.arenahard.batch_runner.run_arenahard_evaluation",
                    side_effect=_fake_eval,
                ),
            ):
                exit_code = run_arenahard_batch(
                    _batch_config(),
                    config_path=str(config_path),
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(infer_calls, ["ultrafeedback-llama3-8b-beta-dpo"])
        self.assertEqual(eval_calls, ["ultrafeedback-llama3-8b-beta-dpo"])

    def test_main_passes_eval_only_flag(self):
        with (
            patch("eval.arenahard.batch_runner.load_yaml", return_value=_batch_config()),
            patch(
                "eval.arenahard.batch_runner.run_arenahard_batch",
                return_value=0,
            ) as mock_run_batch,
        ):
            exit_code = main(
                [
                    "--config",
                    "eval/arenahard/config_arenahard_batch.yaml",
                    "--eval-only",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run_batch.assert_called_once_with(
            _batch_config(),
            config_path="eval/arenahard/config_arenahard_batch.yaml",
            run_inference=False,
            run_evaluation=None,
        )

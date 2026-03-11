"""Unit tests for MT-Bench batch orchestration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.benchmark_common import get_output_dir
from eval.mtbench.batch_runner import build_run_matrix, main, run_mtbench_batch


def _base_config() -> dict:
    return {
        "policy_name": "placeholder/model",
        "precision": "bf16",
        "mtbench": {
            "backend": "vllm",
            "use_custom_chat_template": True,
            "prompt_template": "templates/placeholder.jinja",
            "output_dir": "../../outputs/mtbench/placeholder",
            "question_file": "questions.jsonl",
            "reference_answer_file": "reference_answer/gpt-4-1106-preview.jsonl",
            "judge_command": ["mtbench-judge", "--answers", "{answer_file}"],
            "show_result_command": ["mtbench-show", "--input", "{judgment_file}"],
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
        "base_config": "config_mtbench.yaml",
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


class MTBenchBatchRunnerTest(unittest.TestCase):
    def test_build_run_matrix_applies_model_family_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval" / "mtbench" / "config_mtbench_batch.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}", encoding="utf-8")

            with patch("eval.mtbench.batch_runner.load_yaml", return_value=_base_config()):
                run_plans = build_run_matrix(_batch_config(), config_path=config_path)

        self.assertEqual(len(run_plans), 2)

        qwen_cfg = run_plans[0]["config"]["mtbench"]
        self.assertFalse(qwen_cfg["use_custom_chat_template"])
        self.assertNotIn("prompt_template", qwen_cfg)
        self.assertEqual(qwen_cfg["generation"]["stop_token_ids"], [151645])

        llama_cfg = run_plans[1]["config"]["mtbench"]
        self.assertTrue(llama_cfg["use_custom_chat_template"])
        self.assertEqual(
            llama_cfg["prompt_template"],
            "templates/ultrafeedback-llama3-8b-beta-dpo.jinja",
        )
        self.assertEqual(llama_cfg["generation"]["stop_token_ids"], [128001, 128009])

    def test_run_batch_respects_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval" / "mtbench" / "config_mtbench_batch.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}", encoding="utf-8")

            with patch("eval.mtbench.batch_runner.load_yaml", return_value=_base_config()):
                run_plans = build_run_matrix(_batch_config(), config_path=config_path)

            first_config = run_plans[0]["config"]
            output_dir = get_output_dir(first_config, "mtbench")
            (output_dir / "model_answer.jsonl").write_text("{}\n", encoding="utf-8")
            results_dir = output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "mtbench_judgments.jsonl").write_text("{}\n", encoding="utf-8")

            infer_calls: list[str] = []
            eval_calls: list[str] = []

            def _fake_infer(config):
                infer_calls.append(config["mtbench"]["pretty_name"])
                return get_output_dir(config, "mtbench") / "model_answer.jsonl"

            def _fake_eval(config, *, model_answer_path=None):
                eval_calls.append(config["mtbench"]["pretty_name"])
                return get_output_dir(config, "mtbench") / "results"

            with (
                patch("eval.mtbench.batch_runner.load_yaml", return_value=_base_config()),
                patch(
                    "eval.mtbench.batch_runner.run_mtbench_inference",
                    side_effect=_fake_infer,
                ),
                patch(
                    "eval.mtbench.batch_runner.run_mtbench_evaluation",
                    side_effect=_fake_eval,
                ),
            ):
                exit_code = run_mtbench_batch(
                    _batch_config(),
                    config_path=str(config_path),
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(infer_calls, ["ultrafeedback-llama3-8b-beta-dpo"])
        self.assertEqual(eval_calls, ["ultrafeedback-llama3-8b-beta-dpo"])

    def test_main_passes_inference_only_flag(self):
        with (
            patch("eval.mtbench.batch_runner.load_yaml", return_value=_batch_config()),
            patch(
                "eval.mtbench.batch_runner.run_mtbench_batch",
                return_value=0,
            ) as mock_run_batch,
        ):
            exit_code = main(
                [
                    "--config",
                    "eval/mtbench/config_mtbench_batch.yaml",
                    "--inference-only",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run_batch.assert_called_once_with(
            _batch_config(),
            config_path="eval/mtbench/config_mtbench_batch.yaml",
            run_inference=None,
            run_evaluation=False,
        )

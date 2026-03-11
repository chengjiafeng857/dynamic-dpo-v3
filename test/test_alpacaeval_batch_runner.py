"""Unit tests for AlpacaEval batch orchestration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.alpacaeval.alpacaeval_common import get_output_dir
from eval.alpacaeval.batch_runner import build_run_matrix, main, run_alpacaeval_batch


def _base_config() -> dict:
    return {
        "policy_name": "placeholder/model",
        "precision": "bf16",
        "alpacaeval": {
            "backend": "vllm",
            "simpo_compat": True,
            "use_custom_chat_template": True,
            "prompt_template": "templates/ultrafeedback-llama3-8b-margin-dpo.txt",
            "output_dir": "../../outputs/alpacaeval/placeholder",
            "annotators_config": "weighted_alpaca_eval_gpt4_turbo",
            "generation": {
                "batch_size": 32,
                "do_sample": True,
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        },
    }


def _batch_config() -> dict:
    return {
        "base_config": "config_alpacaeval.yaml",
        "run_inference": True,
        "run_evaluation": True,
        "skip_existing": True,
        "continue_on_error": False,
        "use_model_configs": False,
        "overrides": {
            "alpacaeval": {
                "simpo_compat": False,
            }
        },
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


class AlpacaEvalBatchRunnerTest(unittest.TestCase):
    def test_build_run_matrix_applies_model_family_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval" / "alpacaeval" / "config_alpacaeval_batch.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}", encoding="utf-8")

            with patch(
                "eval.alpacaeval.batch_runner.load_yaml",
                return_value=_base_config(),
            ):
                run_plans = build_run_matrix(_batch_config(), config_path=config_path)

        self.assertEqual(len(run_plans), 2)

        qwen_cfg = run_plans[0]["config"]["alpacaeval"]
        self.assertFalse(qwen_cfg["use_custom_chat_template"])
        self.assertNotIn("prompt_template", qwen_cfg)
        self.assertEqual(qwen_cfg["output_dir"], "../../outputs/alpacaeval/ultrafeedback-qwen3-8b-margin-dpo")
        self.assertFalse(qwen_cfg["simpo_compat"])
        self.assertEqual(qwen_cfg["generation"]["stop_token_ids"], [151645])

        llama_cfg = run_plans[1]["config"]["alpacaeval"]
        self.assertTrue(llama_cfg["use_custom_chat_template"])
        self.assertEqual(
            llama_cfg["prompt_template"],
            "templates/ultrafeedback-llama3-8b-beta-dpo.txt",
        )
        self.assertEqual(llama_cfg["output_dir"], "../../outputs/alpacaeval/ultrafeedback-llama3-8b-beta-dpo")
        self.assertEqual(llama_cfg["generation"]["stop_token_ids"], [128001, 128009])

    def test_run_batch_calls_infer_then_eval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "eval" / "alpacaeval" / "config_alpacaeval_batch.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{}", encoding="utf-8")

            infer_calls: list[str] = []
            eval_calls: list[tuple[str, str | None, bool]] = []

            def _fake_infer(config):
                output_dir = get_output_dir(config)
                infer_calls.append(config["alpacaeval"]["pretty_name"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "model_outputs.json").write_text("[]", encoding="utf-8")
                return output_dir / "model_outputs.json"

            def _fake_eval(config, *, model_outputs_path=None, use_model_configs=False):
                eval_calls.append(
                    (
                        config["alpacaeval"]["pretty_name"],
                        model_outputs_path,
                        use_model_configs,
                    )
                )
                results_dir = get_output_dir(config) / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                return results_dir

            with (
                patch(
                    "eval.alpacaeval.batch_runner.load_yaml",
                    return_value=_base_config(),
                ),
                patch(
                    "eval.alpacaeval.batch_runner.run_alpacaeval_inference",
                    side_effect=_fake_infer,
                ),
                patch(
                    "eval.alpacaeval.batch_runner.run_alpacaeval_evaluation",
                    side_effect=_fake_eval,
                ),
            ):
                exit_code = run_alpacaeval_batch(
                    _batch_config(),
                    config_path=str(config_path),
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            infer_calls,
            [
                "ultrafeedback-qwen3-8b-margin-dpo",
                "ultrafeedback-llama3-8b-beta-dpo",
            ],
        )
        self.assertEqual(
            eval_calls,
            [
                (
                    "ultrafeedback-qwen3-8b-margin-dpo",
                    str(
                        (
                            Path(tmp_dir)
                            / "outputs/alpacaeval/ultrafeedback-qwen3-8b-margin-dpo/model_outputs.json"
                        ).resolve()
                    ),
                    False,
                ),
                (
                    "ultrafeedback-llama3-8b-beta-dpo",
                    str(
                        (
                            Path(tmp_dir)
                            / "outputs/alpacaeval/ultrafeedback-llama3-8b-beta-dpo/model_outputs.json"
                        ).resolve()
                    ),
                    False,
                ),
            ],
        )

    def test_main_passes_eval_only_flag(self):
        with (
            patch(
                "eval.alpacaeval.batch_runner.load_yaml",
                return_value=_batch_config(),
            ),
            patch(
                "eval.alpacaeval.batch_runner.run_alpacaeval_batch",
                return_value=0,
            ) as mock_run_batch,
        ):
            exit_code = main(
                [
                    "--config",
                    "eval/alpacaeval/config_alpacaeval_batch.yaml",
                    "--eval-only",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run_batch.assert_called_once_with(
            _batch_config(),
            config_path="eval/alpacaeval/config_alpacaeval_batch.yaml",
            run_inference=False,
            run_evaluation=None,
            use_model_configs=None,
        )

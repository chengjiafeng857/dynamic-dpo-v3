"""Unit tests for AlpacaEval inference and evaluation helpers."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from src.cli import main_alpacaeval_eval, main_alpacaeval_infer
from eval.alpacaeval.alpacaeval_eval import (
    _build_runtime_model_config,
    run_alpacaeval_evaluation,
)
from eval.alpacaeval.alpacaeval_infer import run_alpacaeval_inference


def _base_config(output_dir: str) -> dict:
    return {
        "policy_name": "dummy/local-model",
        "precision": "bf16",
        "alpacaeval": {
            "pretty_name": "Dummy-SimPO",
            "model_name_or_path": "dummy/local-model",
            "backend": "transformers",
            "simpo_compat": False,
            "prompt_template": "templates/llama3.txt",
            "output_dir": output_dir,
            "dataset_name": "tatsu-lab/alpaca_eval",
            "dataset_config": "alpaca_eval",
            "dataset_split": "eval",
            "annotators_config": "weighted_alpaca_eval_gpt4_turbo",
            "evaluation_mode": "outputs",
            "generation": {
                "batch_size": 2,
                "do_sample": False,
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop_token_ids": [128001, 128009],
            },
        },
        "_config_path": str(
            Path("/home/feng/github/dynamic-dpo-v3/eval/alpacaeval/config_alpacaeval.yaml")
        ),
    }


class AlpacaEvalPipelineTest(unittest.TestCase):
    def test_inference_writes_outputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            dataset = Dataset.from_list(
                [
                    {
                        "instruction": "Say hi",
                        "dataset": "helpful_base",
                        "datasplit": "eval",
                    }
                ]
            )

            with (
                patch("eval.alpacaeval.alpacaeval_infer.load_dataset", return_value=dataset),
                patch(
                    "eval.alpacaeval.alpacaeval_infer._generate_with_transformers",
                    return_value=["Hello there."],
                ),
            ):
                model_outputs_path = run_alpacaeval_inference(config)

            payload = json.loads(model_outputs_path.read_text(encoding="utf-8"))
            metadata = json.loads(
                (Path(tmp_dir) / "metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(payload[0]["instruction"], "Say hi")
            self.assertEqual(payload[0]["output"], "Hello there.")
            self.assertEqual(payload[0]["generator"], "Dummy-SimPO")
            self.assertEqual(metadata["backend"], "transformers")
            self.assertTrue(metadata["prompt_template"].endswith("llama3.txt"))

    def test_inference_can_use_model_default_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            config["alpacaeval"]["use_custom_chat_template"] = False
            dataset = Dataset.from_list([{"instruction": "Say hi"}])
            test_case = self

            class _DummyTokenizer:
                chat_template = "dummy"

                def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                    test_case.assertFalse(tokenize)
                    test_case.assertTrue(add_generation_prompt)
                    return f"default::{messages[0]['content']}"

            with (
                patch("eval.alpacaeval.alpacaeval_infer.load_dataset", return_value=dataset),
                patch(
                    "eval.alpacaeval.alpacaeval_infer.AutoTokenizer.from_pretrained",
                    return_value=_DummyTokenizer(),
                ),
                patch(
                    "eval.alpacaeval.alpacaeval_infer._generate_with_transformers",
                    return_value=["Hello there."],
                ),
            ):
                model_outputs_path = run_alpacaeval_inference(config)

            payload = json.loads(model_outputs_path.read_text(encoding="utf-8"))
            metadata = json.loads((Path(tmp_dir) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["output"], "Hello there.")
            self.assertEqual(metadata["prompt_template"], "model_default")
            self.assertFalse(metadata["use_custom_chat_template"])

    def test_evaluation_builds_evaluate_command_from_model_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            model_outputs_path = Path(tmp_dir) / "model_outputs.json"
            model_outputs_path.write_text("[]", encoding="utf-8")

            with (
                patch("eval.alpacaeval.alpacaeval_eval.metadata.version", return_value="0.6.2"),
                patch("eval.alpacaeval.alpacaeval_eval.subprocess.run") as mock_run,
            ):
                results_dir = run_alpacaeval_evaluation(
                    config,
                    model_outputs_path=str(model_outputs_path),
                )

            command = mock_run.call_args.args[0]
            self.assertEqual(command[:3], [sys.executable, "-m", "alpaca_eval.main"])
            self.assertIn("evaluate", command)
            self.assertIn("--model_outputs", command)
            self.assertIn(str(model_outputs_path), command)
            self.assertIn("--annotators_config", command)
            self.assertIn("weighted_alpaca_eval_gpt4_turbo", command)
            self.assertEqual(results_dir, Path(tmp_dir) / "results")

    def test_evaluation_builds_runtime_model_config_for_evaluate_from_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            config["alpacaeval"]["backend"] = "vllm"
            config["alpacaeval"]["evaluation_mode"] = "model_configs"

            with (
                patch("eval.alpacaeval.alpacaeval_eval.metadata.version", return_value="0.6.2"),
                patch("eval.alpacaeval.alpacaeval_eval.subprocess.run") as mock_run,
            ):
                results_dir = run_alpacaeval_evaluation(config, use_model_configs=True)

            model_config = yaml_safe_load(Path(tmp_dir) / "alpacaeval_model_config.yaml")
            key = next(iter(model_config))
            self.assertEqual(model_config[key]["fn_completions"], "vllm_local_completions")
            self.assertEqual(
                model_config[key]["completions_kwargs"]["model_name"],
                "dummy/local-model",
            )
            self.assertEqual(results_dir, Path(tmp_dir) / "results")
            command = mock_run.call_args.args[0]
            self.assertIn("evaluate_from_model", command)
            self.assertIn("--model_configs", command)
            self.assertIn(str(Path(tmp_dir) / "alpacaeval_model_config.yaml"), command)

    def test_runtime_model_config_matches_repo_template_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            config_path = _build_runtime_model_config(config)
            payload = yaml_safe_load(config_path)
            key = next(iter(payload))
            self.assertTrue(payload[key]["prompt_template"].endswith("llama3.txt"))
            self.assertEqual(payload[key]["pretty_name"], "Dummy-SimPO")

    def test_evaluate_from_model_rejects_model_default_chat_template_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            config["alpacaeval"]["use_custom_chat_template"] = False

            with self.assertRaisesRegex(
                ValueError,
                "use_custom_chat_template=false is not supported with evaluate_from_model",
            ):
                _build_runtime_model_config(config)

    def test_main_alpacaeval_infer_loads_config_and_runs(self):
        config = _base_config("/tmp/alpacaeval-test")
        with (
            patch("src.cli.load_yaml", return_value=config),
            patch("eval.alpacaeval.alpacaeval_infer.run_alpacaeval_inference") as mock_run,
            patch.object(
                sys,
                "argv",
                ["alpacaeval-infer", "--config", "eval/alpacaeval/config_alpacaeval.yaml"],
            ),
        ):
            main_alpacaeval_infer()
        self.assertEqual(
            mock_run.call_args.args[0]["_config_path"],
            "eval/alpacaeval/config_alpacaeval.yaml",
        )

    def test_main_alpacaeval_eval_passes_cli_flags(self):
        config = _base_config("/tmp/alpacaeval-test")
        with (
            patch("src.cli.load_yaml", return_value=config),
            patch("eval.alpacaeval.alpacaeval_eval.run_alpacaeval_evaluation") as mock_run,
            patch.object(
                sys,
                "argv",
                [
                    "alpacaeval-eval",
                    "--config",
                    "eval/alpacaeval/config_alpacaeval.yaml",
                    "--model-outputs",
                    "outputs.json",
                    "--use-model-configs",
                ],
            ),
        ):
            main_alpacaeval_eval()
        self.assertEqual(mock_run.call_args.kwargs["model_outputs_path"], "outputs.json")
        self.assertTrue(mock_run.call_args.kwargs["use_model_configs"])


def yaml_safe_load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))

"""Unit tests for the batch DPO orchestration flow."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.batch_dpo_runner import build_run_matrix, main, run_batch_dpo


def _batch_config() -> dict:
    return {
        "hf_username": "W-61",
        "cleanup": {
            "delete_run_output": False,
            "delete_policy_model_cache": False,
            "delete_dataset_cache": False,
            "delete_completed_policy_model_cache": False,
        },
        "execution_order": "run_major",
        "trainers": [
            {
                "trainer_type": "beta",
                "trainer_slug": "beta-dpo",
                "base_config": "config_beta_dpo.yaml",
            },
            {
                "trainer_type": "margin",
                "trainer_slug": "margin-dpo",
                "base_config": "config_margin_dpo.yaml",
            },
        ],
        "runs": [
            {
                "dataset_slug": "hh-helpful-base",
                "model_slug": "qwen3-8b",
                "policy_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "ref_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "dataset_data_dir": "helpful-base",
                "chat_template_name": "qwen3",
                "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
            },
            {
                "dataset_slug": "hh-harmless-base",
                "model_slug": "qwen3-8b",
                "policy_name": "W-61/hh-harmless-base-qwen3-8b-sft",
                "ref_name": "W-61/hh-harmless-base-qwen3-8b-sft",
                "dataset_data_dir": "harmless-base",
                "chat_template_name": "qwen3",
                "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
            },
            {
                "dataset_slug": "hh-helpful-base",
                "model_slug": "llama3-8b",
                "policy_name": "W-61/hh-helpful-base-llama3-8b-sft",
                "ref_name": "W-61/hh-helpful-base-llama3-8b-sft",
                "dataset_data_dir": "helpful-base",
                "chat_template_name": "llama3",
                "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            },
            {
                "dataset_slug": "hh-harmless-base",
                "model_slug": "llama3-8b",
                "policy_name": "W-61/hh-harmless-base-llama3-8b-sft",
                "ref_name": "W-61/hh-harmless-base-llama3-8b-sft",
                "dataset_data_dir": "harmless-base",
                "chat_template_name": "llama3",
                "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            },
        ],
    }


def _beta_base_config() -> dict:
    return {
        "policy_name": "placeholder/policy",
        "ref_name": "placeholder/ref",
        "dataset": {
            "dataset_name": "Anthropic/hh-rlhf",
            "data_dir": "placeholder-dir",
            "chat_template": True,
            "chat_template_name": "llama3",
        },
        "dpo_training": {
            "run_name": "placeholder-run",
            "save_dir": "placeholder-save",
            "hub_model_id": "placeholder-hub",
            "fsdp": {
                "enabled": True,
                "transformer_layer_cls_to_wrap": "OldLayer",
            },
        },
        "beta_dpo": {
            "beta": 0.1,
            "rho": 0.8,
            "alpha": 1.0,
        },
    }


def _margin_base_config() -> dict:
    return {
        "policy_name": "placeholder/policy",
        "ref_name": "placeholder/ref",
        "dataset": {
            "dataset_name": "Anthropic/hh-rlhf",
            "data_dir": "placeholder-dir",
            "chat_template": True,
            "chat_template_name": "llama3",
        },
        "dpo_training": {
            "run_name": "placeholder-run",
            "save_dir": "placeholder-save",
            "hub_model_id": "placeholder-hub",
            "fsdp": {
                "enabled": True,
                "transformer_layer_cls_to_wrap": "OldLayer",
            },
        },
        "margin_log": {
            "log_dir": "logs/placeholder",
        },
    }


def _load_yaml_stub(path: str) -> dict:
    name = Path(path).name
    if name == "config_beta_dpo.yaml":
        return _beta_base_config()
    if name == "config_margin_dpo.yaml":
        return _margin_base_config()
    raise AssertionError(f"Unexpected config path: {path}")


class DPOBatchRunnerTest(unittest.TestCase):
    def test_main_accepts_torchrun_local_rank_argument(self):
        with (
            patch("src.batch_dpo_runner.load_yaml", return_value=_batch_config()),
            patch("src.batch_dpo_runner.run_batch_dpo", return_value=0) as mock_run_batch,
        ):
            exit_code = main(["--config", "config_dpo_batch.yaml", "--local-rank", "5"])

        self.assertEqual(exit_code, 0)
        mock_run_batch.assert_called_once()

    def test_build_run_matrix_expands_eight_runs_in_run_major_order(self):
        with patch("src.batch_dpo_runner.load_yaml", side_effect=_load_yaml_stub):
            run_plans = build_run_matrix(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(len(run_plans), 8)
        self.assertEqual(
            [
                (plan["dataset_slug"], plan["model_slug"], plan["trainer_slug"])
                for plan in run_plans
            ],
            [
                ("hh-helpful-base", "qwen3-8b", "beta-dpo"),
                ("hh-helpful-base", "qwen3-8b", "margin-dpo"),
                ("hh-harmless-base", "qwen3-8b", "beta-dpo"),
                ("hh-harmless-base", "qwen3-8b", "margin-dpo"),
                ("hh-helpful-base", "llama3-8b", "beta-dpo"),
                ("hh-helpful-base", "llama3-8b", "margin-dpo"),
                ("hh-harmless-base", "llama3-8b", "beta-dpo"),
                ("hh-harmless-base", "llama3-8b", "margin-dpo"),
            ],
        )

        for plan in run_plans:
            dataset_slug = plan["dataset_slug"]
            policy_name = plan["policy_name"]
            dataset_data_dir = plan["config"]["dataset"]["data_dir"]
            if "helpful" in dataset_slug:
                self.assertIn("helpful", policy_name)
                self.assertEqual(dataset_data_dir, "helpful-base")
            if "harmless" in dataset_slug:
                self.assertIn("harmless", policy_name)
                self.assertEqual(dataset_data_dir, "harmless-base")

        first_cfg = run_plans[0]["config"]
        second_cfg = run_plans[1]["config"]
        last_cfg = run_plans[-1]["config"]

        self.assertEqual(first_cfg["policy_name"], "W-61/hh-helpful-base-qwen3-8b-sft")
        self.assertEqual(first_cfg["ref_name"], "W-61/hh-helpful-base-qwen3-8b-sft")
        self.assertEqual(first_cfg["dataset"]["chat_template_name"], "qwen3")
        self.assertEqual(first_cfg["dpo_training"]["run_name"], "hh-helpful-base-qwen3-8b-beta-dpo")
        self.assertEqual(first_cfg["dpo_training"]["save_dir"], "batch_dpo_runs/hh-helpful-base-qwen3-8b-beta-dpo")
        self.assertEqual(first_cfg["dpo_training"]["hub_model_id"], "W-61/hh-helpful-base-qwen3-8b-beta-dpo")
        self.assertEqual(
            first_cfg["dpo_training"]["fsdp"]["transformer_layer_cls_to_wrap"],
            "Qwen3DecoderLayer",
        )
        self.assertEqual(first_cfg["beta_dpo"]["rho"], 0.8)
        self.assertEqual(run_plans[0]["output_dir"], "batch_dpo_outputs/hh-helpful-base-qwen3-8b-beta-dpo")

        self.assertEqual(second_cfg["margin_log"]["log_dir"], "logs/hh-helpful-base-qwen3-8b-margin-dpo-margins")
        self.assertEqual(last_cfg["policy_name"], "W-61/hh-harmless-base-llama3-8b-sft")
        self.assertEqual(last_cfg["dataset"]["chat_template_name"], "llama3")
        self.assertEqual(
            last_cfg["dpo_training"]["fsdp"]["transformer_layer_cls_to_wrap"],
            "LlamaDecoderLayer",
        )

    def test_run_batch_dpo_calls_beta_then_margin_for_each_run(self):
        batch_config = _batch_config()
        batch_config["cleanup"]["delete_completed_policy_model_cache"] = True

        with patch("src.batch_dpo_runner.load_yaml", side_effect=_load_yaml_stub):
            run_plans = build_run_matrix(batch_config, config_dir=Path("/tmp/project"))

        log: list[str] = []

        def _fake_beta(config, *, output_dir):
            log.append(f"beta:{config['dpo_training']['run_name']}:{output_dir}")

        def _fake_margin(config, *, output_dir):
            log.append(f"margin:{config['dpo_training']['run_name']}:{output_dir}")

        def _fake_cleanup(*, run_plan, cleanup_config):
            del cleanup_config
            log.append(f"cleanup:{run_plan['config']['dpo_training']['run_name']}")

        def _fake_cleanup_completed_policy_cache(*, completed_policy_name, cleanup_config):
            del cleanup_config
            log.append(f"delete:{completed_policy_name}")

        with (
            patch("src.batch_dpo_runner.build_run_matrix", return_value=run_plans),
            patch("src.batch_dpo_runner.run_beta_dpo_training", side_effect=_fake_beta),
            patch("src.batch_dpo_runner.run_margin_dpo_training", side_effect=_fake_margin),
            patch("src.batch_dpo_runner.cleanup_run_artifacts", side_effect=_fake_cleanup),
            patch(
                "src.batch_dpo_runner.cleanup_completed_policy_cache",
                side_effect=_fake_cleanup_completed_policy_cache,
            ),
            patch("src.batch_dpo_runner._distributed_barrier"),
            patch("src.batch_dpo_runner._is_main_process", return_value=True),
        ):
            exit_code = run_batch_dpo(batch_config, config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            log[:8],
            [
                "beta:hh-helpful-base-qwen3-8b-beta-dpo:batch_dpo_outputs/hh-helpful-base-qwen3-8b-beta-dpo",
                "cleanup:hh-helpful-base-qwen3-8b-beta-dpo",
                "margin:hh-helpful-base-qwen3-8b-margin-dpo:batch_dpo_outputs/hh-helpful-base-qwen3-8b-margin-dpo",
                "cleanup:hh-helpful-base-qwen3-8b-margin-dpo",
                "beta:hh-harmless-base-qwen3-8b-beta-dpo:batch_dpo_outputs/hh-harmless-base-qwen3-8b-beta-dpo",
                "cleanup:hh-harmless-base-qwen3-8b-beta-dpo",
                "margin:hh-harmless-base-qwen3-8b-margin-dpo:batch_dpo_outputs/hh-harmless-base-qwen3-8b-margin-dpo",
                "cleanup:hh-harmless-base-qwen3-8b-margin-dpo",
            ],
        )
        self.assertEqual(
            [entry for entry in log if entry.startswith("delete:")],
            [
                "delete:W-61/hh-helpful-base-qwen3-8b-sft",
                "delete:W-61/hh-harmless-base-qwen3-8b-sft",
                "delete:W-61/hh-helpful-base-llama3-8b-sft",
                "delete:W-61/hh-harmless-base-llama3-8b-sft",
            ],
        )

    def test_run_batch_dpo_fails_fast_on_first_error(self):
        with patch("src.batch_dpo_runner.load_yaml", side_effect=_load_yaml_stub):
            run_plans = build_run_matrix(_batch_config(), config_dir=Path("/tmp/project"))

        buffer = io.StringIO()
        with (
            patch("src.batch_dpo_runner.build_run_matrix", return_value=run_plans),
            patch(
                "src.batch_dpo_runner.run_beta_dpo_training",
                side_effect=RuntimeError("boom"),
            ),
            patch("src.batch_dpo_runner.run_margin_dpo_training") as mock_margin,
            patch("src.batch_dpo_runner.cleanup_run_artifacts") as mock_cleanup,
            patch("src.batch_dpo_runner._distributed_barrier"),
            patch("src.batch_dpo_runner._is_main_process", return_value=True),
            redirect_stdout(buffer),
        ):
            exit_code = run_batch_dpo(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(mock_margin.called)
        self.assertFalse(mock_cleanup.called)
        self.assertIn("[DPO-BATCH] Failed hh-helpful-base / qwen3-8b / beta-dpo (1/8): boom", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the batch DPO orchestration flow."""

import errno
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.batch_dpo_runner import archive_margin_logs, build_run_matrix, main, run_batch_dpo


def _batch_config() -> dict:
    return {
        "hf_username": "W-61",
        "cleanup": {
            "delete_run_output": True,
            "delete_policy_model_cache": False,
            "delete_dataset_cache": False,
            "delete_completed_policy_model_cache": True,
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
            "archive_after_run": True,
            "delete_local_after_archive": True,
            "hf_dataset_repo_id": "W-61/placeholder-margin-logs",
            "hf_dataset_private": False,
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
        self.assertTrue(second_cfg["margin_log"]["archive_after_run"])
        self.assertTrue(second_cfg["margin_log"]["delete_local_after_archive"])
        self.assertEqual(
            second_cfg["margin_log"]["hf_dataset_repo_id"],
            "W-61/hh-helpful-base-qwen3-8b-margin-dpo-margin-logs",
        )
        self.assertFalse(second_cfg["margin_log"]["hf_dataset_private"])
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
            patch("src.batch_dpo_runner.archive_margin_logs", return_value=None),
            patch("src.batch_dpo_runner.cleanup_run_artifacts", side_effect=_fake_cleanup),
            patch(
                "src.batch_dpo_runner.cleanup_completed_policy_cache",
                side_effect=_fake_cleanup_completed_policy_cache,
            ),
            patch("src.batch_dpo_runner.distributed_barrier"),
            patch("src.batch_dpo_runner.is_main_process", return_value=True),
        ):
            exit_code = run_batch_dpo(batch_config, config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            log[:12],
            [
                "beta:hh-helpful-base-qwen3-8b-beta-dpo:batch_dpo_outputs/hh-helpful-base-qwen3-8b-beta-dpo",
                "cleanup:hh-helpful-base-qwen3-8b-beta-dpo",
                "margin:hh-helpful-base-qwen3-8b-margin-dpo:batch_dpo_outputs/hh-helpful-base-qwen3-8b-margin-dpo",
                "cleanup:hh-helpful-base-qwen3-8b-margin-dpo",
                "delete:W-61/hh-helpful-base-qwen3-8b-sft",
                "beta:hh-harmless-base-qwen3-8b-beta-dpo:batch_dpo_outputs/hh-harmless-base-qwen3-8b-beta-dpo",
                "cleanup:hh-harmless-base-qwen3-8b-beta-dpo",
                "margin:hh-harmless-base-qwen3-8b-margin-dpo:batch_dpo_outputs/hh-harmless-base-qwen3-8b-margin-dpo",
                "cleanup:hh-harmless-base-qwen3-8b-margin-dpo",
                "delete:W-61/hh-harmless-base-qwen3-8b-sft",
                "beta:hh-helpful-base-llama3-8b-beta-dpo:batch_dpo_outputs/hh-helpful-base-llama3-8b-beta-dpo",
                "cleanup:hh-helpful-base-llama3-8b-beta-dpo",
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
            patch("src.batch_dpo_runner.archive_margin_logs", return_value=None),
            patch("src.batch_dpo_runner.cleanup_run_artifacts") as mock_cleanup,
            patch("src.batch_dpo_runner.distributed_barrier"),
            patch("src.batch_dpo_runner.is_main_process", return_value=True),
            redirect_stdout(buffer),
        ):
            exit_code = run_batch_dpo(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(mock_margin.called)
        self.assertFalse(mock_cleanup.called)
        self.assertIn("[DPO-BATCH] Failed hh-helpful-base / qwen3-8b / beta-dpo (1/8): boom", buffer.getvalue())

    def test_archive_margin_logs_deletes_local_logs_after_hf_upload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "margins"
            log_dir.mkdir()
            (log_dir / "margins.jsonl").write_text("{}", encoding="utf-8")
            run_plan = {
                "trainer_type": "margin",
                "config": {
                    "dpo_training": {
                        "run_name": "margin-run",
                        "wandb_project": "wandb-project",
                    },
                    "margin_log": {
                        "log_dir": str(log_dir),
                        "archive_after_run": True,
                        "delete_local_after_archive": True,
                        "hf_dataset_repo_id": "W-61/margin-run-margin-logs",
                        "hf_dataset_private": False,
                    },
                },
            }

            with patch(
                "src.batch_dpo_runner._archive_margin_logs_to_hf_dataset",
                return_value=True,
            ) as mock_hf:
                error = archive_margin_logs(run_plan)

        self.assertIsNone(error)
        mock_hf.assert_called_once()
        self.assertFalse(log_dir.exists())

    def test_archive_margin_logs_skips_non_margin_runs(self):
        run_plan = {
            "trainer_type": "beta",
            "config": {
                "dpo_training": {"run_name": "beta-run"},
                "margin_log": {"log_dir": "logs/unused"},
            },
        }

        with patch("src.batch_dpo_runner._archive_margin_logs_to_hf_dataset") as mock_hf:
            error = archive_margin_logs(run_plan)

        self.assertIsNone(error)
        mock_hf.assert_not_called()

    def test_run_batch_dpo_fails_when_margin_log_archive_fails(self):
        run_plans = [
            {
                "trainer_type": "margin",
                "trainer_slug": "margin-dpo",
                "base_config_path": "config_margin_dpo.yaml",
                "dataset_slug": "hh-helpful-base",
                "model_slug": "qwen3-8b",
                "policy_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "ref_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "hub_model_id": "W-61/hh-helpful-base-qwen3-8b-margin-dpo",
                "output_dir": "batch_dpo_outputs/hh-helpful-base-qwen3-8b-margin-dpo",
                "config": {
                    "dataset": {"dataset_name": "Anthropic/hh-rlhf"},
                    "dpo_training": {
                        "run_name": "hh-helpful-base-qwen3-8b-margin-dpo",
                        "save_dir": "batch_dpo_runs/hh-helpful-base-qwen3-8b-margin-dpo",
                    },
                    "margin_log": {
                        "log_dir": "logs/hh-helpful-base-qwen3-8b-margin-dpo-margins",
                        "delete_local_after_archive": True,
                    },
                },
            }
        ]
        buffer = io.StringIO()

        with (
            patch("src.batch_dpo_runner.build_run_matrix", return_value=run_plans),
            patch("src.batch_dpo_runner.run_margin_dpo_training"),
            patch(
                "src.batch_dpo_runner.archive_margin_logs",
                return_value="Failed to archive margin logs for hh-helpful-base-qwen3-8b-margin-dpo.",
            ),
            patch("src.batch_dpo_runner.cleanup_run_artifacts") as mock_cleanup,
            patch("src.batch_dpo_runner.distributed_barrier", return_value=None),
            patch("src.batch_dpo_runner.is_main_process", return_value=True),
            redirect_stdout(buffer),
        ):
            exit_code = run_batch_dpo(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(mock_cleanup.called)
        self.assertIn("Failed to archive margin logs", buffer.getvalue())

    def test_run_batch_dpo_fails_fast_on_oom_error(self):
        run_plans = [
            {
                "trainer_type": "beta",
                "trainer_slug": "beta-dpo",
                "base_config_path": "config_beta_dpo.yaml",
                "dataset_slug": "hh-helpful-base",
                "model_slug": "qwen3-8b",
                "policy_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "ref_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "hub_model_id": "W-61/hh-helpful-base-qwen3-8b-beta-dpo",
                "output_dir": "batch_dpo_outputs/hh-helpful-base-qwen3-8b-beta-dpo",
                "config": {
                    "dataset": {"dataset_name": "Anthropic/hh-rlhf"},
                    "dpo_training": {
                        "run_name": "hh-helpful-base-qwen3-8b-beta-dpo",
                        "save_dir": "batch_dpo_runs/hh-helpful-base-qwen3-8b-beta-dpo",
                    },
                },
            }
        ]
        buffer = io.StringIO()

        with (
            patch("src.batch_dpo_runner.build_run_matrix", return_value=run_plans),
            patch(
                "src.batch_dpo_runner.run_beta_dpo_training",
                side_effect=RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB"),
            ),
            patch("src.batch_dpo_runner.cleanup_run_artifacts") as mock_cleanup,
            patch("src.batch_dpo_runner.distributed_barrier", return_value=None),
            patch("src.batch_dpo_runner.is_main_process", return_value=True),
            redirect_stdout(buffer),
        ):
            exit_code = run_batch_dpo(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(mock_cleanup.called)
        self.assertIn("OOM: CUDA out of memory", buffer.getvalue())

    def test_run_batch_dpo_fails_fast_on_disk_full_error(self):
        run_plans = [
            {
                "trainer_type": "beta",
                "trainer_slug": "beta-dpo",
                "base_config_path": "config_beta_dpo.yaml",
                "dataset_slug": "hh-helpful-base",
                "model_slug": "qwen3-8b",
                "policy_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "ref_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "hub_model_id": "W-61/hh-helpful-base-qwen3-8b-beta-dpo",
                "output_dir": "batch_dpo_outputs/hh-helpful-base-qwen3-8b-beta-dpo",
                "config": {
                    "dataset": {"dataset_name": "Anthropic/hh-rlhf"},
                    "dpo_training": {
                        "run_name": "hh-helpful-base-qwen3-8b-beta-dpo",
                        "save_dir": "batch_dpo_runs/hh-helpful-base-qwen3-8b-beta-dpo",
                    },
                },
            }
        ]
        buffer = io.StringIO()

        with (
            patch("src.batch_dpo_runner.build_run_matrix", return_value=run_plans),
            patch(
                "src.batch_dpo_runner.run_beta_dpo_training",
                side_effect=OSError(errno.ENOSPC, "No space left on device"),
            ),
            patch("src.batch_dpo_runner.cleanup_run_artifacts") as mock_cleanup,
            patch("src.batch_dpo_runner.distributed_barrier", return_value=None),
            patch("src.batch_dpo_runner.is_main_process", return_value=True),
            redirect_stdout(buffer),
        ):
            exit_code = run_batch_dpo(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(mock_cleanup.called)
        self.assertIn("Disk full: [Errno 28] No space left on device", buffer.getvalue())

    def test_run_batch_dpo_fails_on_distributed_sync_error_without_local_exception(self):
        run_plans = [
            {
                "trainer_type": "beta",
                "trainer_slug": "beta-dpo",
                "base_config_path": "config_beta_dpo.yaml",
                "dataset_slug": "hh-helpful-base",
                "model_slug": "qwen3-8b",
                "policy_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "ref_name": "W-61/hh-helpful-base-qwen3-8b-sft",
                "hub_model_id": "W-61/hh-helpful-base-qwen3-8b-beta-dpo",
                "output_dir": "batch_dpo_outputs/hh-helpful-base-qwen3-8b-beta-dpo",
                "config": {
                    "dataset": {"dataset_name": "Anthropic/hh-rlhf"},
                    "dpo_training": {
                        "run_name": "hh-helpful-base-qwen3-8b-beta-dpo",
                        "save_dir": "batch_dpo_runs/hh-helpful-base-qwen3-8b-beta-dpo",
                    },
                },
            }
        ]
        buffer = io.StringIO()

        with (
            patch("src.batch_dpo_runner.build_run_matrix", return_value=run_plans),
            patch("src.batch_dpo_runner.run_beta_dpo_training"),
            patch(
                "src.batch_dpo_runner.gather_error_messages",
                return_value=[
                    "Distributed synchronization failed while collecting errors: peer reset"
                ],
            ),
            patch("src.batch_dpo_runner.cleanup_run_artifacts") as mock_cleanup,
            patch("src.batch_dpo_runner.distributed_barrier", return_value=None),
            patch("src.batch_dpo_runner.is_main_process", return_value=True),
            redirect_stdout(buffer),
        ):
            exit_code = run_batch_dpo(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(mock_cleanup.called)
        self.assertIn("Distributed synchronization failed while collecting errors", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

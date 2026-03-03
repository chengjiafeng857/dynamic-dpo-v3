"""Unit tests for the batch SFT orchestration flow."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.batch_sft_runner import build_run_matrix, cleanup_run_artifacts, run_batch_sft


def _batch_config() -> dict:
    return {
        "hf_username": "W-61",
        "cleanup": {
            "delete_run_output": True,
            "delete_hf_download_cache": True,
        },
        "execution_order": "dataset_major",
        "datasets": [
            {
                "base_config": "config_sft_hh_helpful_base.yaml",
                "dataset_slug": "hh-helpful-base",
            },
            {
                "base_config": "config_sft_hh_harmless_base.yaml",
                "dataset_slug": "hh-harmless-base",
            },
            {
                "base_config": "config_sft_ultrachat.yaml",
                "dataset_slug": "ultrachat",
            },
        ],
        "models": [
            {
                "policy_name": "meta-llama/Meta-Llama-3-8B",
                "ref_name": "meta-llama/Meta-Llama-3-8B",
                "model_slug": "llama3-8b",
                "chat_template_name": "llama3",
                "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            },
            {
                "policy_name": "Qwen/Qwen3-8B",
                "ref_name": "Qwen/Qwen3-8B",
                "model_slug": "qwen3-8b",
                "chat_template_name": "qwen3",
                "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
            },
        ],
    }


def _hh_base_config(config_name: str) -> dict:
    return {
        "policy_name": "placeholder/model",
        "ref_name": "placeholder/model",
        "dataset": {
            "dataset_name": "Anthropic/hh-rlhf",
            "config_name": config_name,
            "chat_template_name": "llama3",
        },
        "sft_training": {
            "run_name": "placeholder",
            "save_dir": "placeholder",
            "hub_model_id": "placeholder",
            "push_to_hub": True,
            "fsdp": {
                "enabled": True,
                "transformer_layer_cls_to_wrap": "OldLayer",
            },
        },
    }


def _ultrachat_base_config() -> dict:
    return {
        "policy_name": "placeholder/model",
        "dataset": {
            "dataset_name": "HuggingFaceH4/ultrachat_200k",
            "chat_template_name": "llama3",
        },
        "sft_training": {
            "run_name": "placeholder",
            "save_dir": "placeholder",
            "hub_model_id": "placeholder",
            "push_to_hub": True,
            "fsdp": {
                "enabled": True,
                "transformer_layer_cls_to_wrap": "OldLayer",
            },
        },
    }


def _load_yaml_stub(path: str) -> dict:
    name = Path(path).name
    if name == "config_sft_hh_helpful_base.yaml":
        return _hh_base_config("helpful-base")
    if name == "config_sft_hh_harmless_base.yaml":
        return _hh_base_config("harmless-base")
    if name == "config_sft_ultrachat.yaml":
        return _ultrachat_base_config()
    raise AssertionError(f"Unexpected config path: {path}")


class _DummyTrainer:
    def __init__(self, log: list[str], hub_model_id: str):
        self._log = log
        self._hub_model_id = hub_model_id
        self.model = object()

    def push_to_hub(self):
        self._log.append(f"push:{self._hub_model_id}")


class _DeleteStrategy:
    def __init__(self, log: list[tuple[str, tuple[str, ...]]], hashes: tuple[str, ...]):
        self._log = log
        self._hashes = hashes

    def execute(self):
        self._log.append(("execute", self._hashes))


class _CacheInfo:
    def __init__(self, log: list[tuple[str, tuple[str, ...]]]):
        self._log = log
        self.repos = [
            SimpleNamespace(
                repo_id="meta-llama/Meta-Llama-3-8B",
                revisions=[SimpleNamespace(commit_hash="model-hash")],
            ),
            SimpleNamespace(
                repo_id="Anthropic/hh-rlhf",
                revisions=[SimpleNamespace(commit_hash="dataset-hash")],
            ),
            SimpleNamespace(
                repo_id="unused/repo",
                revisions=[SimpleNamespace(commit_hash="unused-hash")],
            ),
        ]

    def delete_revisions(self, *hashes: str):
        self._log.append(("delete", hashes))
        return _DeleteStrategy(self._log, hashes)


class SFTBatchRunnerTest(unittest.TestCase):
    def test_build_run_matrix_expands_six_runs_and_applies_overrides(self):
        batch_config = _batch_config()

        with patch("src.batch_sft_runner.load_yaml", side_effect=_load_yaml_stub):
            run_plans = build_run_matrix(batch_config, config_dir=Path("/tmp/project"))

        self.assertEqual(len(run_plans), 6)
        self.assertEqual(
            [(plan["dataset_slug"], plan["model_slug"]) for plan in run_plans],
            [
                ("hh-helpful-base", "llama3-8b"),
                ("hh-helpful-base", "qwen3-8b"),
                ("hh-harmless-base", "llama3-8b"),
                ("hh-harmless-base", "qwen3-8b"),
                ("ultrachat", "llama3-8b"),
                ("ultrachat", "qwen3-8b"),
            ],
        )

        first_cfg = run_plans[0]["config"]
        second_cfg = run_plans[1]["config"]
        last_cfg = run_plans[-1]["config"]

        self.assertEqual(first_cfg["policy_name"], "meta-llama/Meta-Llama-3-8B")
        self.assertEqual(first_cfg["ref_name"], "meta-llama/Meta-Llama-3-8B")
        self.assertEqual(first_cfg["dataset"]["chat_template_name"], "llama3")
        self.assertEqual(
            first_cfg["sft_training"]["fsdp"]["transformer_layer_cls_to_wrap"],
            "LlamaDecoderLayer",
        )
        self.assertEqual(
            first_cfg["sft_training"]["hub_model_id"],
            "W-61/hh-helpful-base-llama3-8b-sft",
        )
        self.assertEqual(
            first_cfg["sft_training"]["save_dir"],
            "batch_sft_runs/hh-helpful-base-llama3-8b-sft",
        )
        self.assertFalse(first_cfg["sft_training"]["push_to_hub"])

        self.assertEqual(second_cfg["policy_name"], "Qwen/Qwen3-8B")
        self.assertEqual(second_cfg["ref_name"], "Qwen/Qwen3-8B")
        self.assertEqual(second_cfg["dataset"]["chat_template_name"], "qwen3")
        self.assertEqual(
            second_cfg["sft_training"]["fsdp"]["transformer_layer_cls_to_wrap"],
            "Qwen3DecoderLayer",
        )

        self.assertEqual(last_cfg["policy_name"], "Qwen/Qwen3-8B")
        self.assertNotIn("ref_name", last_cfg)
        self.assertEqual(last_cfg["dataset"]["chat_template_name"], "qwen3")

    def test_run_batch_sft_calls_train_push_and_cleanup_in_order(self):
        log: list[str] = []
        run_plans = [
            {
                "base_config_path": "config_sft_hh_helpful_base.yaml",
                "dataset_slug": "hh-helpful-base",
                "model_slug": "llama3-8b",
                "policy_name": "meta-llama/Meta-Llama-3-8B",
                "hub_model_id": "W-61/hh-helpful-base-llama3-8b-sft",
                "config": {
                    "policy_name": "meta-llama/Meta-Llama-3-8B",
                    "dataset": {"dataset_name": "Anthropic/hh-rlhf"},
                    "sft_training": {
                        "hub_model_id": "W-61/hh-helpful-base-llama3-8b-sft",
                        "save_dir": "batch_sft_runs/hh-helpful-base-llama3-8b-sft",
                    },
                },
            }
        ]

        def _fake_train(run_config: dict):
            hub_model_id = run_config["sft_training"]["hub_model_id"]
            log.append(f"train:{hub_model_id}")
            return _DummyTrainer(log, hub_model_id)

        def _fake_cleanup(*, trainer, run_config, cleanup_config):
            del trainer, cleanup_config
            log.append(f"cleanup:{run_config['sft_training']['hub_model_id']}")

        with (
            patch("src.batch_sft_runner.build_run_matrix", return_value=run_plans),
            patch("src.batch_sft_runner.run_sft_training", side_effect=_fake_train),
            patch("src.batch_sft_runner.cleanup_run_artifacts", side_effect=_fake_cleanup),
        ):
            exit_code = run_batch_sft(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            log,
            [
                "train:W-61/hh-helpful-base-llama3-8b-sft",
                "push:W-61/hh-helpful-base-llama3-8b-sft",
                "cleanup:W-61/hh-helpful-base-llama3-8b-sft",
            ],
        )

    def test_cleanup_run_artifacts_deletes_run_output_and_both_cache_targets(self):
        log: list[tuple[str, tuple[str, ...]]] = []
        run_config = {
            "policy_name": "meta-llama/Meta-Llama-3-8B",
            "dataset": {"dataset_name": "Anthropic/hh-rlhf"},
            "sft_training": {"save_dir": "batch_sft_runs/hh-helpful-base-llama3-8b-sft"},
        }
        trainer = _DummyTrainer([], "unused")

        with (
            patch("src.batch_sft_runner._finish_wandb_run"),
            patch("src.batch_sft_runner.gc.collect"),
            patch("src.batch_sft_runner._clear_cuda_memory"),
            patch("src.batch_sft_runner.shutil.rmtree") as mock_rmtree,
            patch(
                "src.batch_sft_runner.scan_cache_dir",
                return_value=_CacheInfo(log),
            ),
        ):
            cleanup_run_artifacts(
                trainer=trainer,
                run_config=run_config,
                cleanup_config={
                    "delete_run_output": True,
                    "delete_hf_download_cache": True,
                },
            )

        mock_rmtree.assert_called_once_with(
            Path("batch_sft_runs/hh-helpful-base-llama3-8b-sft"),
            ignore_errors=True,
        )
        self.assertIsNone(trainer.model)
        self.assertEqual(log[0], ("delete", ("model-hash", "dataset-hash")))
        self.assertEqual(log[1], ("execute", ("model-hash", "dataset-hash")))

    def test_run_batch_sft_fails_fast_and_preserves_failed_artifacts(self):
        run_plans = []
        for index in range(6):
            run_plans.append(
                {
                    "base_config_path": f"config-{index}.yaml",
                    "dataset_slug": f"dataset-{index}",
                    "model_slug": "llama3-8b",
                    "policy_name": "meta-llama/Meta-Llama-3-8B",
                    "hub_model_id": f"W-61/dataset-{index}-llama3-8b-sft",
                    "config": {
                        "policy_name": "meta-llama/Meta-Llama-3-8B",
                        "dataset": {"dataset_name": "Anthropic/hh-rlhf"},
                        "sft_training": {
                            "hub_model_id": f"W-61/dataset-{index}-llama3-8b-sft",
                            "save_dir": f"batch_sft_runs/dataset-{index}-llama3-8b-sft",
                        },
                    },
                }
            )

        cleanup_log: list[str] = []
        call_count = {"value": 0}

        def _fake_train(run_config: dict):
            call_count["value"] += 1
            if call_count["value"] == 3:
                raise RuntimeError("boom")
            return _DummyTrainer([], run_config["sft_training"]["hub_model_id"])

        def _fake_cleanup(*, trainer, run_config, cleanup_config):
            del trainer, cleanup_config
            cleanup_log.append(run_config["sft_training"]["save_dir"])

        buffer = io.StringIO()
        with (
            patch("src.batch_sft_runner.build_run_matrix", return_value=run_plans),
            patch("src.batch_sft_runner.run_sft_training", side_effect=_fake_train),
            patch("src.batch_sft_runner.cleanup_run_artifacts", side_effect=_fake_cleanup),
            redirect_stdout(buffer),
        ):
            exit_code = run_batch_sft(_batch_config(), config_dir=Path("/tmp/project"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(call_count["value"], 3)
        self.assertEqual(
            cleanup_log,
            [
                "batch_sft_runs/dataset-0-llama3-8b-sft",
                "batch_sft_runs/dataset-1-llama3-8b-sft",
            ],
        )
        self.assertIn("[SFT-BATCH] Failed dataset-2 / llama3-8b (3/6): boom", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()

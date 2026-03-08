"""Unit tests for HH-based Beta, Margin, and Epsilon DPO CLI entry points."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from datasets import Dataset
from trl import DPOConfig

from src.cli import main_beta_dpo, main_e_dpo, main_margin_dpo
from src.trainers.beta_dpo_trainer import BetaDPOConfig
from src.trainers.epsilon_dpo_trainer import EpsilonDPOConfig


def _base_dpo_config() -> dict:
    return {
        "policy_name": "dummy/policy",
        "ref_name": "dummy/ref",
        "precision": "fp32",
        "dataset": {
            "dataset_name": "Anthropic/hh-rlhf",
            "data_dir": "helpful-base",
            "generated_data": False,
            "chat_template": False,
            "subset": "train",
            "val_ratio": 0.5,
            "seed": 7,
            "max_len": 64,
        },
        "dpo_training": {
            "epochs": 1,
            "batch_size": 2,
            "eval_batch_size": 2,
            "learning_rate": 1e-6,
            "log_steps": 1,
            "eval_steps": 1,
            "save_steps": 1,
            "gradient_accumulation": 2,
            "max_grad_norm": 1.0,
            "warmup_steps": 0,
            "report": None,
            "run_name": "hh-dpo-test",
            "save_dir": "tmp_dpo",
        },
    }


def _dpo_fsdp_config() -> dict:
    return {
        "enabled": True,
        "mode": "full_shard",
        "auto_wrap_policy": "transformer_based_wrap",
        "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "min_num_params": None,
        "version": 1,
        "offload_params": False,
        "backward_prefetch": "no_prefetch",
        "forward_prefetch": False,
        "use_orig_params": True,
        "cpu_ram_efficient_loading": True,
        "sync_module_states": True,
        "activation_checkpointing": False,
        "state_dict_type": "FULL_STATE_DICT",
    }


def _raw_hh_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "chosen": "\n\nHuman: Hi\n\nAssistant: Hello!",
                "rejected": "\n\nHuman: Hi\n\nAssistant: Go away.",
            },
            {
                "chosen": "\n\nHuman: 2+2?\n\nAssistant: 4.",
                "rejected": "\n\nHuman: 2+2?\n\nAssistant: 5.",
            },
            {
                "chosen": "\n\nHuman: Say bye\n\nAssistant: Bye!",
                "rejected": "\n\nHuman: Say bye\n\nAssistant: No.",
            },
            {
                "chosen": "\n\nHuman: Name a color\n\nAssistant: Blue.",
                "rejected": "\n\nHuman: Name a color\n\nAssistant: Maybe.",
            },
        ]
    )


class _DummyParam:
    def __init__(self):
        self.requires_grad_value = True

    def requires_grad_(self, value):
        self.requires_grad_value = value
        return self


class _DummyModel:
    def __init__(self):
        self._params = [_DummyParam()]
        self.eval_called = False
        self.pushed_to_hub = None

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return self._params

    def push_to_hub(self, hub_model_id):
        self.pushed_to_hub = hub_model_id


class _DummyTokenizer:
    def __init__(self):
        self.pad_token_id = None
        self.eos_token = "</s>"
        self.pad_token = None


class _DummyTrainer:
    instances = []

    def __init__(self, *args, **kwargs):
        self.model = kwargs["model"]
        self.ref_model = kwargs["ref_model"]
        self.args = kwargs["args"]
        self.train_dataset = kwargs["train_dataset"]
        self.eval_dataset = kwargs["eval_dataset"]
        self.processing_class = kwargs.get("processing_class")
        self.margin_log_path = kwargs.get("margin_log_path")
        self.train_called = False
        self.saved_path = None
        self.push_to_hub_called = False
        _DummyTrainer.instances.append(self)

    def train(self):
        self.train_called = True

    def save_model(self, path):
        self.saved_path = path

    def push_to_hub(self):
        self.push_to_hub_called = True


class _DummyWandbRun:
    def __init__(self, project_url: str, run_url: str):
        self.project_url = project_url
        self.url = run_url


class _DummyWandbModule:
    def __init__(self):
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return _DummyWandbRun(
            project_url="https://wandb.example/project",
            run_url="https://wandb.example/run",
        )


class DPOCliTest(unittest.TestCase):
    def setUp(self):
        _DummyTrainer.instances.clear()

    def _run_main(self, main_fn, config: dict, trainer_patch_path: str, argv: list[str]):
        raw_dataset = _raw_hh_dataset()
        policy = _DummyModel()
        ref_model = _DummyModel()

        def _fake_load_dataset(path, data_dir=None, split=None):
            self.assertEqual(path, config["dataset"]["dataset_name"])
            self.assertEqual(data_dir, config["dataset"]["data_dir"])
            self.assertEqual(split, config["dataset"]["subset"])
            return raw_dataset

        with (
            patch("src.cli.load_yaml", return_value=config),
            patch("src.cli.load_dataset", side_effect=_fake_load_dataset),
            patch(
                "src.cli.AutoTokenizer.from_pretrained",
                side_effect=lambda *args, **kwargs: _DummyTokenizer(),
            ),
            patch(
                "src.cli.AutoModelForCausalLM.from_pretrained",
                side_effect=[policy, ref_model],
            ),
            patch(trainer_patch_path, _DummyTrainer),
            patch.object(sys, "argv", argv),
        ):
            main_fn()

        return _DummyTrainer.instances[-1], ref_model

    @staticmethod
    def _fsdp_tokens(fsdp_value):
        if isinstance(fsdp_value, str):
            return set(fsdp_value.split())
        return {
            str(option.value if hasattr(option, "value") else option)
            for option in fsdp_value
        }

    def test_main_beta_dpo_builds_beta_config_and_hh_triplets(self):
        config = _base_dpo_config()
        config["beta_dpo"] = {
            "beta": 0.2,
            "rho": 0.7,
            "alpha": 1.3,
            "ema_momentum": 0.8,
            "beta_min": 0.05,
            "sync_global_mask": False,
        }

        trainer, ref_model = self._run_main(
            main_beta_dpo,
            config,
            "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
            ["train-beta-dpo", "--config", "config_beta_dpo.yaml", "--output_dir", "beta_out"],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "beta_out/final")
        self.assertIsInstance(trainer.args, BetaDPOConfig)
        self.assertEqual(trainer.args.beta, 0.2)
        self.assertEqual(trainer.args.rho, 0.7)
        self.assertEqual(trainer.args.alpha, 1.3)
        self.assertEqual(trainer.args.ema_momentum, 0.8)
        self.assertEqual(trainer.args.beta_min, 0.05)
        self.assertFalse(trainer.args.sync_global_mask)
        self.assertEqual(trainer.args.max_length, 64)
        self.assertEqual(set(trainer.train_dataset.column_names), {"prompt", "chosen", "rejected"})
        self.assertEqual(set(trainer.eval_dataset.column_names), {"prompt", "chosen", "rejected"})
        self.assertTrue(ref_model.eval_called)
        self.assertFalse(ref_model.parameters()[0].requires_grad_value)

    def test_main_beta_dpo_accepts_torchrun_local_rank_argument(self):
        config = _base_dpo_config()

        trainer, _ = self._run_main(
            main_beta_dpo,
            config,
            "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
            [
                "train-beta-dpo",
                "--config",
                "config_beta_dpo.yaml",
                "--output_dir",
                "beta_out",
                "--local-rank",
                "3",
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "beta_out/final")

    def test_main_margin_dpo_passes_margin_log_path_and_standard_dpo_config(self):
        config = _base_dpo_config()
        config["margin_log"] = {"log_dir": "logs/custom_margins"}

        trainer, _ = self._run_main(
            main_margin_dpo,
            config,
            "src.trainers.margin_dpo_trainer.MarginDPOTrainer",
            [
                "train-margin-dpo",
                "--config",
                "config_margin_dpo.yaml",
                "--output_dir",
                "margin_out",
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "margin_out/final")
        self.assertIsInstance(trainer.args, DPOConfig)
        self.assertEqual(trainer.margin_log_path, "logs/custom_margins")
        self.assertEqual(trainer.args.max_length, 64)
        self.assertEqual(set(trainer.train_dataset.column_names), {"prompt", "chosen", "rejected"})
        self.assertEqual(set(trainer.eval_dataset.column_names), {"prompt", "chosen", "rejected"})

    def test_main_margin_dpo_accepts_torchrun_local_rank_argument(self):
        config = _base_dpo_config()

        trainer, _ = self._run_main(
            main_margin_dpo,
            config,
            "src.trainers.margin_dpo_trainer.MarginDPOTrainer",
            [
                "train-margin-dpo",
                "--config",
                "config_margin_dpo.yaml",
                "--output_dir",
                "margin_out",
                "--local_rank",
                "2",
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "margin_out/final")

    def test_main_e_dpo_builds_epsilon_config_and_hh_triplets(self):
        config = _base_dpo_config()
        config["e_dpo"] = {
            "beta": 0.15,
            "epsilon": 0.02,
        }

        trainer, ref_model = self._run_main(
            main_e_dpo,
            config,
            "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
            ["train-e-dpo", "--config", "config_e_dpo.yaml", "--output_dir", "e_out"],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "e_out/final")
        self.assertIsInstance(trainer.args, EpsilonDPOConfig)
        self.assertEqual(trainer.args.beta, 0.15)
        self.assertEqual(trainer.args.epsilon, 0.02)
        self.assertEqual(trainer.args.max_length, 64)
        self.assertEqual(trainer.args.output_dir, "e_out")
        self.assertEqual(set(trainer.train_dataset.column_names), {"prompt", "chosen", "rejected"})
        self.assertEqual(set(trainer.eval_dataset.column_names), {"prompt", "chosen", "rejected"})
        self.assertIsInstance(trainer.processing_class, _DummyTokenizer)
        self.assertTrue(ref_model.eval_called)
        self.assertFalse(ref_model.parameters()[0].requires_grad_value)

    def test_main_e_dpo_uses_trainer_push_to_hub_when_hub_model_id_is_set(self):
        config = _base_dpo_config()
        config["dpo_training"]["hub_model_id"] = "user/e-dpo-test"
        config["e_dpo"] = {
            "beta": 0.15,
            "epsilon": 0.02,
        }

        trainer, _ = self._run_main(
            main_e_dpo,
            config,
            "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
            ["train-e-dpo", "--config", "config_e_dpo.yaml", "--output_dir", "e_out"],
        )

        self.assertEqual(trainer.args.hub_model_id, "user/e-dpo-test")
        self.assertTrue(trainer.push_to_hub_called)
        self.assertIsNone(trainer.model.pushed_to_hub)

    def test_main_e_dpo_accepts_torchrun_local_rank_argument(self):
        config = _base_dpo_config()

        trainer, _ = self._run_main(
            main_e_dpo,
            config,
            "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
            [
                "train-e-dpo",
                "--config",
                "config_e_dpo.yaml",
                "--output_dir",
                "e_out",
                "--local-rank",
                "1",
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "e_out/final")

    def test_main_e_dpo_initializes_wandb_with_project_and_run_name_and_prints_urls(self):
        config = _base_dpo_config()
        config["dpo_training"]["wandb_project"] = "wandb_dpo_project"
        config["e_dpo"] = {
            "beta": 0.15,
            "epsilon": 0.02,
        }
        stdout = io.StringIO()
        wandb_module = _DummyWandbModule()

        with patch.dict(sys.modules, {"wandb": wandb_module}):
            with redirect_stdout(stdout):
                trainer, _ = self._run_main(
                    main_e_dpo,
                    config,
                    "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
                    [
                        "train-e-dpo",
                        "--config",
                        "config_e_dpo.yaml",
                        "--output_dir",
                        "e_out",
                    ],
                )

        self.assertTrue(trainer.train_called)
        self.assertEqual(
            wandb_module.init_kwargs["project"],
            "wandb_dpo_project",
        )
        self.assertEqual(
            wandb_module.init_kwargs["name"],
            "hh-dpo-test",
        )
        self.assertIn(
            "[DPO] wandb project=wandb_dpo_project run_name=hh-dpo-test",
            stdout.getvalue(),
        )
        self.assertIn(
            "[DPO] wandb project_url=https://wandb.example/project",
            stdout.getvalue(),
        )
        self.assertIn(
            "[DPO] wandb run_url=https://wandb.example/run",
            stdout.getvalue(),
        )

    def test_main_beta_dpo_sets_fsdp_args_and_save_only_model_safety_override(self):
        config = _base_dpo_config()
        config["dpo_training"]["gradient_checkpointing"] = False
        config["dpo_training"]["load_best_model_at_end"] = True
        config["dpo_training"]["save_only_model"] = True
        config["dpo_training"]["fsdp"] = _dpo_fsdp_config()

        trainer, _ = self._run_main(
            main_beta_dpo,
            config,
            "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
            ["train-beta-dpo", "--config", "config_beta_dpo.yaml", "--output_dir", "beta_out"],
        )

        self.assertEqual(self._fsdp_tokens(trainer.args.fsdp), {"full_shard", "auto_wrap"})
        self.assertIn("transformer_layer_cls_to_wrap", trainer.args.fsdp_config)
        self.assertEqual(
            trainer.args.fsdp_config["transformer_layer_cls_to_wrap"],
            ["LlamaDecoderLayer"],
        )
        self.assertFalse(trainer.args.save_only_model)
        self.assertFalse(trainer.args.gradient_checkpointing)

    def test_main_margin_dpo_sets_fsdp_args(self):
        config = _base_dpo_config()
        config["dpo_training"]["fsdp"] = _dpo_fsdp_config()

        trainer, _ = self._run_main(
            main_margin_dpo,
            config,
            "src.trainers.margin_dpo_trainer.MarginDPOTrainer",
            [
                "train-margin-dpo",
                "--config",
                "config_margin_dpo.yaml",
                "--output_dir",
                "margin_out",
            ],
        )

        self.assertEqual(self._fsdp_tokens(trainer.args.fsdp), {"full_shard", "auto_wrap"})
        self.assertIn("transformer_layer_cls_to_wrap", trainer.args.fsdp_config)
        self.assertEqual(
            trainer.args.fsdp_config["transformer_layer_cls_to_wrap"],
            ["LlamaDecoderLayer"],
        )

    def test_main_e_dpo_sets_fsdp_args_and_save_only_model_safety_override(self):
        config = _base_dpo_config()
        config["dpo_training"]["gradient_checkpointing"] = False
        config["dpo_training"]["load_best_model_at_end"] = True
        config["dpo_training"]["save_only_model"] = True
        config["dpo_training"]["fsdp"] = _dpo_fsdp_config()

        trainer, _ = self._run_main(
            main_e_dpo,
            config,
            "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
            ["train-e-dpo", "--config", "config_e_dpo.yaml", "--output_dir", "e_out"],
        )

        self.assertEqual(self._fsdp_tokens(trainer.args.fsdp), {"full_shard", "auto_wrap"})
        self.assertIn("transformer_layer_cls_to_wrap", trainer.args.fsdp_config)
        self.assertEqual(
            trainer.args.fsdp_config["transformer_layer_cls_to_wrap"],
            ["LlamaDecoderLayer"],
        )
        self.assertFalse(trainer.args.save_only_model)
        self.assertFalse(trainer.args.gradient_checkpointing)

    def test_main_beta_dpo_invalid_fsdp_auto_wrap_policy_raises_value_error(self):
        config = _base_dpo_config()
        config["dpo_training"]["fsdp"] = _dpo_fsdp_config()
        config["dpo_training"]["fsdp"]["auto_wrap_policy"] = "invalid_policy"

        with self.assertRaisesRegex(
            ValueError,
            "dpo_training\\.fsdp\\.auto_wrap_policy",
        ):
            self._run_main(
                main_beta_dpo,
                config,
                "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
                ["train-beta-dpo", "--config", "config_beta_dpo.yaml", "--output_dir", "beta_out"],
            )

    def test_main_beta_dpo_fsdp_layer_cls_and_min_num_params_are_mutually_exclusive(self):
        config = _base_dpo_config()
        config["dpo_training"]["fsdp"] = _dpo_fsdp_config()
        config["dpo_training"]["fsdp"]["min_num_params"] = 1234

        with self.assertRaisesRegex(
            ValueError,
            "dpo_training\\.fsdp\\.transformer_layer_cls_to_wrap and "
            "dpo_training\\.fsdp\\.min_num_params are mutually exclusive",
        ):
            self._run_main(
                main_beta_dpo,
                config,
                "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
                ["train-beta-dpo", "--config", "config_beta_dpo.yaml", "--output_dir", "beta_out"],
            )

    def test_main_beta_dpo_invalid_dataset_max_len_raises_value_error(self):
        config = _base_dpo_config()
        config["dataset"]["max_len"] = 0

        with self.assertRaisesRegex(ValueError, "dataset\\.max_len must be > 0"):
            self._run_main(
                main_beta_dpo,
                config,
                "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
                ["train-beta-dpo", "--config", "config_beta_dpo.yaml", "--output_dir", "beta_out"],
            )

    def test_main_e_dpo_invalid_dataset_max_len_raises_value_error(self):
        config = _base_dpo_config()
        config["dataset"]["max_len"] = 0

        with self.assertRaisesRegex(ValueError, "dataset\\.max_len must be > 0"):
            self._run_main(
                main_e_dpo,
                config,
                "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
                ["train-e-dpo", "--config", "config_e_dpo.yaml", "--output_dir", "e_out"],
            )

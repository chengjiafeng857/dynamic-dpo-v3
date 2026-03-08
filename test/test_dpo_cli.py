"""Unit tests for DPO CLI entry points and preference dataset builders."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from typing import Callable, Optional
from unittest.mock import patch

import trl
from datasets import Dataset
from trl import DPOConfig

from src.cli import _finalize_dpo_training, main_beta_dpo, main_e_dpo, main_margin_dpo
from src.data.ultrafeedback_dataset import build_ultrafeedback_preference_dataset
from src.trainers.beta_dpo_trainer import BetaDPOConfig
from src.trainers.epsilon_dpo_trainer import EpsilonDPOConfig


DUMMY_ASSISTANT_SUFFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"


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
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "</s>"
        self.pad_token = None
        self.chat_template = None

    @staticmethod
    def _encode_text(text):
        return [ord(ch) + 2 for ch in text]

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        **kwargs,
    ):
        rendered = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            rendered.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        text = "".join(rendered)
        if add_generation_prompt:
            text += DUMMY_ASSISTANT_SUFFIX
        if not tokenize:
            return text

        input_ids = self._encode_text(text)
        if return_dict:
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        return input_ids

    def __call__(self, text, **kwargs):
        input_ids = self._encode_text(str(text))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


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
        self.saved_paths = []
        self.push_to_hub_called = False
        self.is_fsdp_enabled = False
        self.accelerator = None
        _DummyTrainer.instances.append(self)

    def train(self):
        self.train_called = True

    def save_model(self, path):
        self.saved_path = path
        self.saved_paths.append(path)

    def push_to_hub(self):
        self.push_to_hub_called = True


class _DummyFSDPPlugin:
    def __init__(self, state_dict_type):
        self.state_dict_type = state_dict_type
        self.set_state_dict_type_calls = []

    def set_state_dict_type(self, state_dict_type=None):
        self.state_dict_type = state_dict_type
        self.set_state_dict_type_calls.append(state_dict_type)


class _DummyAcceleratorState:
    def __init__(self, fsdp_plugin):
        self.fsdp_plugin = fsdp_plugin


class _DummyAccelerator:
    def __init__(self, fsdp_plugin):
        self.state = _DummyAcceleratorState(fsdp_plugin)


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

    def _run_main(
        self,
        main_fn,
        config: dict,
        trainer_patch_path: str,
        argv: list[str],
        load_dataset_side_effect: Optional[Callable] = None,
    ):
        raw_dataset = _raw_hh_dataset()
        policy = _DummyModel()
        ref_model = _DummyModel()

        def _fake_load_dataset(path, data_dir=None, split=None):
            self.assertEqual(path, config["dataset"]["dataset_name"])
            self.assertEqual(data_dir, config["dataset"]["data_dir"])
            self.assertEqual(split, config["dataset"]["subset"])
            return raw_dataset

        effective_load_dataset = (
            load_dataset_side_effect
            if load_dataset_side_effect is not None
            else _fake_load_dataset
        )

        with (
            patch("src.cli.load_yaml", return_value=config),
            patch("src.cli.load_dataset", side_effect=effective_load_dataset),
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
            ["train-beta-dpo", "--config", "config_beta_dpo.yaml"],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "tmp_dpo/final")
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
                "--local-rank",
                "3",
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "tmp_dpo/final")

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
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "tmp_dpo/final")
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
                "--local_rank",
                "2",
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "tmp_dpo/final")

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
            ["train-e-dpo", "--config", "config_e_dpo.yaml"],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "tmp_dpo/final")
        self.assertIsInstance(trainer.args, EpsilonDPOConfig)
        self.assertEqual(trainer.args.beta, 0.15)
        self.assertEqual(trainer.args.epsilon, 0.02)
        self.assertEqual(trainer.args.max_length, 64)
        self.assertEqual(trainer.args.output_dir, "tmp_dpo")
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

        with (
            patch("src.cli.create_repo") as create_repo_mock,
            patch("src.cli.upload_folder") as upload_folder_mock,
        ):
            trainer, _ = self._run_main(
                main_e_dpo,
                config,
                "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
                ["train-e-dpo", "--config", "config_e_dpo.yaml"],
            )

        self.assertEqual(trainer.args.hub_model_id, "user/e-dpo-test")
        self.assertFalse(trainer.push_to_hub_called)
        self.assertEqual(trainer.saved_paths, ["tmp_dpo/final"])
        self.assertEqual(trainer.saved_path, "tmp_dpo/final")
        self.assertIsNone(trainer.model.pushed_to_hub)
        create_repo_mock.assert_called_once_with(
            "user/e-dpo-test",
            token=None,
            private=None,
            exist_ok=True,
        )
        upload_folder_mock.assert_called_once_with(
            repo_id="user/e-dpo-test",
            folder_path="tmp_dpo/final",
            commit_message="End of training",
            token=None,
            revision=None,
        )

    def test_finalize_dpo_training_uses_configured_state_dict_type_for_final_save(self):
        trainer = _DummyTrainer(
            model=_DummyModel(),
            ref_model=_DummyModel(),
            args=DPOConfig(output_dir="tmp_dpo", hub_model_id="user/e-dpo-test"),
            train_dataset=Dataset.from_list([]),
            eval_dataset=Dataset.from_list([]),
        )
        fsdp_plugin = _DummyFSDPPlugin("SHARDED_STATE_DICT")
        trainer.is_fsdp_enabled = True
        trainer.accelerator = _DummyAccelerator(fsdp_plugin)

        with (
            patch("src.cli._is_main_process", return_value=True),
            patch("src.cli.create_repo") as create_repo_mock,
            patch("src.cli.upload_folder") as upload_folder_mock,
        ):
            _finalize_dpo_training(
                trainer,
                {
                    "save_dir": "tmp_dpo",
                    "hub_model_id": "user/e-dpo-test",
                },
            )

        self.assertEqual(trainer.saved_paths, ["tmp_dpo/final"])
        self.assertFalse(trainer.push_to_hub_called)
        self.assertEqual(
            fsdp_plugin.set_state_dict_type_calls,
            ["FULL_STATE_DICT", "SHARDED_STATE_DICT"],
        )
        self.assertEqual(fsdp_plugin.state_dict_type, "SHARDED_STATE_DICT")
        create_repo_mock.assert_called_once_with(
            "user/e-dpo-test",
            token=None,
            private=None,
            exist_ok=True,
        )
        upload_folder_mock.assert_called_once_with(
            repo_id="user/e-dpo-test",
            folder_path="tmp_dpo/final",
            commit_message="End of training",
            token=None,
            revision=None,
        )

    def test_finalize_dpo_training_uploads_save_dir_for_sharded_hub_source(self):
        trainer = _DummyTrainer(
            model=_DummyModel(),
            ref_model=_DummyModel(),
            args=DPOConfig(output_dir="tmp_dpo", hub_model_id="user/e-dpo-test"),
            train_dataset=Dataset.from_list([]),
            eval_dataset=Dataset.from_list([]),
        )
        fsdp_plugin = _DummyFSDPPlugin("SHARDED_STATE_DICT")
        trainer.is_fsdp_enabled = True
        trainer.accelerator = _DummyAccelerator(fsdp_plugin)

        with (
            patch("src.cli._is_main_process", return_value=True),
            patch("src.cli.create_repo") as create_repo_mock,
            patch("src.cli.upload_folder") as upload_folder_mock,
        ):
            _finalize_dpo_training(
                trainer,
                {
                    "save_dir": "tmp_dpo",
                    "hub_model_id": "user/e-dpo-test",
                    "hub_upload_source": "save_dir",
                },
            )

        self.assertEqual(trainer.saved_paths, ["tmp_dpo"])
        self.assertFalse(trainer.push_to_hub_called)
        self.assertEqual(
            fsdp_plugin.set_state_dict_type_calls,
            ["FULL_STATE_DICT", "SHARDED_STATE_DICT"],
        )
        self.assertEqual(fsdp_plugin.state_dict_type, "SHARDED_STATE_DICT")
        create_repo_mock.assert_called_once_with(
            "user/e-dpo-test",
            token=None,
            private=None,
            exist_ok=True,
        )
        upload_folder_mock.assert_called_once_with(
            repo_id="user/e-dpo-test",
            folder_path="tmp_dpo",
            commit_message="End of training",
            token=None,
            revision=None,
        )

    def test_main_e_dpo_requires_save_dir_in_config(self):
        config = _base_dpo_config()
        config["dpo_training"].pop("save_dir")
        config["e_dpo"] = {
            "beta": 0.15,
            "epsilon": 0.02,
        }

        with self.assertRaises(KeyError):
            self._run_main(
                main_e_dpo,
                config,
                "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
                ["train-e-dpo", "--config", "config_e_dpo.yaml"],
            )

    def test_main_beta_dpo_ultrafeedback_uses_explicit_train_eval_splits(self):
        config = _base_dpo_config()
        config["dataset"] = {
            "preference_source": "ultrafeedback_binarized",
            "dataset_name": "HuggingFaceH4/ultrafeedback_binarized",
            "generated_data": False,
            "chat_template": True,
            "chat_template_name": "llama3",
            "train_split": "train_prefs",
            "eval_split": "test_prefs",
            "max_len": 64,
        }
        config["beta_dpo"] = {"beta": 0.1}

        train_prefs = Dataset.from_list(
            [
                {
                    "prompt": "Say hi",
                    "chosen": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "Hi there."},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "No."},
                    ],
                },
                {
                    "prompt": "Bad row",
                    "chosen": [
                        {"role": "user", "content": "A"},
                        {"role": "assistant", "content": "B"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Different"},
                        {"role": "assistant", "content": "C"},
                    ],
                },
            ]
        )
        test_prefs = Dataset.from_list(
            [
                {
                    "prompt": "2+2",
                    "chosen": [
                        {"role": "user", "content": "2+2"},
                        {"role": "assistant", "content": "4"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "2+2"},
                        {"role": "assistant", "content": "5"},
                    ],
                }
            ]
        )
        load_calls = []

        def _fake_load_dataset(path, data_dir=None, split=None):
            load_calls.append((path, data_dir, split))
            if split == "train_prefs":
                return train_prefs
            if split == "test_prefs":
                return test_prefs
            raise AssertionError(f"Unexpected split request: {split}")

        trainer, _ = self._run_main(
            main_beta_dpo,
            config,
            "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
            ["train-beta-dpo", "--config", "config_beta_dpo.yaml"],
            load_dataset_side_effect=_fake_load_dataset,
        )

        self.assertEqual(
            load_calls,
            [
                (
                    "HuggingFaceH4/ultrafeedback_binarized",
                    None,
                    "train_prefs",
                ),
                (
                    "HuggingFaceH4/ultrafeedback_binarized",
                    None,
                    "test_prefs",
                ),
            ],
        )
        self.assertEqual(set(trainer.train_dataset.column_names), {"prompt", "chosen", "rejected"})
        self.assertEqual(set(trainer.eval_dataset.column_names), {"prompt", "chosen", "rejected"})
        self.assertEqual(len(trainer.train_dataset), 1)
        self.assertEqual(len(trainer.eval_dataset), 1)
        self.assertEqual(
            trainer.train_dataset[0]["prompt"],
            [{"role": "user", "content": "Say hi"}],
        )
        self.assertEqual(
            trainer.train_dataset[0]["chosen"],
            [{"role": "assistant", "content": "Hi there."}],
        )
        self.assertEqual(
            trainer.train_dataset[0]["rejected"],
            [{"role": "assistant", "content": "No."}],
        )

    def test_main_e_dpo_ultrafeedback_generated_data_is_not_supported(self):
        config = _base_dpo_config()
        config["dataset"] = {
            "preference_source": "ultrafeedback_binarized",
            "dataset_name": "HuggingFaceH4/ultrafeedback_binarized",
            "generated_data": True,
            "chat_template": True,
            "chat_template_name": "llama3",
            "train_split": "train_prefs",
            "eval_split": "test_prefs",
            "max_len": 64,
        }
        config["e_dpo"] = {"beta": 0.15, "epsilon": 0.02}

        with self.assertRaisesRegex(
            ValueError,
            "dataset\\.generated_data=true is not supported",
        ):
            self._run_main(
                main_e_dpo,
                config,
                "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
                ["train-e-dpo", "--config", "config_e_dpo.yaml"],
                load_dataset_side_effect=lambda *args, **kwargs: Dataset.from_list([]),
            )

    def test_main_margin_dpo_ultrafeedback_requires_explicit_splits(self):
        config = _base_dpo_config()
        config["dataset"] = {
            "preference_source": "ultrafeedback_binarized",
            "dataset_name": "HuggingFaceH4/ultrafeedback_binarized",
            "generated_data": False,
            "chat_template": True,
            "chat_template_name": "llama3",
            "max_len": 64,
        }

        with self.assertRaisesRegex(
            ValueError,
            "dataset\\.train_split and dataset\\.eval_split are required",
        ):
            self._run_main(
                main_margin_dpo,
                config,
                "src.trainers.margin_dpo_trainer.MarginDPOTrainer",
                ["train-margin-dpo", "--config", "config_margin_dpo.yaml"],
                load_dataset_side_effect=lambda *args, **kwargs: Dataset.from_list([]),
            )


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
                "--local-rank",
                "1",
            ],
        )

        self.assertTrue(trainer.train_called)
        self.assertEqual(trainer.saved_path, "tmp_dpo/final")

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
                    ["train-e-dpo", "--config", "config_e_dpo.yaml"],
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
            ["train-beta-dpo", "--config", "config_beta_dpo.yaml"],
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
            ["train-e-dpo", "--config", "config_e_dpo.yaml"],
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
                ["train-beta-dpo", "--config", "config_beta_dpo.yaml"],
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
                ["train-beta-dpo", "--config", "config_beta_dpo.yaml"],
            )

    def test_main_beta_dpo_invalid_dataset_max_len_raises_value_error(self):
        config = _base_dpo_config()
        config["dataset"]["max_len"] = 0

        with self.assertRaisesRegex(ValueError, "dataset\\.max_len must be > 0"):
            self._run_main(
                main_beta_dpo,
                config,
                "src.trainers.beta_dpo_trainer.BetaDPOTrainer",
                ["train-beta-dpo", "--config", "config_beta_dpo.yaml"],
            )

    def test_main_e_dpo_invalid_dataset_max_len_raises_value_error(self):
        config = _base_dpo_config()
        config["dataset"]["max_len"] = 0

        with self.assertRaisesRegex(ValueError, "dataset\\.max_len must be > 0"):
            self._run_main(
                main_e_dpo,
                config,
                "src.trainers.epsilon_dpo_trainer.EpsilonDPOTrainer",
                ["train-e-dpo", "--config", "config_e_dpo.yaml"],
            )


class UltraFeedbackDatasetBuilderTest(unittest.TestCase):
    def test_build_ultrafeedback_preference_dataset_builds_conversational_triplets(self):
        ds = Dataset.from_list(
            [
                {
                    "prompt": "Explain sky color.",
                    "chosen": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Explain sky color."},
                        {"role": "assistant", "content": "Rayleigh scattering."},
                    ],
                    "rejected": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Explain sky color."},
                        {"role": "assistant", "content": "I do not know."},
                    ],
                }
            ]
        )

        output = build_ultrafeedback_preference_dataset(
            ds,
            _DummyTokenizer(),
            model_name="dummy-llama3",
            chat_template_name="llama3",
        )

        self.assertEqual(set(output.column_names), {"prompt", "chosen", "rejected"})
        self.assertEqual(len(output), 1)
        self.assertEqual(
            output[0]["prompt"],
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Explain sky color."},
            ],
        )
        self.assertEqual(
            output[0]["chosen"],
            [{"role": "assistant", "content": "Rayleigh scattering."}],
        )
        self.assertEqual(
            output[0]["rejected"],
            [{"role": "assistant", "content": "I do not know."}],
        )

    def test_build_ultrafeedback_preference_dataset_keeps_trl_conversational_schema(self):
        ds = Dataset.from_list(
            [
                {
                    "prompt": "Say hi",
                    "chosen": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "Hi there."},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "No."},
                    ],
                }
            ]
        )

        output = build_ultrafeedback_preference_dataset(
            ds,
            _DummyTokenizer(),
            model_name="dummy-llama3",
            chat_template_name="llama3",
        )

        row = output[0]
        self.assertIsInstance(row["prompt"], list)
        self.assertIsInstance(row["chosen"], list)
        self.assertIsInstance(row["rejected"], list)
        self.assertEqual(row["prompt"][-1]["role"], "user")
        self.assertEqual(row["chosen"][0]["role"], "assistant")
        self.assertEqual(row["rejected"][0]["role"], "assistant")

    def test_build_ultrafeedback_preference_dataset_skips_local_template_setup_when_disabled(self):
        ds = Dataset.from_list(
            [
                {
                    "prompt": "Say hi",
                    "chosen": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "Hi there."},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "No."},
                    ],
                }
            ]
        )
        tokenizer = _DummyTokenizer()

        output = build_ultrafeedback_preference_dataset(
            ds,
            tokenizer,
            model_name="dummy-llama3",
            chat_template_name="llama3",
            add_chat_template=False,
        )

        self.assertEqual(len(output), 1)
        self.assertIsNone(tokenizer.chat_template)

    def test_build_ultrafeedback_preference_dataset_prepares_with_real_trl_path(self):
        ds = Dataset.from_list(
            [
                {
                    "prompt": "Say hi",
                    "chosen": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "Hi there."},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "No."},
                    ],
                }
            ]
        )
        tokenizer = _DummyTokenizer()
        preference_ds = build_ultrafeedback_preference_dataset(
            ds,
            tokenizer,
            model_name="dummy-llama3",
            chat_template_name="llama3",
        )
        trainer = object.__new__(trl.DPOTrainer)
        trainer._is_vlm = False

        prepared = trl.DPOTrainer._prepare_dataset(
            trainer,
            preference_ds,
            tokenizer,
            DPOConfig(output_dir="tmp_dpo"),
            "train",
        )

        row = prepared[0]
        self.assertIn("prompt_ids", row)
        self.assertIn("chosen_ids", row)
        self.assertIn("rejected_ids", row)
        self.assertGreater(len(row["prompt_ids"]), 0)
        self.assertGreater(len(row["chosen_ids"]), 0)
        self.assertGreater(len(row["rejected_ids"]), 0)

    def test_build_ultrafeedback_preference_dataset_filters_malformed_rows(self):
        ds = Dataset.from_list(
            [
                {
                    "prompt": "Keep me",
                    "chosen": [
                        {"role": "user", "content": "Keep me"},
                        {"role": "assistant", "content": "Chosen"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Keep me"},
                        {"role": "assistant", "content": "Rejected"},
                    ],
                },
                {
                    "prompt": "",
                    "chosen": [
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "content": "y"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "content": "z"},
                    ],
                },
                {
                    "prompt": "Bad suffix",
                    "chosen": [
                        {"role": "user", "content": "Bad suffix"},
                        {"role": "assistant", "content": "a"},
                    ],
                    "rejected": [
                        {"role": "assistant", "content": "b"},
                    ],
                },
                {
                    "prompt": "No assistant end",
                    "chosen": [
                        {"role": "user", "content": "No assistant end"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "No assistant end"},
                        {"role": "assistant", "content": "ok"},
                    ],
                },
            ]
        )

        output = build_ultrafeedback_preference_dataset(
            ds,
            _DummyTokenizer(),
            model_name="dummy-llama3",
            chat_template_name="llama3",
        )

        self.assertEqual(len(output), 1)
        self.assertEqual(
            output[0]["prompt"],
            [{"role": "user", "content": "Keep me"}],
        )
        self.assertEqual(
            output[0]["chosen"],
            [{"role": "assistant", "content": "Chosen"}],
        )
        self.assertEqual(
            output[0]["rejected"],
            [{"role": "assistant", "content": "Rejected"}],
        )

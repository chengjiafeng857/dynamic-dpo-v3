"""Smoke tests for the end-to-end SFT pipeline."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch
from datasets import Dataset

from src.data.templates import get_chat_template
from src.trainers.sft_trainer import run_sft_training


class _DummyTokenizer:
    def __init__(self):
        self.chat_template = None
        self.pad_token_id = None
        self.eos_token = "</s>"
        self.pad_token = None


class _DummyTrainer:
    instances = []

    def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.state = SimpleNamespace(best_model_checkpoint="checkpoint-1")
        self.saved = False
        _DummyTrainer.instances.append(self)

    def train(self):
        return None

    def save_model(self):
        self.saved = True


def _base_sft_config() -> dict:
    return {
        "policy_name": "dummy/policy",
        "precision": "fp32",
        "dataset": {
            "dataset_name": "Anthropic/hh-rlhf",
            "subset": "train",
            "val_ratio": 0.5,
            "seed": 7,
            "chat_template_name": "llama3",
        },
        "sft_training": {
            "learning_rate": 1e-5,
            "batch_size": 2,
            "eval_batch_size": 2,
            "gradient_accumulation": 2,
            "epochs": 1,
            "log_steps": 1,
            "eval_steps": 1,
            "save_steps": 1,
            "save_strategy": "best",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 2,
            "warmup_steps": 0,
            "lr_scheduler_type": "linear",
            "max_length": 64,
            "packing": False,
            "completion_only_loss": True,
            "save_dir": "tmp_sft_smoke",
            "save_only_model": True,
            "push_to_hub": False,
            "hub_model_id": None,
            "wandb_project": None,
            "fsdp": {
                "enabled": False,
                "mode": "full_shard auto_wrap",
                "version": 1,
                "offload_params": False,
                "auto_wrap_policy": "transformer_based_wrap",
                "transformer_layer_cls_to_wrap": None,
                "min_num_params": None,
                "backward_prefetch": "no_prefetch",
                "forward_prefetch": False,
                "use_orig_params": True,
                "cpu_ram_efficient_loading": True,
                "sync_module_states": True,
                "activation_checkpointing": False,
                "state_dict_type": "FULL_STATE_DICT",
            },
        },
    }


class SFTPipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        _DummyTrainer.instances.clear()

    def _patches(self, dataset_map: dict, config: dict):
        def _fake_load_dataset(path, name=None, split=None, data_dir=None, cache_dir=None):
            self.assertEqual(path, config["dataset"]["dataset_name"])
            self.assertEqual(cache_dir, config["dataset"].get("cache_dir"))
            if config["dataset"]["dataset_name"] == "Anthropic/hh-rlhf":
                expected_data_dir = config["dataset"].get(
                    "data_dir", config["dataset"].get("config_name")
                )
                self.assertEqual(data_dir, expected_data_dir)
                self.assertIsNone(name)
            else:
                self.assertEqual(name, config["dataset"].get("config_name"))
            return dataset_map[split]

        return (
            patch("src.trainers.sft_trainer.load_dataset", side_effect=_fake_load_dataset),
            patch(
                "src.data.sft_dataset.AutoTokenizer.from_pretrained",
                side_effect=lambda *args, **kwargs: _DummyTokenizer(),
            ),
            patch(
                "src.trainers.sft_trainer.AutoModelForCausalLM.from_pretrained",
                side_effect=lambda *args, **kwargs: object(),
            ),
        )

    def _run_pipeline(
        self,
        config: dict,
        dataset_map: dict,
        trainer_class=_DummyTrainer,
    ) -> tuple[_DummyTrainer, str]:
        buffer = io.StringIO()
        patches = self._patches(dataset_map, config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("src.trainers.sft_trainer.SFTTrainer", trainer_class),
            redirect_stdout(buffer),
        ):
            trainer = run_sft_training(config)
        return trainer, buffer.getvalue()

    def test_hh_pipeline_smoke_prints_prompt_completion_samples(self):
        config = _base_sft_config()
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Name a planet.\n\nAssistant: Mars."},
                {"chosen": "\n\nHuman: 2+2?\n\nAssistant: 4."},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )
        trainer, output = self._run_pipeline(config, {"train": raw_hh})

        self.assertIs(trainer, _DummyTrainer.instances[-1])
        self.assertTrue(trainer.saved)
        self.assertIn("[SFT] train_sample_keys=['prompt', 'completion']", output)
        self.assertIn("[SFT] eval_sample_keys=['prompt', 'completion']", output)
        self.assertIn("[SFT] best_model_checkpoint=checkpoint-1", output)
        self.assertEqual(trainer.args.gradient_accumulation_steps, 2)
        self.assertEqual(str(trainer.args.save_strategy), "SaveStrategy.BEST")
        self.assertTrue(trainer.args.load_best_model_at_end)

        sample = trainer.train_dataset[0]
        self.assertEqual(sample["prompt"][-1]["role"], "user")
        self.assertEqual(sample["completion"][0]["role"], "assistant")

    def test_hh_pipeline_supports_dataset_config_name(self):
        config = _base_sft_config()
        config["dataset"]["config_name"] = "harmless-base"
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        trainer, output = self._run_pipeline(config, {"train": raw_hh})

        self.assertTrue(trainer.saved)
        self.assertIn("[SFT] mode=hh", output)

    def test_hh_pipeline_supports_dataset_data_dir(self):
        config = _base_sft_config()
        config["dataset"]["data_dir"] = "helpful-base"
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        trainer, output = self._run_pipeline(config, {"train": raw_hh})

        self.assertTrue(trainer.saved)
        self.assertIn("[SFT] mode=hh", output)

    def test_hh_pipeline_supports_dataset_cache_dir(self):
        config = _base_sft_config()
        config["dataset"]["cache_dir"] = "/tmp/shared-hf-cache"
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        trainer, output = self._run_pipeline(config, {"train": raw_hh})

        self.assertTrue(trainer.saved)
        self.assertIn("[SFT] mode=hh", output)

    def test_hh_pipeline_preview_logs_before_trainer_construction(self):
        config = _base_sft_config()
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )
        buffer = io.StringIO()
        test_case = self

        class _AssertingTrainer(_DummyTrainer):
            def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
                current_output = buffer.getvalue()
                test_case.assertIn("[SFT] train_sample=", current_output)
                test_case.assertIn("[SFT] eval_sample=", current_output)
                super().__init__(model, args, train_dataset, eval_dataset, processing_class)

        patches = self._patches({"train": raw_hh}, config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("src.trainers.sft_trainer.SFTTrainer", _AssertingTrainer),
            redirect_stdout(buffer),
        ):
            run_sft_training(config)

    def test_ultrachat_pipeline_smoke_messages_mode(self):
        config = _base_sft_config()
        config["dataset"]["dataset_name"] = "HuggingFaceH4/ultrachat_200k"
        config["dataset"]["subset"] = "train_sft"
        config["dataset"]["eval_subset"] = "test_sft"
        config["sft_training"]["completion_only_loss"] = False
        config["sft_training"]["packing"] = True
        config["sft_training"]["lr_scheduler_type"] = "cosine"

        train_uc = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Tell me a joke"},
                        {"role": "assistant", "content": "Why did the robot cross the road?"},
                    ]
                }
            ]
        )
        eval_uc = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "What is 1+1?"},
                        {"role": "assistant", "content": "2"},
                    ]
                }
            ]
        )

        trainer, output = self._run_pipeline(
            config,
            {"train_sft": train_uc, "test_sft": eval_uc},
        )
        self.assertIn("[SFT] train_sample_keys=['messages']", output)
        self.assertIn("[SFT] eval_sample_keys=['messages']", output)
        self.assertEqual(trainer.train_dataset.column_names, ["messages"])

    def test_ultrachat_pipeline_smoke_completion_only_mode(self):
        config = _base_sft_config()
        config["dataset"]["dataset_name"] = "HuggingFaceH4/ultrachat_200k"
        config["dataset"]["subset"] = "train_sft"
        config["dataset"]["eval_subset"] = "test_sft"
        config["sft_training"]["completion_only_loss"] = True
        config["sft_training"]["packing"] = False
        config["sft_training"]["lr_scheduler_type"] = "cosine"

        train_uc = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Question 1"},
                        {"role": "assistant", "content": "Answer 1"},
                        {"role": "user", "content": "Question 2"},
                        {"role": "assistant", "content": "Answer 2"},
                    ]
                }
            ]
        )
        eval_uc = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Question eval"},
                        {"role": "assistant", "content": "Answer eval"},
                    ]
                }
            ]
        )

        trainer, output = self._run_pipeline(
            config,
            {"train_sft": train_uc, "test_sft": eval_uc},
        )
        self.assertIn("[SFT] train_sample_keys=['prompt', 'completion']", output)
        self.assertIn("[SFT] eval_sample_keys=['prompt', 'completion']", output)
        self.assertEqual(set(trainer.train_dataset.column_names), {"prompt", "completion"})

    def test_ultrachat_pipeline_supports_dataset_cache_dir(self):
        config = _base_sft_config()
        config["dataset"]["dataset_name"] = "HuggingFaceH4/ultrachat_200k"
        config["dataset"]["subset"] = "train_sft"
        config["dataset"]["eval_subset"] = "test_sft"
        config["dataset"]["cache_dir"] = "/tmp/shared-hf-cache"
        config["sft_training"]["completion_only_loss"] = False
        config["sft_training"]["packing"] = True

        train_uc = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Tell me a joke"},
                        {"role": "assistant", "content": "Why did the robot cross the road?"},
                    ]
                }
            ]
        )
        eval_uc = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "What is 1+1?"},
                        {"role": "assistant", "content": "2"},
                    ]
                }
            ]
        )

        trainer, output = self._run_pipeline(
            config,
            {"train_sft": train_uc, "test_sft": eval_uc},
        )
        self.assertTrue(trainer.saved)
        self.assertIn("[SFT] mode=ultrachat", output)

    def test_qwen_pipeline_smoke_with_explicit_chat_template(self):
        config = _base_sft_config()
        config["policy_name"] = "Qwen/Qwen3-8B"
        config["dataset"]["chat_template_name"] = "qwen3"
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        trainer, _ = self._run_pipeline(config, {"train": raw_hh})

        self.assertEqual(trainer.processing_class.chat_template, get_chat_template("qwen3"))

    def test_qwen_pipeline_smoke_infers_chat_template(self):
        config = _base_sft_config()
        config["policy_name"] = "Qwen/Qwen3-8B"
        config["dataset"].pop("chat_template_name", None)
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        trainer, _ = self._run_pipeline(config, {"train": raw_hh})

        self.assertEqual(trainer.processing_class.chat_template, get_chat_template("qwen3"))

    def test_attn_implementation_is_forwarded_to_model_loader(self):
        config = _base_sft_config()
        config["precision"] = "bf16"
        config["sft_training"]["attn_implementation"] = "flash_attention_2"
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        captured_kwargs = {}

        def _fake_model_loader(*args, **kwargs):
            del args
            captured_kwargs.update(kwargs)
            return object()

        buffer = io.StringIO()
        patches = self._patches({"train": raw_hh}, config)
        with (
            patches[0],
            patches[1],
            patch(
                "src.trainers.sft_trainer.AutoModelForCausalLM.from_pretrained",
                side_effect=_fake_model_loader,
            ),
            patch("src.trainers.sft_trainer.SFTTrainer", _DummyTrainer),
            redirect_stdout(buffer),
        ):
            run_sft_training(config)

        self.assertEqual(
            captured_kwargs.get("attn_implementation"),
            "flash_attention_2",
        )
        self.assertEqual(captured_kwargs.get("torch_dtype"), torch.bfloat16)
        self.assertIn("attn_impl=flash_attention_2", buffer.getvalue())

    def test_flash_attention_2_requires_mixed_precision(self):
        config = _base_sft_config()
        config["sft_training"]["attn_implementation"] = "flash_attention_2"
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        with (
            patch(
                "src.trainers.sft_trainer.load_dataset",
                side_effect=lambda *args, **kwargs: raw_hh,
            ),
            patch(
                "src.data.sft_dataset.AutoTokenizer.from_pretrained",
                side_effect=lambda *args, **kwargs: _DummyTokenizer(),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "flash_attention_2 requires precision fp16 or bf16",
            ):
                run_sft_training(config)

    def test_fsdp_save_only_model_is_auto_overridden(self):
        config = _base_sft_config()
        config["sft_training"]["fsdp"]["enabled"] = True
        config["sft_training"]["save_only_model"] = True

        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Name a planet.\n\nAssistant: Mars."},
                {"chosen": "\n\nHuman: 2+2?\n\nAssistant: 4."},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )
        trainer, output = self._run_pipeline(config, {"train": raw_hh})

        self.assertIn("Overriding save_only_model=False for FSDP", output)
        self.assertFalse(trainer.args.save_only_model)
        self.assertTrue(bool(trainer.args.fsdp))
        self.assertEqual(trainer.args.fsdp_config.get("version"), 1)

    def test_optional_sft_args_forwarding_and_defaults(self):
        config = _base_sft_config()
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )
        captured_kwargs = []

        def _capture_sft_config(**kwargs):
            captured_kwargs.append(kwargs)
            return SimpleNamespace(**kwargs)

        patches = self._patches({"train": raw_hh}, config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("src.trainers.sft_trainer.SFTConfig", side_effect=_capture_sft_config),
            patch("src.trainers.sft_trainer.SFTTrainer", _DummyTrainer),
        ):
            run_sft_training(config)

        first_kwargs = captured_kwargs[-1]
        for key in (
            "dataset_num_proc",
            "packing_strategy",
            "padding_free",
            "eval_packing",
            "max_steps",
        ):
            self.assertNotIn(key, first_kwargs)

        config["sft_training"]["dataset_num_proc"] = 4
        config["sft_training"]["packing_strategy"] = "wrapped"
        config["sft_training"]["padding_free"] = False
        config["sft_training"]["eval_packing"] = False
        config["sft_training"]["max_steps"] = 3

        patches = self._patches({"train": raw_hh}, config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("src.trainers.sft_trainer.SFTConfig", side_effect=_capture_sft_config),
            patch("src.trainers.sft_trainer.SFTTrainer", _DummyTrainer),
        ):
            run_sft_training(config)

        second_kwargs = captured_kwargs[-1]
        self.assertEqual(second_kwargs["dataset_num_proc"], 4)
        self.assertEqual(second_kwargs["packing_strategy"], "wrapped")
        self.assertFalse(second_kwargs["padding_free"])
        self.assertFalse(second_kwargs["eval_packing"])
        self.assertEqual(second_kwargs["max_steps"], 3)

    def test_invalid_gradient_accumulation_raises(self):
        config = _base_sft_config()
        config["sft_training"]["gradient_accumulation"] = 0
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        with (
            patch(
                "src.trainers.sft_trainer.load_dataset",
                side_effect=lambda *args, **kwargs: raw_hh,
            ),
            patch(
                "src.data.sft_dataset.AutoTokenizer.from_pretrained",
                side_effect=lambda *args, **kwargs: _DummyTokenizer(),
            ),
        ):
            with self.assertRaises(ValueError):
                run_sft_training(config)

    def test_invalid_fsdp_autowrap_raises(self):
        config = _base_sft_config()
        config["sft_training"]["fsdp"]["enabled"] = True
        config["sft_training"]["fsdp"]["auto_wrap_policy"] = "bad_policy"
        raw_hh = Dataset.from_list(
            [
                {"chosen": "\n\nHuman: Hi\n\nAssistant: Hello!"},
                {"chosen": "\n\nHuman: Bye\n\nAssistant: See you."},
            ]
        )

        with (
            patch(
                "src.trainers.sft_trainer.load_dataset",
                side_effect=lambda *args, **kwargs: raw_hh,
            ),
            patch(
                "src.data.sft_dataset.AutoTokenizer.from_pretrained",
                side_effect=lambda *args, **kwargs: _DummyTokenizer(),
            ),
        ):
            with self.assertRaises(ValueError):
                run_sft_training(config)


if __name__ == "__main__":
    unittest.main()

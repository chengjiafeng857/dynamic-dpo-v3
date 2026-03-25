"""Unit tests for margin-DPO trainer sample logging."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.trainers.margin_dpo_trainer import MarginDPOTrainer


class _DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_processes = 1
        self.process_index = 0

    def gather_for_metrics(self, tensor):
        tensor = torch.as_tensor(tensor)
        if tensor.ndim == 0:
            return tensor.unsqueeze(0)
        return tensor


class _DummyOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _DummyModel:
    def __init__(self, logits: torch.Tensor, *, training: bool = True):
        self._logits = logits
        self.training = training

    def __call__(self, *args, **kwargs):
        return _DummyOutput(self._logits)


def _modern_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                [1, 2, 5, 6, 0],
                [3, 4, 7, 8, 9],
                [1, 2, 6, 7, 8],
                [3, 4, 8, 9, 0],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
        "completion_mask": torch.tensor(
            [
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
    }


def _build_trainer(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    *,
    margin_log_path: str,
    sample_log_path: str,
    log_samples: int,
) -> tuple[MarginDPOTrainer, _DummyModel]:
    trainer = object.__new__(MarginDPOTrainer)
    trainer.accelerator = _DummyAccelerator()
    trainer.model = _DummyModel(policy_logits, training=True)
    trainer.ref_model = _DummyModel(ref_logits, training=True)
    trainer.precompute_ref_logps = False
    trainer._truncate_inputs = lambda input_ids, attention_mask, completion_mask: (
        input_ids,
        attention_mask,
        completion_mask,
    )
    trainer.is_world_process_zero = lambda: True
    trainer.state = SimpleNamespace(global_step=3)
    trainer.margin_log_path = margin_log_path
    trainer.sample_log_path = sample_log_path
    trainer.log_samples = log_samples
    trainer._logged_sample_count = 0
    trainer._printed_console_sample = False
    trainer._sample_log_jsonl_path = str(
        Path(sample_log_path) / "margin_sample_ids.jsonl"
    )
    trainer.f_divergence_type = "reverse_kl"
    trainer.f_alpha_divergence_coef = 1.0
    trainer.beta = 0.1
    return trainer, trainer.model


class MarginDPOTrainerLoggingTest(unittest.TestCase):
    def test_logging_writes_configured_sample_count_and_preserves_margin_logs(self):
        inputs = _modern_inputs()
        policy_logits = torch.randn(4, 5, 13)
        ref_logits = torch.randn(4, 5, 13)

        with TemporaryDirectory() as margin_tmp_dir, TemporaryDirectory() as sample_tmp_dir:
            trainer, policy_model = _build_trainer(
                policy_logits,
                ref_logits,
                margin_log_path=margin_tmp_dir,
                sample_log_path=sample_tmp_dir,
                log_samples=2,
            )
            stdout = io.StringIO()

            with (
                redirect_stdout(stdout),
                patch(
                    "src.trainers.margin_dpo_trainer.torch.randint",
                    return_value=torch.tensor([1]),
                ),
            ):
                trainer._compute_loss(policy_model, inputs, return_outputs=False)
                trainer._compute_loss(policy_model, inputs, return_outputs=False)

            sample_log_path = Path(sample_tmp_dir) / "margin_sample_ids.jsonl"
            self.assertTrue(sample_log_path.exists())
            records = [
                json.loads(line)
                for line in sample_log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["global_step"], 3)
            self.assertEqual(records[0]["batch_index"], 0)
            self.assertEqual(records[0]["input_ids"], [1, 2, 5, 6, 0])
            self.assertEqual(records[0]["labels"], [2, 5, 6, 0])
            self.assertEqual(records[0]["completion_mask"], [0, 1, 1, 0])
            self.assertEqual(records[0]["loss_labels"], [-100, 5, 6, -100])
            self.assertEqual(records[1]["batch_index"], 1)
            self.assertEqual(
                stdout.getvalue().count("[MARGIN_DPO] random_sample_input_ids="), 1
            )
            self.assertIn("[MARGIN_DPO] random_sample_labels=", stdout.getvalue())
            self.assertIn("[MARGIN_DPO] random_sample_loss_labels=", stdout.getvalue())

            self.assertTrue((Path(margin_tmp_dir) / "margins.jsonl").exists())
            self.assertTrue((Path(margin_tmp_dir) / "step_00003.npy").exists())

    def test_log_samples_zero_is_no_op_and_loss_is_unchanged(self):
        inputs = _modern_inputs()
        policy_logits = torch.randn(4, 5, 11)
        ref_logits = torch.randn(4, 5, 11)

        with (
            TemporaryDirectory() as logging_margin_tmp_dir,
            TemporaryDirectory() as logging_sample_tmp_dir,
            TemporaryDirectory() as no_log_margin_tmp_dir,
            TemporaryDirectory() as no_log_sample_tmp_dir,
        ):
            logging_trainer, logging_model = _build_trainer(
                policy_logits,
                ref_logits,
                margin_log_path=logging_margin_tmp_dir,
                sample_log_path=logging_sample_tmp_dir,
                log_samples=2,
            )
            no_log_trainer, no_log_model = _build_trainer(
                policy_logits,
                ref_logits,
                margin_log_path=no_log_margin_tmp_dir,
                sample_log_path=no_log_sample_tmp_dir,
                log_samples=0,
            )

            logging_stdout = io.StringIO()
            with (
                redirect_stdout(logging_stdout),
                patch(
                    "src.trainers.margin_dpo_trainer.torch.randint",
                    return_value=torch.tensor([0]),
                ),
            ):
                logging_loss = logging_trainer._compute_loss(
                    logging_model, inputs, return_outputs=False
                )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                no_log_loss = no_log_trainer._compute_loss(
                    no_log_model, inputs, return_outputs=False
                )

            self.assertAlmostEqual(logging_loss.item(), no_log_loss.item(), places=7)
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(
                (Path(no_log_sample_tmp_dir) / "margin_sample_ids.jsonl").exists()
            )
            self.assertTrue((Path(no_log_margin_tmp_dir) / "margins.jsonl").exists())
            self.assertTrue((Path(no_log_margin_tmp_dir) / "step_00003.npy").exists())


if __name__ == "__main__":
    unittest.main()

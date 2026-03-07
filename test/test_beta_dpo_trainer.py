"""Unit tests for BetaDPOTrainer-specific behavior."""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.trainers.beta_dpo_trainer import BetaDPOTrainer


class _DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_processes = 2
        self.process_index = 1
        self.is_main_process = False
        self.gather_calls = 0
        self.gather_for_metrics_calls = 0

    def gather(self, tensor):
        self.gather_calls += 1
        return torch.cat((tensor, tensor), dim=0)

    def gather_for_metrics(self, tensor):
        self.gather_for_metrics_calls += 1
        return tensor


class _DummyModel:
    def __init__(self, vocab_size):
        self.training = True
        self.vocab_size = vocab_size

    def __call__(self, input_ids, attention_mask, use_cache=False):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float32)
        return SimpleNamespace(logits=logits)


class BetaDPOTrainerTest(unittest.TestCase):
    def test_compute_loss_uses_full_gather_for_training_mask(self):
        trainer = BetaDPOTrainer.__new__(BetaDPOTrainer)
        trainer.accelerator = _DummyAccelerator()
        trainer.model = SimpleNamespace(training=True)
        trainer.ref_model = _DummyModel(vocab_size=8)
        trainer.precompute_ref_logps = False
        trainer.beta = 0.1
        trainer.args = SimpleNamespace(alpha=1.0, beta_min=1e-3, rho=0.8)
        trainer.r_gap_mean = torch.zeros(())
        trainer.r_gap_std = torch.ones(())
        trainer._metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }
        trainer._truncate_inputs = lambda input_ids, attention_mask, completion_mask: (
            input_ids,
            attention_mask,
            completion_mask,
        )
        trainer.ema_update_gap_mean_and_std = lambda _: None
        trainer.sample_data_mask = lambda global_gap: torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        trainer.compute_beta_used = lambda r_gap_global, global_mask, beta: torch.tensor(
            0.1, dtype=torch.float32
        )

        inputs = {
            "input_ids": torch.tensor(
                [
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(8, 2, dtype=torch.long),
            "completion_mask": torch.ones(8, 2, dtype=torch.long),
        }

        with patch.object(trainer, "_is_distributed", return_value=True):
            loss = trainer._compute_loss(_DummyModel(vocab_size=8), inputs, return_outputs=False)

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(trainer.accelerator.gather_calls, 1)
        self.assertEqual(trainer.accelerator.gather_for_metrics_calls, 0)


if __name__ == "__main__":
    unittest.main()

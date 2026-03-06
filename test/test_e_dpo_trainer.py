"""Focused unit tests for epsilon-DPO math and beta state updates."""

from collections import defaultdict
import unittest

import torch

from src.trainers.e_dpo_trainer import EpsilonDPOTrainer


class _DummyGradientState:
    def __init__(self, sync_gradients: bool = True):
        self.sync_gradients = sync_gradients


class _DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.gradient_state = _DummyGradientState(sync_gradients=True)

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.unsqueeze(0)
        return tensor


class _DummyOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _DummyModel:
    def __init__(self, logits: torch.Tensor, training: bool):
        self._logits = logits
        self.training = training

    def __call__(self, **kwargs):
        return _DummyOutput(self._logits)


def _build_trainer_stub() -> EpsilonDPOTrainer:
    trainer = object.__new__(EpsilonDPOTrainer)
    trainer.epsilon = 0.2
    trainer.beta = 0.1
    trainer.label_smoothing = 0.0
    trainer.precompute_ref_logps = False
    trainer.accelerator = _DummyAccelerator()
    trainer._truncate_inputs = lambda input_ids, attention_mask, completion_mask: (
        input_ids,
        attention_mask,
        completion_mask,
    )
    trainer._metrics = {
        "train": defaultdict(list),
        "eval": defaultdict(list),
    }
    trainer._beta_tilde_accum_sum = torch.zeros((), dtype=torch.float32)
    trainer._beta_tilde_accum_count = torch.zeros((), dtype=torch.float32)
    return trainer


class EpsilonDPOTrainerTest(unittest.TestCase):
    def test_compute_beta_tilde_selects_paper_branches(self):
        trainer = _build_trainer_stub()

        margin = torch.tensor([1.0, 1.0, 3.0])
        margin_plus = torch.tensor([2.0, 0.0, 2.0])
        margin_minus = torch.tensor([0.0, 2.0, 1.0])

        beta_tilde, use_beta_minus, use_beta_plus = trainer._compute_beta_tilde(
            margin, margin_plus, margin_minus
        )

        expected_beta_minus = trainer.beta / (1.0 + trainer.epsilon)
        expected_beta_plus = trainer.beta / (1.0 - trainer.epsilon)

        self.assertEqual(use_beta_minus.tolist(), [True, False, False])
        self.assertEqual(use_beta_plus.tolist(), [False, True, False])
        self.assertTrue(torch.allclose(beta_tilde[0], torch.tensor(expected_beta_minus)))
        self.assertTrue(torch.allclose(beta_tilde[1], torch.tensor(expected_beta_plus)))
        self.assertTrue(torch.allclose(beta_tilde[2], torch.tensor(trainer.beta)))

    def test_apply_beta_update_uses_exact_mean_beta_tilde(self):
        trainer = _build_trainer_stub()

        trainer._accumulate_beta_tilde(torch.tensor([0.05, 0.10], dtype=torch.float32))
        trainer._accumulate_beta_tilde(torch.tensor([0.20], dtype=torch.float32))

        trainer._apply_beta_update()

        expected = (0.05 + 0.10 + 0.20) / 3.0
        self.assertAlmostEqual(trainer.beta, expected, places=7)
        self.assertEqual(float(trainer._beta_tilde_accum_sum), 0.0)
        self.assertEqual(float(trainer._beta_tilde_accum_count), 0.0)

    def test_compute_loss_does_not_mutate_beta_during_eval(self):
        trainer = _build_trainer_stub()

        torch.manual_seed(0)
        policy_logits = torch.randn(2, 4, 8)
        ref_logits = torch.randn(2, 4, 8)

        policy_model = _DummyModel(policy_logits, training=False)
        ref_model = _DummyModel(ref_logits, training=False)

        trainer.model = policy_model
        trainer.ref_model = ref_model

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 0], [1, 3, 2, 0]], dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "completion_mask": torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.long),
        }

        beta_before = trainer.beta
        loss = trainer._compute_loss(policy_model, inputs, return_outputs=False)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(trainer.beta, beta_before)


if __name__ == "__main__":
    unittest.main()

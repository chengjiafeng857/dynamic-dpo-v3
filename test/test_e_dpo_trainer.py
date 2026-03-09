"""Regression tests for epsilon-DPO trainer streaming epsilon-step estimation."""

from collections import defaultdict
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from trl.trainer.utils import selective_log_softmax

from src.trainers.epsilon_dpo_trainer import EpsilonDPOTrainer


class _DummyGradientState:
    def __init__(self, sync_gradients: bool = True):
        self.sync_gradients = sync_gradients


class _DummyAccelerator:
    def __init__(self, sync_gradients: bool = True):
        self.device = torch.device("cpu")
        self.gradient_state = _DummyGradientState(sync_gradients=sync_gradients)

    def gather_for_metrics(self, tensor):
        tensor = torch.as_tensor(tensor)
        return tensor.unsqueeze(0) if tensor.ndim == 0 else tensor

    def gather(self, tensor):
        tensor = torch.as_tensor(tensor)
        return tensor.unsqueeze(0) if tensor.ndim == 0 else tensor

    @staticmethod
    def unwrap_model(model):
        return model


class _DummyOutput:
    def __init__(self, logits: torch.Tensor, aux_loss: torch.Tensor | None = None):
        self.logits = logits
        self.aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0)


class _DummyModel(nn.Module):
    def __init__(
        self,
        logits: torch.Tensor,
        *,
        aux_loss: torch.Tensor | None = None,
        training: bool = True,
    ):
        super().__init__()
        self._logits = logits
        self._aux_loss = aux_loss
        self.is_gradient_checkpointing = False
        self.train(training)

    def forward(self, *args, **kwargs):
        return _DummyOutput(self._logits, self._aux_loss)

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, _kwargs=None):
        self.is_gradient_checkpointing = True


def _build_args(rpo_alpha: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        gradient_accumulation_steps=2,
        gradient_checkpointing_kwargs=None,
        rpo_alpha=rpo_alpha,
    )


def _build_trainer(
    trainer_cls=EpsilonDPOTrainer,
    *,
    beta: float = 0.1,
    epsilon: float = 0.2,
    label_smoothing: float = 0.0,
    use_weighting: bool = False,
    rpo_alpha: float | None = None,
    aux_loss_enabled: bool = False,
    aux_loss_coef: float = 0.0,
    precompute_ref_logps: bool = False,
    ld_alpha: float | None = None,
    sync_gradients: bool = True,
):
    trainer = object.__new__(trainer_cls)
    trainer.epsilon = epsilon
    trainer.beta = beta
    trainer.label_smoothing = label_smoothing
    trainer.use_weighting = use_weighting
    trainer.aux_loss_enabled = aux_loss_enabled
    trainer.aux_loss_coef = aux_loss_coef
    trainer.precompute_ref_logps = precompute_ref_logps
    trainer.loss_types = ["sigmoid"]
    trainer.loss_weights = [1.0]
    trainer.ld_alpha = ld_alpha
    trainer.args = _build_args(rpo_alpha=rpo_alpha)
    trainer.accelerator = _DummyAccelerator(sync_gradients=sync_gradients)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._truncate_inputs = lambda input_ids, attention_mask, completion_mask: (
        input_ids,
        attention_mask,
        completion_mask,
    )
    trainer.steps = 0.0
    trainer._total_train_tokens = 0
    trainer.model = None
    trainer.ref_model = None
    return trainer


class _MaterializedBaselineEpsilonTrainer(EpsilonDPOTrainer):
    def _estimate_epsilon_steps(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_completion_mask: torch.Tensor,
        base_logps: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            p_epsilon_logits = ((1.0 + self.epsilon) * logits) - (self.epsilon * ref_logits)
            p_eps_per_token_logps = selective_log_softmax(p_epsilon_logits, shift_labels)
            p_eps_logps = self._sum_completion_logps(p_eps_per_token_logps, shift_completion_mask)

            n_epsilon_logits = ((1.0 - self.epsilon) * logits) + (self.epsilon * ref_logits)
            n_eps_per_token_logps = selective_log_softmax(n_epsilon_logits, shift_labels)
            n_eps_logps = self._sum_completion_logps(n_eps_per_token_logps, shift_completion_mask)

            base_chosen, base_rejected = base_logps.chunk(2, dim=0)
            p_chosen, p_rejected = p_eps_logps.chunk(2, dim=0)
            n_chosen, n_rejected = n_eps_logps.chunk(2, dim=0)

            base_logratios = base_chosen - base_rejected
            p_logratios = p_chosen - p_rejected
            n_logratios = n_chosen - n_rejected

            p_steps = (p_logratios > base_logratios) & (base_logratios > n_logratios)
            n_steps = (n_logratios > base_logratios) & (base_logratios > p_logratios)
            steps = p_steps.to(torch.int64) - n_steps.to(torch.int64)

        return steps.to(logits.dtype)


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


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


def _latest_metrics(trainer: EpsilonDPOTrainer) -> dict[str, float]:
    metrics = {}
    for key, values in trainer._metrics["train"].items():
        if values:
            metrics[key] = values[-1]
    return metrics


def _assert_metric_dict_close(
    testcase: unittest.TestCase,
    left: dict[str, float],
    right: dict[str, float],
) -> None:
    testcase.assertEqual(set(left), set(right))
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            left_value = left_value.item()
        if isinstance(right_value, torch.Tensor):
            right_value = right_value.item()
        testcase.assertAlmostEqual(float(left_value), float(right_value), places=7)


def _random_step_inputs(
    trainer: EpsilonDPOTrainer,
    *,
    batch_pairs: int = 3,
    seq_len: int = 5,
    vocab_size: int = 11,
) -> dict[str, torch.Tensor]:
    logits = torch.randn(2 * batch_pairs, seq_len, vocab_size)
    ref_logits = torch.randn(2 * batch_pairs, seq_len, vocab_size)
    shift_labels = torch.randint(0, vocab_size, (2 * batch_pairs, seq_len))

    shift_completion_mask = torch.zeros(2 * batch_pairs, seq_len, dtype=torch.long)
    for i in range(2 * batch_pairs):
        completion_len = int(torch.randint(1, seq_len + 1, (1,)).item())
        shift_completion_mask[i, seq_len - completion_len :] = 1

    per_token_logps = selective_log_softmax(logits, shift_labels)
    base_logps = trainer._sum_completion_logps(per_token_logps, shift_completion_mask)
    return {
        "logits": logits,
        "ref_logits": ref_logits,
        "shift_labels": shift_labels,
        "shift_completion_mask": shift_completion_mask,
        "base_logps": base_logps,
    }


class EpsilonDPOTrainerTest(unittest.TestCase):
    def test_estimate_epsilon_steps_streamed_matches_materialized_without_ld_alpha(self):
        torch.manual_seed(0)
        streamed_trainer = _build_trainer(ld_alpha=None)
        baseline_trainer = _build_trainer(trainer_cls=_MaterializedBaselineEpsilonTrainer, ld_alpha=None)
        step_inputs = _random_step_inputs(streamed_trainer, batch_pairs=4, seq_len=6, vocab_size=13)

        streamed_steps = streamed_trainer._estimate_epsilon_steps(**step_inputs)
        baseline_steps = baseline_trainer._estimate_epsilon_steps(**step_inputs)

        self.assertEqual(streamed_steps.dtype, step_inputs["logits"].dtype)
        self.assertEqual(streamed_steps.shape, (4,))
        self.assertTrue(torch.equal(streamed_steps, baseline_steps))

    def test_estimate_epsilon_steps_streamed_matches_materialized_with_ld_alpha(self):
        torch.manual_seed(1)
        streamed_trainer = _build_trainer(ld_alpha=0.3)
        baseline_trainer = _build_trainer(
            trainer_cls=_MaterializedBaselineEpsilonTrainer,
            ld_alpha=0.3,
        )
        step_inputs = _random_step_inputs(streamed_trainer, batch_pairs=5, seq_len=7, vocab_size=9)

        streamed_steps = streamed_trainer._estimate_epsilon_steps(**step_inputs)
        baseline_steps = baseline_trainer._estimate_epsilon_steps(**step_inputs)

        self.assertTrue(torch.equal(streamed_steps, baseline_steps))

    def test_compute_loss_precompute_ref_logps_keeps_zero_steps(self):
        torch.manual_seed(2)
        trainer = _build_trainer(precompute_ref_logps=True, sync_gradients=True)
        policy_logits = torch.randn(4, 5, 13)
        policy_model = _DummyModel(policy_logits, training=True)
        trainer.model = policy_model
        trainer.ref_model = None

        inputs = _modern_inputs()
        inputs["ref_chosen_logps"] = torch.randn(2)
        inputs["ref_rejected_logps"] = torch.randn(2)
        initial_beta = trainer.beta

        loss = trainer._compute_loss(policy_model, _clone_inputs(inputs), return_outputs=False)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertAlmostEqual(trainer._metrics["train"]["kl/p_epsilon_steps"][-1], 0.0, places=7)
        self.assertAlmostEqual(trainer._metrics["train"]["kl/n_epsilon_steps"][-1], 0.0, places=7)
        self.assertAlmostEqual(trainer._metrics["train"]["kl/avg_steps"][-1], 0.0, places=7)
        self.assertAlmostEqual(trainer.beta, initial_beta, places=7)
        self.assertAlmostEqual(trainer.steps, 0.0, places=7)

    def _assert_compute_loss_matches_materialized_baseline(
        self,
        *,
        use_weighting: bool = False,
        rpo_alpha: float | None = None,
        aux_loss_enabled: bool = False,
        aux_loss_coef: float = 0.0,
    ):
        torch.manual_seed(11)
        policy_logits = torch.randn(4, 5, 17)
        ref_logits = torch.randn(4, 5, 17)
        aux_loss = torch.tensor(0.7) if aux_loss_enabled else None

        streamed_trainer = _build_trainer(
            use_weighting=use_weighting,
            rpo_alpha=rpo_alpha,
            aux_loss_enabled=aux_loss_enabled,
            aux_loss_coef=aux_loss_coef,
            sync_gradients=True,
        )
        baseline_trainer = _build_trainer(
            trainer_cls=_MaterializedBaselineEpsilonTrainer,
            use_weighting=use_weighting,
            rpo_alpha=rpo_alpha,
            aux_loss_enabled=aux_loss_enabled,
            aux_loss_coef=aux_loss_coef,
            sync_gradients=True,
        )

        streamed_policy_model = _DummyModel(policy_logits, aux_loss=aux_loss, training=True)
        streamed_ref_model = _DummyModel(ref_logits, training=True)
        baseline_policy_model = _DummyModel(policy_logits, aux_loss=aux_loss, training=True)
        baseline_ref_model = _DummyModel(ref_logits, training=True)

        streamed_trainer.model = streamed_policy_model
        streamed_trainer.ref_model = streamed_ref_model
        baseline_trainer.model = baseline_policy_model
        baseline_trainer.ref_model = baseline_ref_model

        inputs = _modern_inputs()
        streamed_loss = streamed_trainer._compute_loss(
            streamed_policy_model,
            _clone_inputs(inputs),
            return_outputs=False,
        )
        baseline_loss = baseline_trainer._compute_loss(
            baseline_policy_model,
            _clone_inputs(inputs),
            return_outputs=False,
        )

        self.assertAlmostEqual(streamed_loss.item(), baseline_loss.item(), places=7)
        self.assertAlmostEqual(streamed_trainer.beta, baseline_trainer.beta, places=7)
        self.assertAlmostEqual(streamed_trainer.steps, baseline_trainer.steps, places=7)
        _assert_metric_dict_close(
            self,
            _latest_metrics(streamed_trainer),
            _latest_metrics(baseline_trainer),
        )

    def test_compute_loss_streamed_matches_materialized_baseline(self):
        self._assert_compute_loss_matches_materialized_baseline()

    def test_compute_loss_streamed_matches_materialized_with_weighting_rpo_aux(self):
        self._assert_compute_loss_matches_materialized_baseline(
            use_weighting=True,
            rpo_alpha=0.3,
            aux_loss_enabled=True,
            aux_loss_coef=0.5,
        )


if __name__ == "__main__":
    unittest.main()

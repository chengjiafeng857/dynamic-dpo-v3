"""Reference-parity tests for the epsilon-DPO trainer."""

from collections import defaultdict
from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest

import torch

from reference.archive.e_dpo_trainer import EpsilonDPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def _load_reference_trainer_class():
    config_module = ModuleType("config")
    config_module.EpsilonDPOConfig = object
    previous = sys.modules.get("config")
    sys.modules["config"] = config_module
    try:
        spec = spec_from_file_location(
            "reference_trainer_module", ROOT / "reference" / "trainer.py"
        )
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous
    return module.EpsilonDPOTrainer


ReferenceEpsilonDPOTrainer = _load_reference_trainer_class()


class _DummyGradientState:
    def __init__(self, sync_gradients: bool = True):
        self.sync_gradients = sync_gradients


class _DummyAccelerator:
    def __init__(self, sync_gradients: bool = True):
        self.device = torch.device("cpu")
        self.gradient_state = _DummyGradientState(sync_gradients=sync_gradients)

    def gather_for_metrics(self, tensor):
        tensor = torch.as_tensor(tensor)
        if tensor.ndim == 0:
            return tensor.unsqueeze(0)
        return tensor

    def gather(self, tensor):
        tensor = torch.as_tensor(tensor)
        if tensor.ndim == 0:
            return tensor.unsqueeze(0)
        return tensor


class _DummyOutput:
    def __init__(self, logits: torch.Tensor, aux_loss: torch.Tensor | None = None):
        self.logits = logits
        self.aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0)


class _DummyModel:
    def __init__(
        self,
        logits: torch.Tensor,
        *,
        aux_loss: torch.Tensor | None = None,
        training: bool = True,
    ):
        self._logits = logits
        self._aux_loss = aux_loss
        self.training = training

    def __call__(self, *args, **kwargs):
        return _DummyOutput(self._logits, self._aux_loss)


def _legacy_batch() -> dict[str, torch.Tensor]:
    return {
        "prompt_input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "prompt_attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "chosen_completion_input_ids": torch.tensor(
            [[5, 6, 0], [7, 8, 9]], dtype=torch.long
        ),
        "chosen_completion_attention_mask": torch.tensor(
            [[1, 1, 0], [1, 1, 1]], dtype=torch.long
        ),
        "rejected_completion_input_ids": torch.tensor(
            [[6, 7, 8], [8, 9, 0]], dtype=torch.long
        ),
        "rejected_completion_attention_mask": torch.tensor(
            [[1, 1, 1], [1, 1, 0]], dtype=torch.long
        ),
    }


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


def _concatenated_inputs(
    batch: dict[str, torch.Tensor], padding_value: int
) -> dict[str, torch.Tensor]:
    del padding_value
    return {
        "prompt_input_ids": torch.cat(
            [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
        ),
        "prompt_attention_mask": torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        ),
        "completion_input_ids": torch.cat(
            [
                batch["chosen_completion_input_ids"],
                batch["rejected_completion_input_ids"],
            ],
            dim=0,
        ),
        "completion_attention_mask": torch.cat(
            [
                batch["chosen_completion_attention_mask"],
                batch["rejected_completion_attention_mask"],
            ],
            dim=0,
        ),
    }


def _build_args(rpo_alpha: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        max_length=None,
        rpo_alpha=rpo_alpha,
        gradient_accumulation_steps=2,
    )


def _build_reference_style_trainer(
    trainer_cls,
    *,
    beta: float = 0.1,
    epsilon: float = 0.2,
    label_smoothing: float = 0.0,
    use_weighting: bool = False,
    rpo_alpha: float | None = None,
    aux_loss_enabled: bool = False,
    aux_loss_coef: float = 0.0,
) -> EpsilonDPOTrainer:
    trainer = object.__new__(trainer_cls)
    trainer.epsilon = epsilon
    trainer.beta = beta
    trainer.label_smoothing = label_smoothing
    trainer.padding_value = 0
    trainer.aux_loss_enabled = aux_loss_enabled
    trainer.aux_loss_coef = aux_loss_coef
    trainer.use_weighting = use_weighting
    trainer.use_num_logits_to_keep = False
    trainer.is_encoder_decoder = False
    trainer.label_pad_token_id = -100
    trainer.args = _build_args(rpo_alpha=rpo_alpha)
    trainer.accelerator = _DummyAccelerator(sync_gradients=True)
    trainer.steps = 0.0
    trainer.null_ref_context = nullcontext
    trainer._peft_has_been_casted_to_bf16 = False
    trainer.concatenated_inputs = _concatenated_inputs
    trainer.precompute_ref_logps = False
    trainer.precompute_ref_log_probs = False
    trainer.loss_types = ["sigmoid"]
    trainer.loss_weights = [1.0]
    trainer.ld_alpha = None
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._truncate_inputs = lambda input_ids, attention_mask, completion_mask: (
        input_ids,
        attention_mask,
        completion_mask,
    )
    return trainer


def _build_trainers(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    *,
    use_weighting: bool = False,
    rpo_alpha: float | None = None,
    aux_loss_enabled: bool = False,
    aux_loss_coef: float = 0.0,
    sync_gradients: bool = True,
):
    src_trainer = _build_reference_style_trainer(
        EpsilonDPOTrainer,
        use_weighting=use_weighting,
        rpo_alpha=rpo_alpha,
        aux_loss_enabled=aux_loss_enabled,
        aux_loss_coef=aux_loss_coef,
    )
    ref_trainer = _build_reference_style_trainer(
        ReferenceEpsilonDPOTrainer,
        use_weighting=use_weighting,
        rpo_alpha=rpo_alpha,
        aux_loss_enabled=aux_loss_enabled,
        aux_loss_coef=aux_loss_coef,
    )
    src_trainer.accelerator = _DummyAccelerator(sync_gradients=sync_gradients)
    ref_trainer.accelerator = _DummyAccelerator(sync_gradients=sync_gradients)

    aux_loss = torch.tensor(0.7) if aux_loss_enabled else None
    policy_model = _DummyModel(policy_logits, aux_loss=aux_loss, training=True)
    ref_model = _DummyModel(ref_logits, training=True)

    src_trainer.model = policy_model
    src_trainer.ref_model = ref_model
    ref_trainer.model = policy_model
    ref_trainer.ref_model = ref_model
    return src_trainer, ref_trainer, policy_model


def _assert_tensor_close(testcase, left: torch.Tensor, right: torch.Tensor):
    testcase.assertTrue(torch.allclose(left, right, atol=1e-7, rtol=0.0))


def _assert_metric_dict_close(
    testcase, left: dict[str, float], right: dict[str, float]
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


class EpsilonDPOTrainerTest(unittest.TestCase):
    def test_concatenated_forward_matches_official_reference(self):
        torch.manual_seed(0)
        policy_logits = torch.randn(4, 5, 17)

        src_trainer, ref_trainer, policy_model = _build_trainers(
            policy_logits,
            torch.randn(4, 5, 17),
        )
        batch = _legacy_batch()
        ref_logits = torch.randn(4, 5, 17)

        src_output = src_trainer.concatenated_forward(policy_model, batch, ref_logits)
        ref_output = ref_trainer.concatenated_forward(policy_model, batch, ref_logits)

        self.assertEqual(set(src_output), set(ref_output))
        for key in src_output:
            _assert_tensor_close(self, src_output[key], ref_output[key])

    def test_compute_loss_matches_official_metrics_and_beta_update(self):
        torch.manual_seed(1)
        policy_logits = torch.randn(4, 5, 13)
        ref_logits = torch.randn(4, 5, 13)

        src_trainer, ref_trainer, policy_model = _build_trainers(
            policy_logits,
            ref_logits,
            sync_gradients=True,
        )
        modern_inputs = _modern_inputs()
        legacy_batch = _legacy_batch()

        src_loss = src_trainer._compute_loss(
            policy_model, modern_inputs, return_outputs=False
        )
        ref_loss, ref_metrics = ref_trainer.get_batch_loss_metrics(
            policy_model, legacy_batch, train_eval="train"
        )

        self.assertAlmostEqual(src_loss.item(), ref_loss.item(), places=7)
        src_metrics = {
            key: values[-1] for key, values in src_trainer._metrics["train"].items()
        }
        _assert_metric_dict_close(self, src_metrics, ref_metrics)
        self.assertAlmostEqual(src_trainer.beta, ref_trainer.beta, places=7)
        self.assertAlmostEqual(src_trainer.steps, ref_trainer.steps, places=7)

    def test_optional_branches_match_official_reference(self):
        torch.manual_seed(2)
        policy_logits = torch.randn(4, 5, 11)
        ref_logits = torch.randn(4, 5, 11)

        src_trainer, ref_trainer, policy_model = _build_trainers(
            policy_logits,
            ref_logits,
            use_weighting=True,
            rpo_alpha=0.3,
            aux_loss_enabled=True,
            aux_loss_coef=0.5,
        )
        modern_inputs = _modern_inputs()
        legacy_batch = _legacy_batch()

        src_loss = src_trainer._compute_loss(
            policy_model, modern_inputs, return_outputs=False
        )
        ref_loss, ref_metrics = ref_trainer.get_batch_loss_metrics(
            policy_model, legacy_batch, train_eval="train"
        )

        self.assertAlmostEqual(src_loss.item(), ref_loss.item(), places=7)
        src_metrics = {
            key: values[-1] for key, values in src_trainer._metrics["train"].items()
        }
        _assert_metric_dict_close(self, src_metrics, ref_metrics)
        self.assertIn("nll_loss", src_metrics)
        self.assertIn("aux_loss", src_metrics)


if __name__ == "__main__":
    unittest.main()

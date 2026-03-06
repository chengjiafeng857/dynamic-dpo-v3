import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import selective_log_softmax


@dataclass
class EpsilonDPOConfig(DPOConfig):
    """TRL-style config for epsilon-DPO."""

    beta: float = field(default=0.1)
    epsilon: float = field(default=0.01)

    def __post_init__(self):
        if self.beta <= 0.0:
            raise ValueError("beta must be > 0.")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be >= 0.")
        if self.epsilon >= 1.0:
            raise ValueError("epsilon must be < 1.")

        if bool(getattr(self, "reference_free", False)):
            raise ValueError("EpsilonDPOConfig requires reference_free=False.")
        if bool(getattr(self, "precompute_ref_log_probs", False)):
            raise ValueError(
                "EpsilonDPOConfig requires precompute_ref_log_probs=False."
            )
        raw_loss_type = getattr(self, "loss_type", "sigmoid")
        if isinstance(raw_loss_type, (list, tuple)):
            normalized_loss_types = [
                (
                    str(loss_type.value)
                    if hasattr(loss_type, "value")
                    else str(loss_type)
                ).lower()
                for loss_type in raw_loss_type
            ]
        else:
            normalized_loss_types = [
                (
                    str(raw_loss_type.value)
                    if hasattr(raw_loss_type, "value")
                    else str(raw_loss_type)
                ).lower()
            ]
        if len(normalized_loss_types) != 1 or not normalized_loss_types[0].endswith(
            "sigmoid"
        ):
            raise ValueError("EpsilonDPOConfig requires loss_type='sigmoid'.")

        super().__post_init__()

        # Keep the supported EDPO assumptions explicit.
        self.reference_free = False
        self.precompute_ref_log_probs = False
        self.loss_type = ["sigmoid"] if isinstance(raw_loss_type, (list, tuple)) else "sigmoid"


class EpsilonDPOTrainer(DPOTrainer):
    """TRL DPOTrainer variant implementing epsilon-DPO with stateful beta updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, EpsilonDPOConfig):
            raise TypeError("EpsilonDPOTrainer requires args=EpsilonDPOConfig")

        self.epsilon = float(self.args.epsilon)
        device = self.accelerator.device
        self._beta_tilde_accum_sum = torch.zeros((), device=device, dtype=torch.float32)
        self._beta_tilde_accum_count = torch.zeros((), device=device, dtype=torch.float32)

    def _compute_beta_tilde(
        self,
        margin: torch.Tensor,
        margin_plus: torch.Tensor,
        margin_minus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_beta = float(self.beta)
        beta_minus = base_beta / (1.0 + self.epsilon)
        beta_plus = base_beta / (1.0 - self.epsilon)

        use_beta_minus = (margin_plus > margin) & (margin > margin_minus)
        use_beta_plus = (margin_minus > margin) & (margin > margin_plus)

        beta_tilde = torch.full_like(margin, fill_value=base_beta)
        beta_tilde = torch.where(
            use_beta_minus,
            torch.full_like(beta_tilde, fill_value=beta_minus),
            beta_tilde,
        )
        beta_tilde = torch.where(
            use_beta_plus,
            torch.full_like(beta_tilde, fill_value=beta_plus),
            beta_tilde,
        )
        return beta_tilde, use_beta_minus, use_beta_plus

    def _accumulate_beta_tilde(self, beta_tilde: torch.Tensor) -> None:
        self._beta_tilde_accum_sum = self._beta_tilde_accum_sum + beta_tilde.detach().sum().to(torch.float32)
        self._beta_tilde_accum_count = self._beta_tilde_accum_count + torch.tensor(
            float(beta_tilde.numel()), device=beta_tilde.device, dtype=torch.float32
        )

    def _apply_beta_update(self) -> None:
        global_sum = self.accelerator.gather(self._beta_tilde_accum_sum.detach()).sum()
        global_count = self.accelerator.gather(self._beta_tilde_accum_count.detach()).sum()

        if float(global_count.item()) > 0.0:
            self.beta = float((global_sum / global_count).item())

        self._beta_tilde_accum_sum.zero_()
        self._beta_tilde_accum_count.zero_()

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids, attention_mask, completion_mask = self._truncate_inputs(
            input_ids, attention_mask, completion_mask
        )

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        for key in (
            "pixel_values",
            "pixel_attention_mask",
            "image_grid_thw",
            "image_sizes",
            "token_type_ids",
        ):
            if key in inputs:
                model_kwargs[key] = inputs[key]

        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous().bool()

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[~shift_completion_mask] = 0.0
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        if self.precompute_ref_logps:
            raise ValueError(
                "EpsilonDPOTrainer does not support precomputed reference log probabilities."
            )
        if self.ref_model is None:
            raise ValueError("EpsilonDPOTrainer requires a non-null ref_model.")

        with torch.no_grad():
            ref_outputs = self.ref_model(**model_kwargs)

        ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
        ref_per_token_logps[~shift_completion_mask] = 0.0
        ref_logps = ref_per_token_logps.sum(dim=1)
        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        policy_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        margin = policy_logratios - ref_logratios

        plus_shift_logits = ((1.0 + self.epsilon) * shift_logits) - (
            self.epsilon * ref_shift_logits
        )
        minus_shift_logits = ((1.0 - self.epsilon) * shift_logits) + (
            self.epsilon * ref_shift_logits
        )

        plus_per_token_logps = selective_log_softmax(plus_shift_logits, shift_labels)
        plus_per_token_logps[~shift_completion_mask] = 0.0
        plus_logps = plus_per_token_logps.sum(dim=1)
        plus_chosen_logps, plus_rejected_logps = plus_logps.chunk(2, dim=0)

        minus_per_token_logps = selective_log_softmax(minus_shift_logits, shift_labels)
        minus_per_token_logps[~shift_completion_mask] = 0.0
        minus_logps = minus_per_token_logps.sum(dim=1)
        minus_chosen_logps, minus_rejected_logps = minus_logps.chunk(2, dim=0)

        margin_plus = (plus_chosen_logps - plus_rejected_logps) - ref_logratios
        margin_minus = (minus_chosen_logps - minus_rejected_logps) - ref_logratios

        beta_tilde, use_beta_minus, use_beta_plus = self._compute_beta_tilde(
            margin, margin_plus, margin_minus
        )

        scaled_margin = beta_tilde * margin
        losses = -F.logsigmoid(scaled_margin) * (1.0 - self.label_smoothing) - F.logsigmoid(
            -scaled_margin
        ) * self.label_smoothing
        loss = losses.mean()

        chosen_rewards = beta_tilde * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = beta_tilde * (rejected_logps - ref_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        chosen_shift_completion_mask = shift_completion_mask[: chosen_logps.size(0)]
        rejected_shift_completion_mask = shift_completion_mask[chosen_logps.size(0) :]
        chosen_shift_logits = shift_logits[: chosen_logps.size(0)]
        rejected_shift_logits = shift_logits[chosen_logps.size(0) :]

        if chosen_shift_completion_mask.any():
            mean_chosen_logits = chosen_shift_logits[chosen_shift_completion_mask].mean()
        else:
            mean_chosen_logits = shift_logits.new_zeros(())

        if rejected_shift_completion_mask.any():
            mean_rejected_logits = rejected_shift_logits[
                rejected_shift_completion_mask
            ].mean()
        else:
            mean_rejected_logits = shift_logits.new_zeros(())

        unchanged_mask = ~(use_beta_minus | use_beta_plus)
        metrics = self._metrics[mode]
        metrics["rewards/chosen"].append(
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics["rewards/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics["rewards/accuracies"].append(
            self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        )
        metrics["rewards/margins"].append(
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards)
            .mean()
            .item()
        )
        metrics["logps/chosen"].append(
            self.accelerator.gather_for_metrics(chosen_logps.detach()).mean().item()
        )
        metrics["logps/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_logps.detach()).mean().item()
        )
        metrics["logits/chosen"].append(
            self.accelerator.gather_for_metrics(mean_chosen_logits.detach())
            .mean()
            .item()
        )
        metrics["logits/rejected"].append(
            self.accelerator.gather_for_metrics(mean_rejected_logits.detach())
            .mean()
            .item()
        )

        metrics["e_dpo/beta"].append(float(self.beta))
        metrics["e_dpo/beta_tilde_mean"].append(
            self.accelerator.gather_for_metrics(beta_tilde.detach()).mean().item()
        )
        metrics["e_dpo/beta_minus_frac"].append(
            self.accelerator.gather_for_metrics(use_beta_minus.float()).mean().item()
        )
        metrics["e_dpo/beta_plus_frac"].append(
            self.accelerator.gather_for_metrics(use_beta_plus.float()).mean().item()
        )
        metrics["e_dpo/beta_unchanged_frac"].append(
            self.accelerator.gather_for_metrics(unchanged_mask.float()).mean().item()
        )

        if mode == "train":
            self._accumulate_beta_tilde(beta_tilde)
            if self.accelerator.gradient_state.sync_gradients:
                self._apply_beta_update()

        return (loss, outputs) if return_outputs else loss

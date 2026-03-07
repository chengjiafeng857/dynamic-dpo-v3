from contextlib import nullcontext
from typing import Callable, Optional, Union, Literal
import warnings

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.utils import is_torch_xpu_available
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from datasets import Dataset
from dataclasses import dataclass, field
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import selective_log_softmax, entropy_from_logits

@dataclass
class EpsilonDPOConfig(DPOConfig):
    epsilon: float = field(
        default=0.01,
        metadata={
            "help": "Parameter controlling the step size of KL penalty relaxation."
        },
    )

    def __post_init__(self):

        if self.precompute_ref_log_probs == True:
            warnings.warn(
                "When using `EpsilonDPOTrainer`, you should set `precompute_ref_log_probs=False`. "
                "We have set it for you, but you should do it yourself in the future."
            )
            self.precompute_ref_log_probs = False

        if self.loss_type != "sigmoid":
            warnings.warn(
                "When using `EpsilonDPOTrainer`, you should set `loss_type=\"sigmoid\"`. "
                "We have set it for you, but you should do it yourself in the future."
            )           
            self.loss_type = "sigmoid" 

        super().__post_init__()

class EpsilonDPOTrainer(DPOTrainer):
    """
    TRL DPOTrainer + epsilon-DPO:

    Steps:
    (1) Compute log-probabilities from policy and reference model
        policy_chosen_logprob
        policy_rejected_logprob
        ref_chosen_logprob
        ref_rejected_logprob

        Then compute the standard DPO preference margin:

            logits = (policy_chosen_logprob - policy_rejected_logprob) - (ref_chosen_logprob    - ref_rejected_logprob)

    (2) Construct two perturbed policies around the current policy

        Positive perturbation (move policy slightly away from reference):
            p_epsilon_logits = (1 + epsilon) * policy_logits - epsilon * ref_logits

        Negative perturbation (move policy slightly closer to reference):
            n_epsilon_logits = (1 - epsilon) * policy_logits + epsilon * ref_logits

    (3) Compute sequence log-probabilities under perturbed policies
        Compute token logprobs and sum over completion tokens to obtain:
            p_epsilon_logratios
            n_epsilon_logratios
        where
            p_epsilon_logratios = p_epsilon_chosen_logprob - p_epsilon_rejected_logprob

            n_epsilon_logratios = n_epsilon_chosen_logprob - n_epsilon_rejected_logprob

    (4) Estimate epsilon step direction

        Compare perturbed margins with the original margin:
            original_margin = chosen_logprob - rejected_logprob

        step is defined as:
            step = +1   if   p_epsilon_margin > original_margin > n_epsilon_margin
            step = -1   if   n_epsilon_margin > original_margin > p_epsilon_margin
            step =  0   otherwise

        Interpretation:
            +1 → KL constraint can be relaxed
            -1 → KL constraint should be strengthened
            0 → keep KL strength unchanged

    (5) Compute adaptive beta for each sample
            beta_used = beta / (1 + epsilon * step)
        This dynamically adjusts the KL regularization strength
        based on the estimated epsilon step direction.


    (6) Compute epsilon-DPO loss
        Using the standard DPO sigmoid objective with adaptive beta:

            loss_i = - log_sigmoid(beta_used * logits_i)

        Final batch loss:
            loss = mean(loss_i)
    Notes
    -----
    - The original beta is used when computing policy/reference log-probabilities.
    - The adaptive beta (beta_used) is applied only inside the loss.
    - epsilon controls the perturbation magnitude used for step estimation.
    - epsilon-DPO therefore performs an adaptive KL control mechanism
    while preserving the standard DPO objective form.
    """
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[EpsilonDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        assert isinstance(args, EpsilonDPOConfig), "`EpsilonDPOTrainer` requires `EpsilonDPOConfig` for the `args` argument."
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.epsilon = args.epsilon
        self.steps = 0.
    
    def _estimate_epsilon_steps(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate ε-step direction for each example.
        Returns steps in {-1, 0, +1} with shape [batch_size/2].
        """
        # per-token logps under perturbed logits
        p_epsilon_logits = ((1.0 + self.epsilon) * logits) - (self.epsilon * ref_logits)
        n_epsilon_logits = ((1.0 - self.epsilon) * logits) + (self.epsilon * ref_logits)

        p_eps_per_token_logps = selective_log_softmax(p_epsilon_logits, shift_labels)
        n_eps_per_token_logps = selective_log_softmax(n_epsilon_logits, shift_labels)

        p_eps_per_token_logps[shift_completion_mask == 0] = 0.0
        n_eps_per_token_logps[shift_completion_mask == 0] = 0.0

        if self.ld_alpha is None:
            p_eps_logps = p_eps_per_token_logps.sum(dim=1)
            n_eps_logps = n_eps_per_token_logps.sum(dim=1)
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(logits.device)

            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))
            tail_mask = comp_pos > shared_lens.unsqueeze(1)

            p_eps_shared = (p_eps_per_token_logps * shared_mask).sum(dim=1)
            p_eps_tail = (p_eps_per_token_logps * tail_mask).sum(dim=1)
            p_eps_logps = p_eps_shared + self.ld_alpha * p_eps_tail

            n_eps_shared = (n_eps_per_token_logps * shared_mask).sum(dim=1)
            n_eps_tail = (n_eps_per_token_logps * tail_mask).sum(dim=1)
            n_eps_logps = n_eps_shared + self.ld_alpha * n_eps_tail

        all_logps = selective_log_softmax(logits, shift_labels)
        all_logps[shift_completion_mask == 0] = 0.0
        if self.ld_alpha is None:
            base_logps = all_logps.sum(dim=1)
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(logits.device)

            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))
            tail_mask = comp_pos > shared_lens.unsqueeze(1)

            shared_logps = (all_logps * shared_mask).sum(dim=1)
            tail_logps = (all_logps * tail_mask).sum(dim=1)
            base_logps = shared_logps + self.ld_alpha * tail_logps

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
    
    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids, attention_mask, completion_mask = self._truncate_inputs(input_ids, attention_mask, completion_mask)

        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw", "image_sizes", "token_type_ids"):
            if key in inputs:
                model_kwargs[key] = inputs[key]

        # (1) compute policy and reference logprob by using trl internal functions
        # policy forward
        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens

        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # reference forward (no grad)
        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
            ref_shift_logits = None
        else:
            with torch.no_grad():
                ref_outputs = self.ref_model(**model_kwargs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # DPO logits
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)

        # ε-step estimation
        if ref_shift_logits is None:
            steps = torch.zeros_like(logits)
        else:
            steps = self._estimate_epsilon_steps(
                logits=shift_logits,
                ref_logits=ref_shift_logits,
                shift_labels=shift_labels,
                shift_completion_mask=shift_completion_mask,
            )

        updated_beta = self.beta / (1.0 + self.epsilon * steps)

        loss = 0.0
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            if loss_type == "sigmoid":
                per_sequence_loss = (
                    -F.logsigmoid(updated_beta * logits) * (1 - self.label_smoothing)
                    -F.logsigmoid(updated_beta * logits) * self.label_smoothing
                )
            else:
                raise ValueError("EpsilonDPOTrainer currently only supports loss_type='sigmoid'.")

            if self.use_weighting:
                completion_lengths = shift_completion_mask.sum(dim=1).clamp_min(1)
                with torch.no_grad():
                    lse1 = torch.logsumexp(shift_logits, dim=-1)
                    lse2 = torch.logsumexp(2.0 * shift_logits, dim=-1)
                    log_denom = lse2 - 2.0 * lse1
                    aligned_logps = (per_token_logps - log_denom) * shift_completion_mask
                mean_logps = aligned_logps.sum(dim=1) / completion_lengths
                weights = torch.exp(mean_logps)
                chosen_weights, rejected_weights = weights.chunk(2, dim=0)
                per_sequence_loss *= chosen_weights * rejected_weights

            loss += per_sequence_loss.mean() * loss_weight

        # metric logging
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        entropy = per_token_entropy[shift_completion_mask.bool()].mean()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        chosen_logits_det, rejected_logits_det = shift_logits.detach().chunk(2, dim=0)
        chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)

        total_chosen_logits = chosen_logits_det[chosen_mask.bool()].mean(-1).sum()
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_logits_det[rejected_mask.bool()].mean(-1).sum()
        total_rejected_tokens = rejected_mask.sum()

        total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()

        avg_chosen_logits = total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
        avg_rejected_logits = total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        predictions = chosen_logits_det.argmax(dim=-1)
        chosen_mask_acc = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
        chosen_labels_acc = shift_labels[: len(shift_labels) // 2]
        correct_predictions = (predictions == chosen_labels_acc) & chosen_mask_acc
        total_tokens = chosen_mask_acc.sum()
        correct_tokens = correct_predictions.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        chosen_rewards = updated_beta * chosen_logratios.detach()
        rejected_rewards = updated_beta * rejected_logratios.detach()
        agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
        self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
        self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())

        margins = chosen_rewards - rejected_rewards
        agg_margins = self.accelerator.gather(margins)
        self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())

        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

        self._metrics[mode]["kl/p_epsilon_steps"].append(
            self.accelerator.gather_for_metrics((steps == 1).float()).mean().item()
        )
        self._metrics[mode]["kl/n_epsilon_steps"].append(
            self.accelerator.gather_for_metrics((steps == -1).float()).mean().item()
        )
        self._metrics[mode]["kl/beta"].append(float(self.beta))

        if mode == "train":
            self.steps += steps.float().mean().item() / self.args.gradient_accumulation_steps
            if self.accelerator.gradient_state.sync_gradients:
                mean_steps = self.accelerator.gather(
                    torch.tensor(self.steps, device=device, dtype=shift_logits.dtype)
                ).mean()
                self._metrics[mode]["kl/avg_steps"].append(mean_steps.item())
                self.beta = self.beta / (1.0 + mean_steps.item() * self.epsilon)
                self.steps = 0.0

        return (loss, outputs) if return_outputs else loss
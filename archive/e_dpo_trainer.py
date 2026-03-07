from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union
import warnings

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_torch_xpu_available
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import selective_log_softmax


@dataclass
class EpsilonDPOConfig(DPOConfig):
    epsilon: float = field(
        default=0.01,
        metadata={
            "help": "Parameter controlling the step size of KL penalty relaxation."
        },
    )

    def __post_init__(self):
        if self.precompute_ref_log_probs is True:
            warnings.warn(
                "When using `EpsilonDPOTrainer`, you should set "
                "`precompute_ref_log_probs=False`. We have set it for you, "
                "but you should do it yourself in the future."
            )
            self.precompute_ref_log_probs = False

        if self.loss_type != "sigmoid":
            warnings.warn(
                "When using `EpsilonDPOTrainer`, you should set "
                '`loss_type="sigmoid"`. We have set it for you, but you should '
                "do it yourself in the future."
            )
            self.loss_type = "sigmoid"

        super().__post_init__()


class EpsilonDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[EpsilonDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = (None, None),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional[dict] = None,
    ):
        if not isinstance(args, EpsilonDPOConfig):
            raise TypeError("EpsilonDPOTrainer requires args=EpsilonDPOConfig")

        self.model_init = model_init
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

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
            peft_config=peft_config,
        )

        self.epsilon = args.epsilon
        self.steps = 0.0

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        compute_ref_context_manager = (
            amp.autocast(device_type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with torch.no_grad(), compute_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
        return ref_model_output

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
        ref_logits: Optional[torch.FloatTensor] = None,
    ):
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(
            batch, padding_value=self.padding_value
        )

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch[
                "pixel_attention_mask"
            ]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat(
                (prompt_attention_mask, completion_attention_mask), dim=1
            )
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = (
                torch.nonzero(empty_cols)[0].item()
                if empty_cols.any()
                else attention_mask.size(1)
            )
            input_ids = input_ids[:, :first_empty_col]
            attention_mask = attention_mask[:, :first_empty_col]
            loss_mask = loss_mask[:, :first_empty_col]

            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]

            if self.use_num_logits_to_keep:
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                num_logits_to_keep = (
                    loss_mask.shape[1] - first_compute_index
                ).item() + 1
                model_kwargs["num_logits_to_keep"] = num_logits_to_keep

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_num_logits_to_keep:
                labels = labels[:, -num_logits_to_keep:]
                loss_mask = loss_mask[:, -num_logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = (
                    per_token_logps - weights_adjustment_factor
                )
                all_weights = (
                    per_token_logps_adjusted * loss_mask
                ).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(
                    torch.exp(chosen_weights + rejected_weights), max=1
                )

        if self.args.rpo_alpha is not None:
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if ref_logits is not None:
            p_epsilon_logits = ((1 + self.epsilon) * logits) - (
                self.epsilon * ref_logits
            )
            p_epsilon_per_token_logps = torch.gather(
                p_epsilon_logits.log_softmax(-1),
                dim=2,
                index=labels.unsqueeze(2),
            ).squeeze(2)
            p_epsilon_per_token_logps[~loss_mask] = 0
            p_epsilon_per_token_logps = torch.roll(
                p_epsilon_per_token_logps, shifts=1, dims=1
            )

            n_epsilon_logits = ((1 - self.epsilon) * logits) + (
                self.epsilon * ref_logits
            )
            n_epsilon_per_token_logps = torch.gather(
                n_epsilon_logits.log_softmax(-1),
                dim=2,
                index=labels.unsqueeze(2),
            ).squeeze(2)
            n_epsilon_per_token_logps[~loss_mask] = 0
            n_epsilon_per_token_logps = torch.roll(
                n_epsilon_per_token_logps, shifts=1, dims=1
            )

            p_epsilon_all_logps = p_epsilon_per_token_logps.sum(-1)
            n_epsilon_all_logps = n_epsilon_per_token_logps.sum(-1)

            logratios = all_logps[:num_examples] - all_logps[num_examples:]
            p_epsilon_logratios = (
                p_epsilon_all_logps[:num_examples] - p_epsilon_all_logps[num_examples:]
            )
            n_epsilon_logratios = (
                n_epsilon_all_logps[:num_examples] - n_epsilon_all_logps[num_examples:]
            )

            p_epsilon_steps = (p_epsilon_logratios > logratios) & (
                logratios > n_epsilon_logratios
            )
            n_epsilon_steps = (n_epsilon_logratios > logratios) & (
                logratios > p_epsilon_logratios
            )
            output["steps"] = 1 * p_epsilon_steps - 1 * n_epsilon_steps
        else:
            output["logits"] = logits

        mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
        mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        steps: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = logratios - ref_logratios

        updated_beta = self.beta / (1 + self.epsilon * steps)

        losses = (
            -F.logsigmoid(updated_beta * logits) * (1 - self.label_smoothing)
            -F.logsigmoid(updated_beta * logits) * self.label_smoothing
        )

        chosen_rewards = updated_beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = updated_beta * (
            rejected_logps - ref_rejected_logps
        ).detach()

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        ref_model_output = self.compute_ref_log_probs(batch)
        model_output = self.concatenated_forward(
            model, batch, ref_model_output["logits"]
        )

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_model_output["chosen_logps"],
            ref_model_output["rejected_logps"],
            model_output["steps"],
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/accuracies"] = (
            self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(
                chosen_rewards - rejected_rewards
            ).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"])
            .detach()
            .mean()
            .item()
        )
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"])
                .detach()
                .mean()
                .item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"])
                .detach()
                .mean()
                .item()
            )

        if train_eval == "train":
            self.steps += (
                model_output["steps"].float().mean()
                / self.args.gradient_accumulation_steps
            )

            metrics[f"{prefix}kl/p_epsilon_steps"] = (
                self.accelerator.gather_for_metrics(
                    model_output["steps"] == 1
                ).float().mean().item()
            )
            metrics[f"{prefix}kl/n_epsilon_steps"] = (
                self.accelerator.gather_for_metrics(
                    model_output["steps"] == -1
                ).float().mean().item()
            )

            if self.accelerator.gradient_state.sync_gradients:
                mean_steps = self.accelerator.gather(self.steps).mean()

                metrics[f"{prefix}kl/beta"] = self.beta
                metrics[f"{prefix}kl/avg_steps"] = mean_steps

                self.beta = self.beta / (1 + mean_steps * self.epsilon)
                self.steps = 0.0

        return losses.mean(), metrics

    def _estimate_epsilon_steps(
        self,
        shift_logits: torch.Tensor,
        ref_shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        p_epsilon_logits = ((1 + self.epsilon) * shift_logits) - (
            self.epsilon * ref_shift_logits
        )
        p_epsilon_per_token_logps = selective_log_softmax(
            p_epsilon_logits, shift_labels
        )
        p_epsilon_per_token_logps[~shift_completion_mask] = 0

        n_epsilon_logits = ((1 - self.epsilon) * shift_logits) + (
            self.epsilon * ref_shift_logits
        )
        n_epsilon_per_token_logps = selective_log_softmax(
            n_epsilon_logits, shift_labels
        )
        n_epsilon_per_token_logps[~shift_completion_mask] = 0

        all_logps = selective_log_softmax(shift_logits, shift_labels)
        all_logps[~shift_completion_mask] = 0

        num_examples = shift_logits.shape[0] // 2
        base_logratios = all_logps[:num_examples].sum(-1) - all_logps[num_examples:].sum(
            -1
        )
        p_epsilon_logratios = p_epsilon_per_token_logps[:num_examples].sum(
            -1
        ) - p_epsilon_per_token_logps[num_examples:].sum(-1)
        n_epsilon_logratios = n_epsilon_per_token_logps[:num_examples].sum(
            -1
        ) - n_epsilon_per_token_logps[num_examples:].sum(-1)

        p_epsilon_steps = (p_epsilon_logratios > base_logratios) & (
            base_logratios > n_epsilon_logratios
        )
        n_epsilon_steps = (n_epsilon_logratios > base_logratios) & (
            base_logratios > p_epsilon_logratios
        )

        return p_epsilon_steps.to(torch.int64) - n_epsilon_steps.to(torch.int64)

    # Current TRL routes DPO through `_compute_loss`, so keep a behavior-matched
    # bridge for the modern `input_ids` / `completion_mask` batch format.
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
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
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
        logits = outputs.logits
        if logits.shape[:2] != input_ids.shape[:2]:
            logits = logits[:, -input_ids.shape[1] :]

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(completion_mask, shifts=-1, dims=1).bool()
        labels = labels.clone()
        labels[~loss_mask] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        per_token_logps[~loss_mask] = 0.0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
        all_logps = per_token_logps.sum(dim=1)
        num_examples = logits.shape[0] // 2
        chosen_logps = all_logps[:num_examples]
        rejected_logps = all_logps[num_examples:]

        if self.precompute_ref_logps:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
            steps = torch.zeros_like(chosen_logps, dtype=torch.int64)
        else:
            with torch.no_grad():
                ref_outputs = self.ref_model(**model_kwargs)
            ref_logits = ref_outputs.logits
            if ref_logits.shape[:2] != input_ids.shape[:2]:
                ref_logits = ref_logits[:, -input_ids.shape[1] :]
            ref_per_token_logps = torch.gather(
                ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
            ).squeeze(2)
            ref_per_token_logps[~loss_mask] = 0.0
            ref_per_token_logps = torch.roll(ref_per_token_logps, shifts=1, dims=1)
            ref_all_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps = ref_all_logps[:num_examples]
            ref_rejected_logps = ref_all_logps[num_examples:]
            p_epsilon_logits = ((1 + self.epsilon) * logits) - (self.epsilon * ref_logits)
            p_epsilon_per_token_logps = torch.gather(
                p_epsilon_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
            ).squeeze(2)
            p_epsilon_per_token_logps[~loss_mask] = 0.0
            p_epsilon_per_token_logps = torch.roll(
                p_epsilon_per_token_logps, shifts=1, dims=1
            )

            n_epsilon_logits = ((1 - self.epsilon) * logits) + (self.epsilon * ref_logits)
            n_epsilon_per_token_logps = torch.gather(
                n_epsilon_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
            ).squeeze(2)
            n_epsilon_per_token_logps[~loss_mask] = 0.0
            n_epsilon_per_token_logps = torch.roll(
                n_epsilon_per_token_logps, shifts=1, dims=1
            )

            logratios = all_logps[:num_examples] - all_logps[num_examples:]
            p_epsilon_logratios = (
                p_epsilon_per_token_logps[:num_examples].sum(-1)
                - p_epsilon_per_token_logps[num_examples:].sum(-1)
            )
            n_epsilon_logratios = (
                n_epsilon_per_token_logps[:num_examples].sum(-1)
                - n_epsilon_per_token_logps[num_examples:].sum(-1)
            )
            p_epsilon_steps = (p_epsilon_logratios > logratios) & (
                logratios > n_epsilon_logratios
            )
            n_epsilon_steps = (n_epsilon_logratios > logratios) & (
                logratios > p_epsilon_logratios
            )
            steps = 1 * p_epsilon_steps - 1 * n_epsilon_steps

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            steps=steps,
        )

        if self.args.rpo_alpha is not None:
            chosen_labels = labels[:num_examples]
            nll_loss = F.cross_entropy(
                torch.flatten(logits[:num_examples], end_dim=1),
                torch.flatten(chosen_labels),
                ignore_index=0,
            )
            losses = losses + self.args.rpo_alpha * nll_loss

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = (
                    per_token_logps - weights_adjustment_factor
                )
                all_weights = (
                    per_token_logps_adjusted * loss_mask
                ).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                policy_weights = torch.clamp(
                    torch.exp(chosen_weights + rejected_weights), max=1
                )
            losses = losses * policy_weights

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * outputs.aux_loss

        loss = losses.mean()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if loss_mask[:num_examples].any():
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
        else:
            mean_chosen_logits = logits.new_zeros(())
        if loss_mask[num_examples:].any():
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()
        else:
            mean_rejected_logits = logits.new_zeros(())

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
            self.accelerator.gather_for_metrics(chosen_logps).detach().mean().item()
        )
        metrics["logps/rejected"].append(
            self.accelerator.gather_for_metrics(rejected_logps).detach().mean().item()
        )
        metrics["logits/chosen"].append(
            self.accelerator.gather_for_metrics(mean_chosen_logits).detach().mean().item()
        )
        metrics["logits/rejected"].append(
            self.accelerator.gather_for_metrics(mean_rejected_logits)
            .detach()
            .mean()
            .item()
        )
        if self.args.rpo_alpha is not None:
            metrics["nll_loss"].append(
                self.accelerator.gather_for_metrics(nll_loss).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics["aux_loss"].append(
                self.accelerator.gather_for_metrics(outputs.aux_loss)
                .detach()
                .mean()
                .item()
            )

        if mode == "train":
            self.steps += steps.float().mean().item() / self.args.gradient_accumulation_steps
            metrics["kl/p_epsilon_steps"].append(
                self.accelerator.gather_for_metrics((steps == 1).float())
                .mean()
                .item()
            )
            metrics["kl/n_epsilon_steps"].append(
                self.accelerator.gather_for_metrics((steps == -1).float())
                .mean()
                .item()
            )
            if self.accelerator.gradient_state.sync_gradients:
                mean_steps = self.accelerator.gather(
                    torch.tensor(
                        self.steps,
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                ).mean()
                metrics["kl/beta"].append(float(self.beta))
                metrics["kl/avg_steps"].append(mean_steps.item())
                self.beta = self.beta / (1 + mean_steps.item() * self.epsilon)
                self.steps = 0.0

        return (loss, outputs) if return_outputs else loss

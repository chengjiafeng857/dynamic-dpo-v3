import json
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass, field
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import selective_log_softmax


@dataclass
class BetaDPOConfig(DPOConfig):
    """
    TRL-style config for beta-DPO.

    Mirrors the paper + their repo:
    - selection ratio rho (keep ratio)
    - alpha scaling factor (called 'a' in the repo)
    - EMA momentum m (called gamma in the repo)
    - gaussian sampling based on (gap_mean, gap_std)
    """
    # data filtering: keep ratio rho (paper uses ρ, default 0.8)
    rho: float = field(default=0.8)
    # scaling factor α (paper); the repo often calls it loss_config.a          
    alpha: float = field(default=1.0)   
    # EMA momentum m (paper Eq.7, Eq.9), repo uses gamma=0.9     
    ema_momentum: float = field(default=0.9) 
    # numerical stability
    beta_min: float = field(default=1e-3)
    # distributed / reproducibility
    sync_global_mask: bool = field(default=True)
    # diagnostic logging
    log_samples: int = field(default=0)
    log_dir: str = field(default="logs/beta_samples")

    def __post_init__(self):
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must be in (0, 1].")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if not (0.0 <= self.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if self.log_samples < 0:
            raise ValueError("log_samples must be >= 0.")
        super().__post_init__()


class BetaDPOTrainer(DPOTrainer):
    """
    TRL DPOTrainer + beta-DPO:

    Steps:
    (1) Compute the reward discrepancy for each batch
      r_gap = (policy_chosen_logprob - ref_chosen_logprob) - (policy_rejected_logprob - ref_rejected_logprob)

    (2) If using distributed training:
      gather r_gap from all processes

    (3) compute the threshold r_gap_mean and r_gap_std by using EMA
      matches their repo: each rank EMA-updates then all_reduce average

    (4) Gaussian-weighted sampling without replacement over reward_i:
      w_i = exp(-0.5*((r_gap_i - r_gap_mean)/r_gap_std)^2))
      sample_num = int(N * rho)
      global_mask is 1 for selected indices

    (5) compute the new beta by using sampled subset:
      r_used = mean(r[selected])
      beta_used = beta * (1 + alpha * (r_used - r_gap_mean))
      beta_used = beta_used.clamp(1e-3)

    (6) compute the dpo loss by using sampled reward discrepancy subset and new beta:
      loss = - logsigmoid(beta_used * logits)
      sum(losses * mask_local)/sum(mask_local)

    Note:
      Using original beta to compute chosen and rejected logprob
      beta_used is detached in loss (consistent with their code).
      
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, BetaDPOConfig):
            raise TypeError("BetaDPOTrainer requires args=BetaDPOConfig")

        device = self.accelerator.device
        # define initial r_gam_mean: 0.0 and r_gap_std: 1.0
        self.r_gap_mean = torch.zeros((), device=device)
        self.r_gap_std = torch.ones((), device=device)
        self._gap_std_eps = 1e-6
        self._logged_sample_count = 0
        self._printed_console_sample = False
        self._sample_log_path = os.path.join(
            str(self.args.log_dir), "beta_sample_ids.jsonl"
        )

    def _is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.accelerator.num_processes > 1

    @torch.no_grad()
    def ema_update_gap_mean_and_std(self, r_gap_global: torch.Tensor):
        """
            Match their repo update_and_sync_tensor_mean():
            - compute batch mean/std from global gathered gap
            - EMA update on each rank
            - then all_reduce SUM and / world_size to make them identical across ranks
        """
        m = float(self.args.ema_momentum)
        batch_r_mean = r_gap_global.mean()
        batch_r_std = r_gap_global.std(unbiased=False)

        self.r_gap_mean.mul_(m).add_(batch_r_mean, alpha=1.0 - m)
        self.r_gap_std.mul_(m).add_(batch_r_std, alpha=1.0 - m)

        if self._is_distributed():
            dist.all_reduce(self.r_gap_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.r_gap_std, op=dist.ReduceOp.SUM)
            self.r_gap_mean.div_(self.accelerator.num_processes)
            self.r_gap_std.div_(self.accelerator.num_processes)

    @torch.no_grad()
    def sample_data_mask(self, r_gap_global: torch.Tensor) -> torch.Tensor:
        """
            Gaussian-weighted sampling without replacement (matches their beta_DPO branch).
            weight: Gaussian weight
            N: total num of samples in global batch
            rho: sampling ratio
            k: num of samples to select k = int(N * rho)
            Returns:
            global_mask: shape [N_global], float tensor in {0,1}
        """
        r_mean = self.r_gap_mean
        r_std = torch.clamp(self.r_gap_std, min=self._gap_std_eps)

        weight = torch.exp(-0.5 * ((r_gap_global - r_mean) / r_std).pow(2))
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        if float(weight.sum()) <= 0.0:
            weight = torch.ones_like(weight)
        N = weight.numel()
        k = max(1, min(N, int(N * float(self.args.rho))))

        if self.args.sync_global_mask and self._is_distributed():
            if self.accelerator.is_main_process:
                idx = torch.multinomial(weight, k, replacement=False)
                mask = torch.zeros_like(weight)
                mask[idx] = 1.0
            else:
                mask = torch.zeros_like(weight)

            self.accelerator.wait_for_everyone()
            dist.broadcast(mask, src=0)
            return mask.detach()

        idx = torch.multinomial(weight, k, replacement=False)
        mask = torch.zeros_like(weight)
        mask[idx] = 1.0
        return mask.detach()

    @torch.no_grad()
    def compute_beta_used(self, r_gap_global: torch.Tensor, global_mask: torch.Tensor, beta: float) -> torch.Tensor:
        """
            beta_used = beta * (1 + alpha*(r_used - r_gap_mean))
            where r_used = mean(r[selected by global_mask])
        """
        select = global_mask.bool()
        if bool(select.any()):
            r_used = r_gap_global[select].mean()
        else:
            r_used = r_gap_global.mean()
        beta_used = beta * (1.0 + float(self.args.alpha) * (r_used - self.r_gap_mean))
        beta_used = torch.clamp(beta_used, min=float(self.args.beta_min))
        return beta_used

    def _slice_global_to_local(self, global_vec: torch.Tensor, local_bsz: int) -> torch.Tensor:
        """
            Slice a global vector [N_global] into local chunk for current rank.
            Assumes equal per-rank batch size (drop_last=True recommended).
        """
        rank = self.accelerator.process_index
        start = rank * local_bsz
        end = start + local_bsz
        return global_vec[start:end]

    @torch.no_grad()
    def _maybe_log_sample_ids(
        self,
        input_ids: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_completion_mask: torch.Tensor,
    ) -> None:
        max_logged = int(self.args.log_samples)
        if max_logged <= 0 or not self.model.training or not self.is_world_process_zero():
            return

        batch_size = int(input_ids.size(0))
        if batch_size == 0:
            return

        if not self._printed_console_sample:
            random_index = int(torch.randint(0, batch_size, (1,)).item())
            sample_loss_labels = shift_labels[random_index].masked_fill(
                shift_completion_mask[random_index] == 0, -100
            )
            print(
                f"[BETA_DPO] random_sample_input_ids={input_ids[random_index].detach().cpu().tolist()}"
            )
            print(
                f"[BETA_DPO] random_sample_labels={shift_labels[random_index].detach().cpu().tolist()}"
            )
            print(
                "[BETA_DPO] random_sample_loss_labels="
                f"{sample_loss_labels.detach().cpu().tolist()}"
            )
            self._printed_console_sample = True

        remaining = max_logged - self._logged_sample_count
        if remaining <= 0:
            return

        os.makedirs(str(self.args.log_dir), exist_ok=True)
        rows_to_log = min(batch_size, remaining)
        with open(self._sample_log_path, "a", encoding="utf-8") as handle:
            for batch_index in range(rows_to_log):
                loss_labels = shift_labels[batch_index].masked_fill(
                    shift_completion_mask[batch_index] == 0, -100
                )
                record = {
                    "global_step": int(self.state.global_step),
                    "batch_index": batch_index,
                    "input_ids": input_ids[batch_index].detach().cpu().tolist(),
                    "labels": shift_labels[batch_index].detach().cpu().tolist(),
                    "completion_mask": shift_completion_mask[batch_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "loss_labels": loss_labels.detach().cpu().tolist(),
                }
                handle.write(json.dumps(record) + "\n")

        self._logged_sample_count += rows_to_log

    def _compute_loss(self, model, inputs, return_outputs):
        """
            Core TRL hook. We override it to implement beta-DPO.
        """

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
        self._maybe_log_sample_ids(input_ids, shift_labels, shift_completion_mask)
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens

        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # reference forward (no grad)
        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            with torch.no_grad():
                ref_outputs = self.ref_model(**model_kwargs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # DPO logits
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)

        # (2) reward discrenpancy
        # r_gap = (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)
        r_gap_local = (chosen_logps - ref_chosen_logps - rejected_logps + ref_rejected_logps).detach()
        r_gap_global = self.accelerator.gather_for_metrics(r_gap_local)

        # (3) EMA update on reward gap mean and std
        if mode == "train":
            self.ema_update_gap_mean_and_std(r_gap_global)

        # (4) Sample global mask based on Gaussian weight 
        global_mask = self.sample_data_mask(r_gap_global)

        # (5) Compute beta_used from selected subset
        beta = float(self.beta)
        beta_used = self.compute_beta_used(r_gap_global, global_mask, beta=beta)

        # (6) Losses then apply local mask 
        local_bsz = chosen_logps.size(0)
        mask_local = self._slice_global_to_local(global_mask, local_bsz).to(logits.device)
        losses_local = -F.logsigmoid(beta_used.detach() * logits)
        loss = (losses_local * mask_local).sum() / mask_local.sum().clamp(min=1.0)

        # (7) Keep TRL-like reward metrics (use base beta, not beta_used)
        self._metrics[mode]["beta_dpo/gap_mean"].append(self.r_gap_mean.item())
        self._metrics[mode]["beta_dpo/gap_std"].append(self.r_gap_std.item())
        self._metrics[mode]["beta_dpo/beta_used"].append(float(beta_used))

        return (loss, outputs) if return_outputs else loss

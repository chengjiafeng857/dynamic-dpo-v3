import torch
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer.utils import selective_log_softmax
import os
import numpy as np
import json

# margin-log function
def log_margin(margin, epoch_dir, epoch, step, JSONL_PATH):
    # full array
    m = margin.detach().float().cpu().numpy()  
                
    # 1) save full margins 
    # step: batch index
    npy_path = os.path.join(epoch_dir, f"step_{step:05d}.npy")
    np.save(npy_path, m)

    # 2) write a readable per-batch record to ONE jsonl file
    # summary stats
    # quantiles
    p10, p50, p90 = np.percentile(m, [10, 50, 90])
                
    record = {
        "epoch": int(epoch),
        "step": int(step),
        "batch_size": int(m.shape[0]),
        "mean": float(m.mean()),
        "std": float(m.std(ddof=0)),
        "min": float(m.min()),
        "p10": float(p10),
        "median": float(p50),
        "p90": float(p90),
        "max": float(m.max()),
        "pos_frac": float((m > 0).mean()),
        "npy": npy_path,
        "sample": [float(x) for x in m[:]],
    }
 
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class MarginDPOTrainer(DPOTrainer):
    def __init__(
        self,
        margin_log_path="./margin_logs",
        sample_log_path="logs/margin_samples",
        log_samples=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.margin_log_path = margin_log_path
        self.sample_log_path = sample_log_path
        self.log_samples = int(log_samples)
        if self.log_samples < 0:
            raise ValueError("log_samples must be >= 0.")
        self._logged_sample_count = 0
        self._printed_console_sample = False
        self._sample_log_jsonl_path = os.path.join(
            self.sample_log_path, "margin_sample_ids.jsonl"
        )
        os.makedirs(self.margin_log_path, exist_ok=True)

    def _maybe_log_margin(self, margin_tensor: torch.Tensor):
        """
        margin_tensor: shape [local_batch] on each rank
        """
        # only log margin on training steps
        if not self.model.training:
            return
        
        # gather all ranks into one tensor
        margin_all = self.accelerator.gather_for_metrics(margin_tensor.detach())

        if not self.is_world_process_zero():
            return
        
        # only one epoch
        step = self.state.global_step
        epoch = 0
        os.makedirs(self.margin_log_path, exist_ok=True)

        jsonl_path = os.path.join(self.margin_log_path, "margins.jsonl")
        log_margin(margin_all, self.margin_log_path, epoch, step=step, JSONL_PATH=jsonl_path)

    @torch.no_grad()
    def _maybe_log_sample_ids(
        self,
        input_ids: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_completion_mask: torch.Tensor,
    ) -> None:
        if self.log_samples <= 0 or not self.model.training or not self.is_world_process_zero():
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
                f"[MARGIN_DPO] random_sample_input_ids={input_ids[random_index].detach().cpu().tolist()}"
            )
            print(
                f"[MARGIN_DPO] random_sample_labels={shift_labels[random_index].detach().cpu().tolist()}"
            )
            print(
                "[MARGIN_DPO] random_sample_loss_labels="
                f"{sample_loss_labels.detach().cpu().tolist()}"
            )
            self._printed_console_sample = True

        remaining = self.log_samples - self._logged_sample_count
        if remaining <= 0:
            return

        os.makedirs(self.sample_log_path, exist_ok=True)
        rows_to_log = min(batch_size, remaining)
        with open(self._sample_log_jsonl_path, "a", encoding="utf-8") as handle:
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
        Use trl dpo loss compute pipelines, only get and log margins
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

        # policy logprobs
        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        self._maybe_log_sample_ids(input_ids, shift_labels, shift_completion_mask)
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens

        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # reference logprobs 
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
        
        # margin = (policy_c - policy_r) - (ref_c - ref_r)
        margin = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        margin_tensor = margin.detach()
        self._maybe_log_margin(margin_tensor)

        # compute loss
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        if self.f_divergence_type == "reverse_kl":  # standard DPO
            chosen_scores = chosen_logratios
            rejected_scores = rejected_logratios
        elif self.f_divergence_type == "forward_kl":
            # f'(t) = 1 - 1/t  -> drop constant -> -exp(-logratio)
            chosen_scores = -torch.exp(-chosen_logratios)
            rejected_scores = -torch.exp(-rejected_logratios)
        elif self.f_divergence_type == "js_divergence":
            # f'(t) = log(2t/(t+1)) -> drop log 2
            chosen_scores = F.logsigmoid(chosen_logratios)
            rejected_scores = F.logsigmoid(rejected_logratios)
        elif self.f_divergence_type == "alpha_divergence":
            # alpha-divergence: f'(t) = (t^(α-1) - 1)/(α-1)
            if abs(self.f_alpha_divergence_coef - 1.0) < 1e-6:  # limit case f'(t) -> log(t), fall back to reverse_kl
                chosen_scores = chosen_logratios
                rejected_scores = rejected_logratios
            else:
                coef = 1.0 / (self.f_alpha_divergence_coef - 1.0)
                t_chosen = (self.f_alpha_divergence_coef - 1.0) * chosen_logratios
                t_rejected = (self.f_alpha_divergence_coef - 1.0) * rejected_logratios
                dtype = t_chosen.dtype
                # Clamp max so exp(.) stays representable after casting back
                clamp_max = {torch.float16: 11.0, torch.bfloat16: 80.0, torch.float32: 80.0}[dtype]
                t_chosen_float = torch.clamp(t_chosen.float(), max=clamp_max)
                t_rejected_float = torch.clamp(t_rejected.float(), max=clamp_max)
                chosen_scores = torch.exp(t_chosen_float).to(dtype) * coef
                rejected_scores = torch.exp(t_rejected_float).to(dtype) * coef
        else:
            raise ValueError(f"Unknown f_divergence_type: {self.f_divergence_type}")

        delta_score = chosen_scores - rejected_scores

        beta = getattr(self, "beta", 0.1)
        per_seq_loss = -F.logsigmoid(beta * delta_score)
        loss = per_seq_loss.mean()

        return (loss, outputs) if return_outputs else loss
        

        


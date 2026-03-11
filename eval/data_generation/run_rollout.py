from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

from .hh_parser import extract_prompt_and_reference, messages_have_raw_role_tags
from .rollout import RMJudge, RolloutGenerator
from .utils import load_model, load_tokenizer, seed_everything
from util import LLAMA3_CHAT_TEMPLATE

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
SPECIAL_TOKEN_RE = re.compile(r"<\\|[^>]+?\\|>")


def _is_effectively_empty(text: Optional[str]) -> bool:
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    stripped = SPECIAL_TOKEN_RE.sub("", stripped).strip()
    return not stripped

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_dpo.yaml")
    for name, arg_type in (
        ("output_dir", str),
        ("limit", int),
        ("batch_size", int),
        ("k", int),
        ("responses_per_prompt", int),
        ("temperature", float),
        ("top_p", float),
        ("max_new_tokens", int),
        ("min_new_tokens", int),
        ("seed", int),
        ("device_map", str),
        ("judge", str),
        ("reward_model", str),
        ("reward_batch_size", int),
        ("reward_judge_batch_size", int),
        ("reward_precision", str),
        ("reward_device_map", str),
        ("reward_max_length", int),
        ("reward_quantization", str),
        ("debug_log_path", str),
        ("debug_log_max", int),
        ("flush_every_batches", int),
    ):
        parser.add_argument(f"--{name}", type=arg_type, default=None)
    parser.add_argument("--reward_load_in_8bit", action="store_true", default=None)
    parser.add_argument("--debug_log_all", action="store_true", default=None)
    parser.add_argument("--debug_log_empty_only", action="store_true", default=None)
    parser.add_argument("--log_throughput", action="store_true", default=None)
    parser.add_argument("--no_log_throughput", action="store_true", default=None)
    parser.add_argument("--engine", type=str, default=None, choices=["st", "vllm"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--vllm_batch_size", type=int, default=None)
    return parser.parse_args()


def resolve_rollout_cfg(config: Dict, args: argparse.Namespace) -> Dict:
    rollout_cfg = config.get("rollout", {})
    dataset_cfg = config.get("dataset", {})

    def pick(key: str, default):
        value = getattr(args, key)
        return rollout_cfg.get(key, default) if value is None else value

    return {
        "dataset_name": dataset_cfg.get("dataset_name", "Anthropic/hh-rlhf"),
        "subset": dataset_cfg.get("subset", "train"),
        "model_name": rollout_cfg.get("model_name"),
        "seed": args.seed if args.seed is not None else dataset_cfg.get("seed", 42),
        "output_dir": args.output_dir or rollout_cfg.get("output_dir", "rollout_output"),
        "limit": args.limit if args.limit is not None else rollout_cfg.get("limit"),
        "batch_size": pick("batch_size", 4),
        "responses_per_prompt": pick(
            "responses_per_prompt",
            rollout_cfg.get("responses_per_prompt", rollout_cfg.get("k", 8)),
        ),
        "temperature": pick("temperature", 0.7),
        "top_p": pick("top_p", 0.9),
        "max_new_tokens": pick("max_new_tokens", 512),
        "min_new_tokens": pick("min_new_tokens", 10),
        "device_map": args.device_map if args.device_map is not None else rollout_cfg.get("device_map"),
        "judge": pick("judge", "rm"),
        "reward_model": pick("reward_model", "RLHFlow/ArmoRM-Llama3-8B-v0.1"),
        "reward_batch_size": pick("reward_batch_size", 4),
        "reward_judge_batch_size": pick("reward_judge_batch_size", 1),
        "reward_precision": pick("reward_precision", None),
        "reward_device_map": pick("reward_device_map", None),
        "reward_max_length": pick("reward_max_length", None),
        "reward_quantization": pick("reward_quantization", rollout_cfg.get("reward_quantization", "8bit")),
        "reward_load_in_8bit": (
            rollout_cfg.get("reward_load_in_8bit", False)
            if args.reward_load_in_8bit is None
            else args.reward_load_in_8bit
        ),
        "debug_log_path": pick("debug_log_path", None),
        "debug_log_all": (
            rollout_cfg.get("debug_log_all", False)
            if args.debug_log_all is None
            else args.debug_log_all
        ),
        "debug_log_empty_only": (
            rollout_cfg.get("debug_log_empty_only", True)
            if args.debug_log_empty_only is None
            else args.debug_log_empty_only
        ),
        "debug_log_max": pick("debug_log_max", None),
        "flush_every_batches": pick("flush_every_batches", 1),
        "log_throughput": rollout_cfg.get("log_throughput", True),
        "engine": args.engine if args.engine is not None else rollout_cfg.get("engine", "st"),
        "gpu_memory_utilization": pick("gpu_memory_utilization", 0.9),
        "vllm_batch_size": (
            args.vllm_batch_size
            if args.vllm_batch_size is not None
            else rollout_cfg.get(
                "vllm_batch_size",
                rollout_cfg.get("batch_size", 500),
            )
        ),
    }


def _ensure_generation_prompt(prompt_text: str) -> str:
    trimmed = prompt_text.rstrip()
    if trimmed.endswith(ASSISTANT_HEADER.rstrip()):
        return prompt_text
    return f"{prompt_text}{ASSISTANT_HEADER}"


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rollout_cfg = resolve_rollout_cfg(config, args)
    if args.log_throughput:
        rollout_cfg["log_throughput"] = True
    if args.no_log_throughput:
        rollout_cfg["log_throughput"] = False

    model_name = rollout_cfg.get("model_name") or config.get("policy_name") or config.get("model_name")
    if not model_name:
        raise ValueError("Missing policy_name in config or --model_name override.")

    # Setup Generator
    if rollout_cfg["engine"] == "vllm":
        # vLLM manages its own tokenizer/model loading and seeding
        # We explicitly seed random/numpy here but avoid touching torch.cuda to prevent fork errors
        import random
        import numpy as np
        random.seed(int(rollout_cfg["seed"]))
        np.random.seed(int(rollout_cfg["seed"]))
        
        from .rollout import VLLMRolloutGenerator
        
        # vLLM manages its own tokenizer/model loading
        # We need a temporary tokenizer just for prompt formatting if not loaded
        # But VLLMRolloutGenerator creates one. We'll use a cheap local one for templating logic.
        dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        precision = config.get("precision", "auto")
        vllm_dtype = dtype_map.get(precision, precision) # Fallback to auto or direct string
        
        print(f"Initializing vLLM generator with dtype={vllm_dtype}...")
        generator = VLLMRolloutGenerator(
            model_name=model_name,
            seed=int(rollout_cfg["seed"]),
            gpu_memory_utilization=float(rollout_cfg["gpu_memory_utilization"]),
            dtype=vllm_dtype,
        )
        tokenizer = generator.tokenizer # Use vLLM's tokenizer
        if not tokenizer.chat_template:
            tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

        # vLLM is efficient with large batches
        batch_size = int(rollout_cfg["vllm_batch_size"])
        model = None # No HF model object
    else:
        # Standard HF path
        seed_everything(int(rollout_cfg["seed"]))
        tokenizer = load_tokenizer(model_name, padding_side="left")
        device_map = rollout_cfg["device_map"] or "auto"
        model = load_model(
            model_name,
            precision=config.get("precision"),
            device_map=device_map,
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"Generator device: {device}")
        
        responses_per_prompt = int(rollout_cfg["responses_per_prompt"])
        gen_kwargs = {
            "do_sample": True,
            "temperature": float(rollout_cfg["temperature"]),
            "top_p": float(rollout_cfg["top_p"]),
            "max_new_tokens": int(rollout_cfg["max_new_tokens"]),
            "min_new_tokens": int(rollout_cfg["min_new_tokens"]),
        }
        generator = RolloutGenerator(
            model=model,
            tokenizer=tokenizer,
            num_return_sequences=responses_per_prompt,
            **gen_kwargs,
        )
        batch_size = int(rollout_cfg["batch_size"])

    # Common Logic
    judge_name = str(rollout_cfg["judge"]).lower()
    if judge_name not in ("rm", "reward", "pairrm"):
        raise ValueError(f"Unsupported judge '{judge_name}'. Use 'rm'.")
    quantization = str(rollout_cfg["reward_quantization"]).lower()
    load_in_8bit = quantization in ("8bit", "int8", "bnb8")
    if rollout_cfg["reward_load_in_8bit"]:
        load_in_8bit = True

    output_dir = rollout_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, "rollout_responses.jsonl")
    judged_path = os.path.join(output_dir, "rollout_judged.jsonl")
    debug_path = rollout_cfg.get("debug_log_path") or os.path.join(output_dir, "rollout_debug.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")

    responses_per_prompt = int(rollout_cfg["responses_per_prompt"])
    gen_kwargs = { # Duplicate for manifest logging
        "do_sample": True,
        "temperature": float(rollout_cfg["temperature"]),
        "top_p": float(rollout_cfg["top_p"]),
        "max_new_tokens": int(rollout_cfg["max_new_tokens"]),
        "min_new_tokens": int(rollout_cfg["min_new_tokens"]),
    }
    
    meta_base = {
        "source": "hh_rollout",
        "seed": int(rollout_cfg["seed"]),
        "k_candidates": responses_per_prompt,
        "generator_model": model_name,
    }
    meta_base["judge"] = "rm"
    meta_base["reward_model"] = rollout_cfg["reward_model"]
    meta_base["reward_quantization"] = "8bit" if load_in_8bit else "none"
    manifest = {
        "dataset_name": rollout_cfg["dataset_name"],
        "subset": rollout_cfg["subset"],
        "generation_kwargs": gen_kwargs,
        "responses_file": responses_path,
        "judged_file": judged_path,
        "debug_file": debug_path if (rollout_cfg["debug_log_all"] or rollout_cfg["debug_log_empty_only"]) else None,
        "engine": rollout_cfg["engine"],
        **meta_base,
    }

    raw_ds = load_dataset(rollout_cfg["dataset_name"], split=rollout_cfg["subset"])
    limit = rollout_cfg["limit"]
    
    processed = 0
    generated = 0
    debug_remaining = (
        int(rollout_cfg["debug_log_max"]) if rollout_cfg["debug_log_max"] is not None else None
    )
    flush_every_batches = max(1, int(rollout_cfg["flush_every_batches"]))
    batch_counter = 0
    buffer: List[dict] = []

    def flush(batch: List[dict], responses_f, debug_f, debug_enabled: bool) -> bool:
        nonlocal generated, debug_remaining
        start = time.perf_counter()
        prompt_texts = [b["prompt_text"] for b in batch]
        
        # Branch for generator type
        if rollout_cfg["engine"] == "vllm":
            if debug_enabled and rollout_cfg["log_throughput"]:
                candidates, raw_candidates, token_counts = generator.generate_batch(
                    prompt_texts,
                    num_return_sequences=responses_per_prompt,
                    return_raw=True,
                    return_token_counts=True,
                    **gen_kwargs,
                )
            elif debug_enabled:
                candidates, raw_candidates = generator.generate_batch(
                    prompt_texts,
                    num_return_sequences=responses_per_prompt,
                    return_raw=True,
                    **gen_kwargs,
                )
                token_counts = None
            elif rollout_cfg["log_throughput"]:
                candidates, token_counts = generator.generate_batch(
                    prompt_texts,
                    num_return_sequences=responses_per_prompt,
                    return_token_counts=True,
                    **gen_kwargs,
                )
                raw_candidates = [None] * len(candidates)
            else:
                candidates = generator.generate_batch(
                    prompt_texts,
                    num_return_sequences=responses_per_prompt,
                    **gen_kwargs,
                )
                raw_candidates = [None] * len(candidates)
                token_counts = None
        else:
            # Standard HF logic
            if debug_enabled and rollout_cfg["log_throughput"]:
                candidates, raw_candidates, token_counts = generator.generate_batch(
                    prompt_texts, return_raw=True, return_token_counts=True
                )
            elif debug_enabled:
                candidates, raw_candidates = generator.generate_batch(
                    prompt_texts, return_raw=True
                )
                token_counts = None
            elif rollout_cfg["log_throughput"]:
                candidates, token_counts = generator.generate_batch(
                    prompt_texts, return_token_counts=True
                )
                raw_candidates = [None] * len(candidates)
            else:
                candidates = generator.generate_batch(prompt_texts)
                raw_candidates = [None] * len(candidates)
                token_counts = None

        elapsed = time.perf_counter() - start
        if rollout_cfg["log_throughput"] and elapsed > 0:
            count_str = ""
            if token_counts is not None:
                total_tokens = sum(sum(counts) for counts in token_counts)
                count_str = f"| {total_tokens} tokens | {total_tokens / elapsed:.1f} tok/s"
            elif rollout_cfg["engine"] == "vllm":
                 # Estimate for vLLM since we don't count tokens explicitly to save time
                 count_str = f"| {len(batch)*responses_per_prompt} sequences"

            print(
                f"Batch {len(batch)} prompts {count_str}"
            )

        for item, cand_list, raw_list in tqdm(
            list(zip(batch, candidates, raw_candidates)),
            desc="Saving responses",
            leave=False,
        ):
            responses = [c.strip() for c in cand_list]
            responses_f.write(
                json.dumps(
                    {
                        "prompt_messages": item["prompt_messages"],
                        "responses": responses,
                        "reference_responses": item.get("reference_responses"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            generated += 1
            if debug_f is not None and raw_list is not None:
                empty_indices = [i for i, resp in enumerate(responses) if _is_effectively_empty(resp)]
                should_log = rollout_cfg["debug_log_all"] or (
                    rollout_cfg["debug_log_empty_only"] and empty_indices
                )
                if should_log and (debug_remaining is None or debug_remaining > 0):
                    debug_f.write(
                        json.dumps(
                            {
                                "prompt_messages": item["prompt_messages"],
                                "prompt_text": item["prompt_text"],
                                "raw_responses": raw_list,
                                "cleaned_responses": responses,
                                "empty_indices": empty_indices,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if debug_remaining is not None:
                        debug_remaining -= 1
            if limit is not None and generated >= int(limit):
                return True
        return False

    debug_enabled = rollout_cfg["debug_log_all"] or rollout_cfg["debug_log_empty_only"]
    with open(responses_path, "w", encoding="utf-8") as responses_f, (
        open(debug_path, "w", encoding="utf-8") if debug_enabled else open(os.devnull, "w")
    ) as debug_f:
        for row in tqdm(raw_ds, desc="Rollout prompts"):
            text = row.get("chosen") if isinstance(row, dict) else None
            rejected_text = row.get("rejected") if isinstance(row, dict) else None
            if not text:
                continue
            prompt_messages, reference_response = extract_prompt_and_reference(text)
            _, reference_rejected = (
                extract_prompt_and_reference(rejected_text) if rejected_text else (None, None)
            )
            if not prompt_messages or messages_have_raw_role_tags(prompt_messages):
                continue
            
            # Apply Template using the tokenizer (vLLM or HF both have it)
            buffer.append(
                {
                    "prompt_messages": prompt_messages,
                    "prompt_text": _ensure_generation_prompt(
                        tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                    ),
                    "reference_responses": {
                        "chosen": reference_response,
                        "rejected": reference_rejected,
                    },
                }
            )

            if len(buffer) < batch_size:
                continue
            if flush(buffer, responses_f, debug_f if debug_enabled else None, debug_enabled):
                buffer = []
                batch_counter += 1
                if batch_counter % flush_every_batches == 0:
                    responses_f.flush()
                    if debug_f is not None:
                        debug_f.flush()
                break
            buffer = []
            batch_counter += 1
            if batch_counter % flush_every_batches == 0:
                responses_f.flush()
                if debug_f is not None:
                    debug_f.flush()

        if buffer and (limit is None or generated < int(limit)):
            flush(buffer, responses_f, debug_f if debug_enabled else None, debug_enabled)
            batch_counter += 1
            if batch_counter % flush_every_batches == 0:
                responses_f.flush()
                if debug_f is not None:
                    debug_f.flush()
        responses_f.flush()
        if debug_f is not None:
            debug_f.flush()

    # Teardown Generator if needed
    if rollout_cfg["engine"] == "vllm":
        # Cleanup vLLM to free VRAM for RM
        import gc
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif load_in_8bit:
        # Release the generator to free VRAM before loading the RM.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Re-seed exactly before Reward Model to ensure deterministic judging
    # This is safe here because vLLM subprocesses are done/detached or we are single-process.
    seed_everything(int(rollout_cfg["seed"]))
    judge = RMJudge(
        model_name=rollout_cfg["reward_model"],
        precision=rollout_cfg["reward_precision"] or config.get("precision"),
        device_map=rollout_cfg["reward_device_map"],
        load_in_8bit=load_in_8bit,
        batch_size=int(rollout_cfg["reward_batch_size"]),
        max_length=rollout_cfg["reward_max_length"],
        seed=int(rollout_cfg["seed"]),
    )

    reward_judge_batch_size = max(1, int(rollout_cfg.get("reward_judge_batch_size", 1)))
    with open(responses_path, "r", encoding="utf-8") as responses_f, open(
        judged_path, "w", encoding="utf-8"
    ) as judged_f:
        judge_buffer: List[dict] = []

        def flush_judge_batch(batch: List[dict]) -> bool:
            nonlocal processed
            prompts = [entry["item"]["prompt_messages"] for entry in batch]
            candidates_list = [entry["cleaned"] for entry in batch]
            results = judge.rank_batch(prompts, candidates_list)
            for entry, (best_local, worst_local) in zip(batch, results):
                responses = entry["responses"]
                idx_map = entry["idx_map"]
                best_idx = idx_map[best_local]
                worst_idx = idx_map[worst_local]
                metadata = dict(meta_base)
                if "reference_responses" in entry["item"]:
                    metadata["reference_responses"] = entry["item"]["reference_responses"]
                record = {
                    "prompt_messages": entry["item"]["prompt_messages"],
                    "chosen": [{"role": "assistant", "content": responses[best_idx]}],
                    "rejected": [{"role": "assistant", "content": responses[worst_idx]}],
                    "metadata": metadata,
                }
                judged_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                judged_f.flush()
                processed += 1
                if limit is not None and processed >= int(limit):
                    return True
            return False

        stop = False
        for line in tqdm(responses_f, desc="RM judging"):
            item = json.loads(line)
            responses = item.get("responses", [])
            nonempty = [
                (idx, resp) for idx, resp in enumerate(responses) if not _is_effectively_empty(resp)
            ]
            if len(nonempty) < 2:
                continue
            idx_map, cleaned = zip(*nonempty)
            judge_buffer.append(
                {
                    "item": item,
                    "responses": responses,
                    "idx_map": idx_map,
                    "cleaned": list(cleaned),
                }
            )
            if len(judge_buffer) >= reward_judge_batch_size:
                if flush_judge_batch(judge_buffer):
                    stop = True
                    break
                judge_buffer = []

        if not stop and judge_buffer:
            flush_judge_batch(judge_buffer)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {generated} rows to {responses_path}")
    print(f"Wrote {processed} rows to {judged_path}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()

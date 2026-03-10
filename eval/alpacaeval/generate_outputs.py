"""
Generate model outputs for AlpacaEval 2.0 evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def _load_dataset_from_file(data_file: str) -> List[dict]:
    """load dataset from a local JSON/JSONL/Parquet file. Using datasets library."""
    suffix = Path(data_file).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=data_file)
    elif suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=data_file)
    else:
        raise ValueError(f"Unsupported data file format: {data_file}")

    split = "train" if "train" in dataset else next(iter(dataset.keys()))
    return list(dataset[split])


def _select_dataset_file(repo_id: str) -> str:
    """Select the most appropriate dataset file from the AlpacaEval dataset repo."""
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(
            "Failed to list AlpacaEval dataset files. "
            "Pass --data_file with a local path."
        ) from exc

    candidates = [
        file
        for file in files
        if file.lower().endswith((".json", ".jsonl", ".parquet"))
    ]
    if not candidates:
        raise RuntimeError(
            "No JSON/Parquet files found in the AlpacaEval dataset repo."
        )

    preferred = []
    for file in candidates:
        lowered = file.lower()
        if "alpaca_eval" in lowered and "annotation" not in lowered and "leaderboard" not in lowered:
            preferred.append(file)

    return sorted(preferred or candidates)[0]


def load_alpacaeval_dataset(
    repo_id: str = "tatsu-lab/alpaca_eval",
    data_file: str | None = None,
) -> List[dict]:
    """Load the AlpacaEval evaluation data without dataset scripts."""
    resolved_file = data_file or os.getenv("ALPACAEVAL_DATA_FILE")
    if resolved_file is None:
        filename = _select_dataset_file(repo_id)
        resolved_file = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename=filename
        )
    return _load_dataset_from_file(resolved_file)


def _format_prompt(
    tokenizer: AutoTokenizer, instruction: str, apply_chat_template: bool
) -> str:
    """Format the prompt according to the tokenizer's chat template if available."""
    if (
        apply_chat_template
        and getattr(tokenizer, "apply_chat_template", None)
        and tokenizer.chat_template
    ):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if not apply_chat_template:
        bos_token = tokenizer.bos_token or ""
        return f"{bos_token}{instruction}"
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _resolve_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batched(iterable: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _lookup_token_id(tokenizer: AutoTokenizer, token: str) -> int | None:
    vocab = tokenizer.get_vocab()
    if token in vocab:
        return vocab[token]
    added_vocab = tokenizer.get_added_vocab()
    if token in added_vocab:
        return added_vocab[token]
    if token in tokenizer.all_special_tokens:
        return tokenizer.convert_tokens_to_ids(token)
    return None


def _resolve_eos_token_id(tokenizer: AutoTokenizer) -> int | list[int] | None:
    eos_token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
    elif tokenizer.eos_token:
        token_id = _lookup_token_id(tokenizer, tokenizer.eos_token)
        if token_id is not None:
            eos_token_ids.append(token_id)

    special_map = tokenizer.special_tokens_map or {}
    eot_tokens: list[str] = []
    if "eot_token" in special_map:
        eot_tokens.append(special_map["eot_token"])
    additional = special_map.get("additional_special_tokens") or []
    for token in additional:
        lowered = token.lower()
        if "eot" in lowered or "end_of_turn" in lowered:
            eot_tokens.append(token)
    for token in tokenizer.all_special_tokens:
        lowered = token.lower()
        if "eot" in lowered or "end_of_turn" in lowered:
            eot_tokens.append(token)

    for token in dict.fromkeys(eot_tokens):
        token_id = _lookup_token_id(tokenizer, token)
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    if not eot_tokens:
        fallback_token = "<|eot_id|>"
        token_id = _lookup_token_id(tokenizer, fallback_token)
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    if not eos_token_ids:
        return None
    if len(eos_token_ids) == 1:
        return eos_token_ids[0]
    return eos_token_ids


def generate_model_outputs(
    model_name: str,
    output_file: str,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_input_tokens: int = 2048,
    max_instances: int | None = None,
    seed: int | None = 42,
    dataset_repo: str = "tatsu-lab/alpaca_eval",
    data_file: str | None = None,
    apply_chat_template: bool = True,
) -> None:
    """Generate outputs for AlpacaEval prompts using the specified model."""
    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(resolved_device)

    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = _resolve_eos_token_id(tokenizer)

    model_kwargs = {
        "torch_dtype": dtype,
    }
    if resolved_device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()

    dataset = load_alpacaeval_dataset(repo_id=dataset_repo, data_file=data_file)
    if max_instances is not None:
        dataset = dataset[:max_instances]

    outputs = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    do_sample = temperature > 0

    progress = tqdm(total=len(dataset), desc="Generating", unit="examples")
    for batch in _batched(dataset, batch_size):
        instructions = [item["instruction"] for item in batch]
        prompts = [
            _format_prompt(tokenizer, inst, apply_chat_template)
            for inst in instructions
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        # With left padding, slice off the full padded prompt to avoid leaking it.
        padded_input_length = (
            inputs["input_ids"].shape[1]
            if tokenizer.padding_side == "left"
            else None
        )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)

        for idx, item in enumerate(batch):
            if padded_input_length is not None:
                prompt_length = padded_input_length
            else:
                prompt_length = input_lengths[idx]
            output_ids = generated[idx][prompt_length:]
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            outputs.append(
                {
                    "instruction": item["instruction"],
                    "output": text,
                    "generator": model_name,
                }
            )
        progress.update(len(batch))

    progress.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} outputs to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AlpacaEval outputs")
    parser.add_argument(
        "--model_name",
        type=str,
        default="W-61/hh-llama32-1b-sft",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test/alpacaeval/outputs/model_outputs.json",
        help="Path to save the outputs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=2048,
        help="Maximum input token length",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Limit the number of AlpacaEval prompts",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="tatsu-lab/alpaca_eval",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Local dataset file path (JSON/JSONL/Parquet)",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Whether to apply the tokenizer chat template (true/false).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )

    args = parser.parse_args()
    generate_model_outputs(
        model_name=args.model_name,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_input_tokens=args.max_input_tokens,
        max_instances=args.max_instances,
        seed=args.seed,
        dataset_repo=args.dataset_repo,
        data_file=args.data_file,
        apply_chat_template=args.apply_chat_template,
    )


if __name__ == "__main__":
    main()

"""
Generate summaries for evaluation from multiple models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(data: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(name: str | None, device: str) -> torch.dtype:
    if name:
        name = name.lower()
        if name in {"float16", "fp16"}:
            return torch.float16
        if name in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if name in {"float32", "fp32"}:
            return torch.float32
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def _normalize_model_config(
    key: str, value: str | dict[str, Any], default_model_type: str
) -> dict[str, Any]:
    if isinstance(value, str):
        return {"name": key, "path": value, "model_type": default_model_type}
    if isinstance(value, dict):
        merged = {"name": key, **value}
        if "path" not in merged:
            raise ValueError(f"Model config for '{key}' missing 'path'.")
        merged.setdefault("model_type", default_model_type)
        return merged
    raise ValueError(f"Unexpected model config for '{key}': {value}")


def _load_model_and_tokenizer(
    model_cfg: dict[str, Any], generation_cfg: dict[str, Any]
):
    model_type = model_cfg.get("model_type", "causal")
    device = generation_cfg.get("device", _default_device())
    dtype = _resolve_dtype(generation_cfg.get("torch_dtype"), device)
    revision = model_cfg.get("revision")
    device_map = generation_cfg.get("device_map")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["path"],
        revision=revision,
        use_fast=generation_cfg.get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = (
        AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        model_cfg["path"],
        revision=revision,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if device_map is None:
        model.to(device)
    model.eval()
    return model, tokenizer


def _build_prompt(post: str, template: str | None) -> str:
    if not template:
        return post
    return template.format(post=post)


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    generation_cfg: dict[str, Any],
    model_type: str,
) -> list[str]:
    max_input = generation_cfg.get("max_input_tokens")
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    if max_input:
        tokenizer_kwargs["max_length"] = max_input
    inputs = tokenizer(prompts, **tokenizer_kwargs)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    max_new_tokens = generation_cfg.get("max_new_tokens")
    if max_new_tokens is None:
        max_new_tokens = generation_cfg.get("max_length", 128)

    do_sample = generation_cfg.get("do_sample")
    if do_sample is None:
        do_sample = generation_cfg.get("temperature", 0.0) > 0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": generation_cfg.get("temperature", 0.0),
        "top_p": generation_cfg.get("top_p", 1.0),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    summaries: list[str] = []
    if model_type == "seq2seq":
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for output, input_len in zip(outputs, input_lengths):
            generated = output[int(input_len) :]
            summaries.append(tokenizer.decode(generated, skip_special_tokens=True))

    return [summary.strip() for summary in summaries]


def _is_degenerate(summary: str) -> bool:
    if not summary or not summary.strip():
        return True
    tokens = summary.split()
    if len(tokens) < 3:
        return True
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    return unique_ratio < 0.25


def _merge_existing(
    base_records: list[dict[str, Any]],
    existing: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not existing:
        return base_records

    existing_by_id = {row.get("post_id"): row for row in existing}
    merged: list[dict[str, Any]] = []
    for record in base_records:
        existing_row = existing_by_id.get(record.get("post_id"))
        if existing_row:
            existing_row.update(record)
            merged.append(existing_row)
        else:
            merged.append(record)
    return merged


def generate_all_summaries(
    eval_data_path: str,
    model_configs: dict[str, Any],
    output_path: str,
    generation_cfg: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> None:
    eval_data = _load_json(eval_data_path)
    base_records = [
        {
            "post_id": row.get("post_id"),
            "post": row.get("post"),
            "reference_summary": row.get("reference_summary"),
            "metadata": row.get("metadata"),
        }
        for row in eval_data
    ]

    existing = None
    if resume and Path(output_path).exists():
        existing = _load_json(output_path)

    records = _merge_existing(base_records, existing)

    default_model_type = generation_cfg.get("model_type", "causal")
    batch_size = generation_cfg.get("batch_size", 8)
    prompt_template = generation_cfg.get("prompt_template")

    for key, raw_cfg in model_configs.items():
        model_cfg = _normalize_model_config(key, raw_cfg, default_model_type)
        summary_key = f"{key}_summary"
        indices = [
            idx
            for idx, record in enumerate(records)
            if overwrite or not record.get(summary_key)
        ]
        if not indices:
            print(f"Skipping {key}: summaries already present.")
            continue

        print(f"Loading model '{key}' from {model_cfg['path']}")
        model, tokenizer = _load_model_and_tokenizer(model_cfg, generation_cfg)
        model_type = model_cfg.get("model_type", default_model_type)

        degenerate_count = 0
        for start in tqdm(range(0, len(indices), batch_size), desc=f"{key} batches"):
            batch_indices = indices[start : start + batch_size]
            prompts = [
                _build_prompt(records[idx]["post"], prompt_template)
                for idx in batch_indices
            ]
            summaries = _generate_batch(
                model,
                tokenizer,
                prompts,
                generation_cfg,
                model_type,
            )
            for idx, summary in zip(batch_indices, summaries):
                if _is_degenerate(summary):
                    degenerate_count += 1
                records[idx][summary_key] = summary

            _save_json(records, output_path)

        print(f"Finished {key}. Degenerate summaries: {degenerate_count}")

        if generation_cfg.get("device_map") is None:
            model.to("cpu")
        del model
        torch.cuda.empty_cache()

    print(f"Saved summaries to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summaries for model judge")
    parser.add_argument(
        "--config",
        type=str,
        default="test/gpt_judge_TLDR/config_evaluation.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output JSON path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation and keep existing summaries",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing summaries",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    generation_cfg = config.get("generation", {})
    model_configs = config.get("models", {})
    output_cfg = config.get("output", {})
    output_path = args.output or str(
        Path(output_cfg.get("summaries_dir", "test/gpt_judge_TLDR/outputs"))
        / "all_summaries.json"
    )

    generate_all_summaries(
        eval_data_path=args.eval_data,
        model_configs=model_configs,
        output_path=output_path,
        generation_cfg=generation_cfg,
        resume=args.resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

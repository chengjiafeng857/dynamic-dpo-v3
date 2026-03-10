"""Inference pipeline for AlpacaEval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.loader import resolve_torch_dtype
from .alpacaeval_common import (
    DEFAULT_ALPACA_EVAL_CONFIG,
    DEFAULT_ALPACA_EVAL_DATASET,
    DEFAULT_ALPACA_EVAL_SPLIT,
    get_alpacaeval_config,
    get_generation_config,
    get_model_name_or_path,
    get_output_dir,
    get_package_versions,
    get_pretty_name,
    load_prompt_template,
    render_prompt,
    use_custom_chat_template,
    write_json,
)


def _summarize_text(text: str, *, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _iter_batches(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start_idx in range(0, len(items), batch_size):
        yield items[start_idx : start_idx + batch_size]


def _normalize_vllm_dtype(config: Dict[str, Any]) -> str | None:
    torch_dtype = resolve_torch_dtype(config.get("precision", "fp32"))
    if torch_dtype is torch.float16:
        return "float16"
    if torch_dtype is torch.bfloat16:
        return "bfloat16"
    if torch_dtype is torch.float32:
        return "float32"
    return None


def _generate_with_vllm(
    config: Dict[str, Any],
    prompts: Sequence[str],
    generation_cfg: Dict[str, Any],
) -> list[str]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "alpacaeval.backend=vllm requires the optional vllm package."
        ) from exc

    alpacaeval_cfg = get_alpacaeval_config(config)
    vllm_cfg = alpacaeval_cfg.get("vllm", {})
    if not isinstance(vllm_cfg, dict):
        raise ValueError("alpacaeval.vllm must be a mapping.")

    llm_kwargs: Dict[str, Any] = {
        "model": get_model_name_or_path(config),
        "tensor_parallel_size": int(vllm_cfg.get("tensor_parallel_size", 1)),
        "trust_remote_code": bool(vllm_cfg.get("trust_remote_code", False)),
    }
    gpu_memory_utilization = vllm_cfg.get("gpu_memory_utilization")
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)

    dtype_name = _normalize_vllm_dtype(config)
    if dtype_name is not None:
        llm_kwargs["dtype"] = dtype_name

    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": int(generation_cfg.get("max_new_tokens", 1024)),
        "temperature": (
            float(generation_cfg.get("temperature", 0.9))
            if bool(generation_cfg.get("do_sample", False))
            else 0.0
        ),
        "top_p": float(generation_cfg.get("top_p", 1.0)),
    }
    stop_token_ids = generation_cfg.get("stop_token_ids")
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = [int(token_id) for token_id in stop_token_ids]

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, SamplingParams(**sampling_kwargs))
    return [
        result.outputs[0].text if result.outputs else ""
        for result in outputs
    ]


def _generate_with_transformers(
    config: Dict[str, Any],
    prompts: Sequence[str],
    generation_cfg: Dict[str, Any],
) -> list[str]:
    alpacaeval_cfg = get_alpacaeval_config(config)
    transformers_cfg = alpacaeval_cfg.get("transformers", {})
    if not isinstance(transformers_cfg, dict):
        raise ValueError("alpacaeval.transformers must be a mapping.")

    model_name_or_path = get_model_name_or_path(config)
    trust_remote_code = bool(transformers_cfg.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    torch_dtype = resolve_torch_dtype(config.get("precision", "fp32"))
    if torch.cuda.is_available() and torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    device_map = transformers_cfg.get("device_map")
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    device = str(transformers_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("alpacaeval.transformers.device requests CUDA but CUDA is unavailable.")
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    batch_size = int(generation_cfg.get("batch_size", 8))
    do_sample = bool(generation_cfg.get("do_sample", False))
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 1024)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = float(generation_cfg.get("temperature", 1.0))
        generate_kwargs["top_p"] = float(generation_cfg.get("top_p", 1.0))
    stop_token_ids = generation_cfg.get("stop_token_ids")
    if stop_token_ids:
        eos_token_id = [int(token_id) for token_id in stop_token_ids]
        generate_kwargs["eos_token_id"] = eos_token_id if len(eos_token_id) > 1 else eos_token_id[0]

    outputs: list[str] = []
    add_special_tokens = use_custom_chat_template(config)
    for prompt_batch in _iter_batches(prompts, batch_size):
        tokenized = tokenizer(
            list(prompt_batch),
            add_special_tokens=add_special_tokens,
            padding=True,
            return_tensors="pt",
        )
        tokenized = {name: value.to(model.device) for name, value in tokenized.items()}
        with torch.inference_mode():
            generated = model.generate(**tokenized, **generate_kwargs)
        prompt_length = tokenized["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )
        outputs.extend(decoded)

    return outputs


def _render_prompts(
    config: Dict[str, Any],
    instructions: Sequence[str],
) -> tuple[list[str], str]:
    if use_custom_chat_template(config):
        template_path, template_text = load_prompt_template(config)
        prompts = [render_prompt(template_text, instruction) for instruction in instructions]
        return prompts, str(template_path)

    alpacaeval_cfg = get_alpacaeval_config(config)
    transformers_cfg = alpacaeval_cfg.get("transformers", {})
    if not isinstance(transformers_cfg, dict):
        raise ValueError("alpacaeval.transformers must be a mapping.")

    tokenizer = AutoTokenizer.from_pretrained(
        get_model_name_or_path(config),
        use_fast=True,
        trust_remote_code=bool(transformers_cfg.get("trust_remote_code", False)),
    )
    if (
        getattr(tokenizer, "chat_template", None) is None
        and getattr(tokenizer, "default_chat_template", None) is None
    ):
        raise ValueError(
            "Model tokenizer does not define a default chat template. "
            "Set alpacaeval.use_custom_chat_template=true and provide alpacaeval.prompt_template."
        )

    prompts = [
        str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        for instruction in instructions
    ]
    return prompts, "model_default"


def run_alpacaeval_inference(config: Dict[str, Any]) -> Path:
    alpacaeval_cfg = get_alpacaeval_config(config)
    dataset_name = str(alpacaeval_cfg.get("dataset_name", DEFAULT_ALPACA_EVAL_DATASET))
    dataset_config = str(alpacaeval_cfg.get("dataset_config", DEFAULT_ALPACA_EVAL_CONFIG))
    dataset_split = str(alpacaeval_cfg.get("dataset_split", DEFAULT_ALPACA_EVAL_SPLIT))
    backend = str(alpacaeval_cfg.get("backend", "transformers")).lower()
    max_instances = alpacaeval_cfg.get("max_instances")

    dataset = load_dataset(dataset_name, name=dataset_config, split=dataset_split)
    if max_instances is not None:
        dataset = dataset.select(range(min(len(dataset), int(max_instances))))

    instructions: list[str] = []
    outputs_payload: list[Dict[str, Any]] = []
    generator_name = get_pretty_name(config)

    for row in dataset:
        sample = dict(row)
        instruction = str(sample.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("Each AlpacaEval sample must contain a non-empty instruction.")
        instructions.append(instruction)
        sample["generator"] = generator_name
        outputs_payload.append(sample)

    if not instructions:
        raise ValueError("AlpacaEval dataset is empty after filtering.")

    prompts, prompt_template_ref = _render_prompts(config, instructions)

    print(f"[AlpacaEval] dataset={dataset_name} config={dataset_config} split={dataset_split}")
    print(f"[AlpacaEval] num_examples={len(prompts)} backend={backend}")
    print(f"[AlpacaEval] sample_instruction={_summarize_text(str(outputs_payload[0]['instruction']))}")
    print(f"[AlpacaEval] sample_prompt={_summarize_text(prompts[0])}")

    generation_cfg = get_generation_config(config)
    if backend == "vllm":
        generations = _generate_with_vllm(config, prompts, generation_cfg)
    elif backend == "transformers":
        generations = _generate_with_transformers(config, prompts, generation_cfg)
    else:
        raise ValueError("alpacaeval.backend must be 'vllm' or 'transformers'.")

    for sample, output_text in zip(outputs_payload, generations, strict=True):
        sample["output"] = output_text

    output_dir = get_output_dir(config)
    model_outputs_path = output_dir / "model_outputs.json"
    metadata_path = output_dir / "metadata.json"

    write_json(model_outputs_path, outputs_payload)
    write_json(
        metadata_path,
        {
            "model_name_or_path": get_model_name_or_path(config),
            "pretty_name": generator_name,
            "backend": backend,
            "prompt_template": prompt_template_ref,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "generation": generation_cfg,
            "simpo_compat": bool(alpacaeval_cfg.get("simpo_compat", False)),
            "use_custom_chat_template": use_custom_chat_template(config),
            "package_versions": get_package_versions(),
        },
    )

    print(f"[AlpacaEval] wrote_model_outputs={model_outputs_path}")
    print(f"[AlpacaEval] wrote_metadata={metadata_path}")
    return model_outputs_path

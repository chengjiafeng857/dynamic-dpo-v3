"""
Generate HH single-turn outputs with either transformers or vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import yaml

TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_config_value(cli_value: Any, config_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def _strip_one_leading_newline(text: str) -> str:
    return text[1:] if text.startswith("\n") else text


def _parse_hh_to_messages(text: str) -> list[dict]:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("\n\nHuman:") and not text.startswith("\n\nAssistant:"):
        text = "\n\n" + text

    parts = TAG_RE.split(text)
    messages = []
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        content = _strip_one_leading_newline(content).strip()
        if not content:
            continue
        role = "user" if role_tag == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def _extract_single_turn_instruction(text: str) -> str | None:
    messages = _parse_hh_to_messages(text)
    if len(messages) != 2:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    return messages[0]["content"]


def _extract_single_turn_pair(text: str) -> tuple[str, str] | None:
    messages = _parse_hh_to_messages(text)
    if len(messages) != 2:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    return messages[0]["content"], messages[1]["content"]


def load_hh_single_turn_instructions(
    repo_id: str = "Anthropic/hh-rlhf",
    split: str = "test",
    data_dir: str | None = None,
) -> List[str]:
    dataset = load_dataset(repo_id, data_dir=data_dir, split=split)
    instructions: list[str] = []
    for row in dataset:
        text = row.get("chosen") or row.get("prompt") or row.get("text")
        if text is None:
            continue
        instruction = _extract_single_turn_instruction(text)
        if instruction is None:
            continue
        instructions.append(instruction)
    return instructions


def load_hh_single_turn_chosen_outputs(
    repo_id: str = "Anthropic/hh-rlhf",
    split: str = "test",
    data_dir: str | None = None,
) -> list[dict[str, str]]:
    dataset = load_dataset(repo_id, data_dir=data_dir, split=split)
    rows: list[dict[str, str]] = []
    for row in dataset:
        chosen = row.get("chosen")
        if chosen is None:
            continue
        pair = _extract_single_turn_pair(chosen)
        if pair is None:
            continue
        instruction, output = pair
        rows.append(
            {
                "instruction": instruction,
                "output": output,
                "generator": f"{repo_id}:{split}:chosen",
            }
        )
    return rows


def _format_prompt(
    tokenizer: AutoTokenizer, instruction: str, apply_chat_template: bool
) -> str:
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


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
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


def _generate_outputs_with_vllm(
    model_name: str,
    prompts: list[str],
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float | None,
    trust_remote_code: bool,
) -> list[str]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "backend=vllm requires the optional vllm package."
        ) from exc

    llm_kwargs = {
        "model": model_name,
        "tokenizer": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if seed is not None:
        llm_kwargs["seed"] = seed
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    llm = LLM(**llm_kwargs)
    stop_token_ids = _resolve_eos_token_id(tokenizer)
    sampling_kwargs = {
        "max_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else 0.0,
        "top_p": top_p if temperature > 0 else 1.0,
    }
    if stop_token_ids is not None:
        if isinstance(stop_token_ids, int):
            sampling_kwargs["stop_token_ids"] = [stop_token_ids]
        else:
            sampling_kwargs["stop_token_ids"] = list(stop_token_ids)

    sampling_params = SamplingParams(**sampling_kwargs)
    outputs = llm.generate(prompts, sampling_params)
    return [
        result.outputs[0].text.strip() if result.outputs else ""
        for result in outputs
    ]


def _generate_outputs_with_transformers(
    model_name: str,
    prompts: list[str],
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    batch_size: int,
    device: str,
    temperature: float,
    top_p: float,
    max_input_tokens: int,
) -> list[str]:
    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(resolved_device)
    eos_token_id = _resolve_eos_token_id(tokenizer)

    model_kwargs = {"torch_dtype": dtype}
    if resolved_device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()

    do_sample = temperature > 0
    outputs: list[str] = []
    for prompt_batch in _batched(prompts, batch_size):
        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
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

        for idx in range(len(prompt_batch)):
            if padded_input_length is not None:
                prompt_length = padded_input_length
            else:
                prompt_length = input_lengths[idx]
            output_ids = generated[idx][prompt_length:]
            outputs.append(
                tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            )

    return outputs


def generate_model_outputs(
    model_name: str,
    output_file: str,
    max_new_tokens: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_input_tokens: int = 2048,
    max_instances: int | None = None,
    seed: int | None = 42,
    dataset_repo: str = "Anthropic/hh-rlhf",
    dataset_split: str = "test",
    dataset_data_dir: str | None = None,
    apply_chat_template: bool = True,
    extract_chosen: bool = False,
    backend: str = "transformers",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float | None = None,
    trust_remote_code: bool = False,
) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if extract_chosen:
        outputs = load_hh_single_turn_chosen_outputs(
            repo_id=dataset_repo,
            split=dataset_split,
            data_dir=dataset_data_dir,
        )
        if max_instances is not None:
            outputs = outputs[:max_instances]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(outputs)} chosen outputs to {output_file}")
        return

    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    instructions = load_hh_single_turn_instructions(
        repo_id=dataset_repo,
        split=dataset_split,
        data_dir=dataset_data_dir,
    )
    if max_instances is not None:
        instructions = instructions[:max_instances]

    prompts = [
        _format_prompt(tokenizer, instruction, apply_chat_template)
        for instruction in instructions
    ]
    if prompts:
        print("[HH-EVAL] Sample formatted prompt:")
        print(prompts[0])
        print("[HH-EVAL] End sample formatted prompt")

    if backend == "vllm":
        if device != "cuda":
            raise ValueError("backend=vllm currently requires --device cuda.")
        generated_texts = _generate_outputs_with_vllm(
            model_name=model_name,
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )
    else:
        generated_texts = _generate_outputs_with_transformers(
            model_name=model_name,
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_input_tokens=max_input_tokens,
        )

    outputs = [
        {
            "instruction": instruction,
            "output": text,
            "generator": model_name,
        }
        for instruction, text in zip(instructions, generated_texts)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} outputs to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate outputs for HH single-turn prompts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="eval/gpt_judge_HH/config_eval_HH.yaml",
        help="Path to HH evaluation config YAML.",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="Model key under config.models to generate.",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default=None,
        help="Output key under config.inputs to write.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the outputs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=None,
        help="Maximum input token length",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Limit the number of HH prompts after filtering",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="Dataset split to use",
    )
    parser.add_argument(
        "--dataset_data_dir",
        "--data_dir",
        dest="dataset_data_dir",
        type=str,
        default=None,
        help="HH dataset sub-directory, for example helpful-base or harmless-base.",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=_parse_bool_arg,
        default=None,
        help="Whether to apply the tokenizer chat template (true/false).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--extract_chosen",
        action="store_true",
        default=None,
        help="Write HH chosen responses as outputs instead of running model generation.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["transformers", "vllm"],
        help="Generation backend to use.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="Optional vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=None,
        help="Allow custom model/tokenizer code when loading backends.",
    )

    args = parser.parse_args()
    config = _load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    generation_cfg = config.get("generation", {})
    models_cfg = config.get("models", {})
    inputs_cfg = config.get("inputs", {})

    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a mapping.")
    if not isinstance(generation_cfg, dict):
        raise ValueError("generation config must be a mapping.")
    if not isinstance(models_cfg, dict):
        raise ValueError("models config must be a mapping.")
    if not isinstance(inputs_cfg, dict):
        raise ValueError("inputs config must be a mapping.")

    extract_chosen = _resolve_config_value(
        args.extract_chosen,
        generation_cfg.get("extract_chosen"),
        False,
    )
    default_model_key = "chosen" if extract_chosen else "sft"
    model_key = _resolve_config_value(
        args.model_key,
        generation_cfg.get("model_key"),
        default_model_key,
    )
    output_key = _resolve_config_value(
        args.output_key,
        generation_cfg.get("output_key"),
        model_key,
    )

    model_name = _resolve_config_value(args.model_name, models_cfg.get(model_key), None)
    if not extract_chosen and not model_name:
        raise ValueError(
            f"No model configured for model_key={model_key!r}. "
            "Set config.models.<key> or pass --model_name."
        )

    output_file = _resolve_config_value(args.output_file, inputs_cfg.get(output_key), None)
    if not output_file:
        raise ValueError(
            f"No output file configured for output_key={output_key!r}. "
            "Set config.inputs.<key> or pass --output_file."
        )

    generate_model_outputs(
        model_name=model_name or "Anthropic/hh-rlhf",
        output_file=output_file,
        max_new_tokens=_resolve_config_value(args.max_new_tokens, generation_cfg.get("max_new_tokens"), 512),
        batch_size=_resolve_config_value(args.batch_size, generation_cfg.get("batch_size"), 1),
        device=_resolve_config_value(
            args.device,
            generation_cfg.get("device"),
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
        temperature=_resolve_config_value(args.temperature, generation_cfg.get("temperature"), 0.7),
        top_p=_resolve_config_value(args.top_p, generation_cfg.get("top_p"), 0.9),
        max_input_tokens=_resolve_config_value(
            args.max_input_tokens,
            generation_cfg.get("max_input_tokens"),
            2048,
        ),
        max_instances=_resolve_config_value(
            args.max_instances,
            dataset_cfg.get("max_instances"),
            None,
        ),
        seed=_resolve_config_value(args.seed, generation_cfg.get("seed"), 42),
        dataset_repo=_resolve_config_value(
            args.dataset_repo,
            dataset_cfg.get("repo_id") or dataset_cfg.get("name"),
            "Anthropic/hh-rlhf",
        ),
        dataset_split=_resolve_config_value(args.dataset_split, dataset_cfg.get("split"), "test"),
        dataset_data_dir=_resolve_config_value(
            args.dataset_data_dir,
            dataset_cfg.get("data_dir") or dataset_cfg.get("config_name"),
            None,
        ),
        apply_chat_template=_resolve_config_value(
            args.apply_chat_template,
            generation_cfg.get("apply_chat_template"),
            True,
        ),
        extract_chosen=extract_chosen,
        backend=_resolve_config_value(args.backend, generation_cfg.get("backend"), "transformers"),
        tensor_parallel_size=_resolve_config_value(
            args.tensor_parallel_size,
            generation_cfg.get("tensor_parallel_size"),
            1,
        ),
        gpu_memory_utilization=_resolve_config_value(
            args.gpu_memory_utilization,
            generation_cfg.get("gpu_memory_utilization"),
            None,
        ),
        trust_remote_code=_resolve_config_value(
            args.trust_remote_code,
            generation_cfg.get("trust_remote_code"),
            False,
        ),
    )


if __name__ == "__main__":
    main()

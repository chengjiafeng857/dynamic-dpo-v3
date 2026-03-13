"""
Generate HH single-turn outputs with either transformers or vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, List

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import yaml
from eval.gpt_judge_HH.data_utils import (
    load_hh_chosen_outputs,
    load_hh_eval_examples,
    render_hh_prompt,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    """Load the HH eval YAML config from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_bool_arg(value: str) -> bool:
    """Parse a CLI boolean flag from common true/false strings."""
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_config_value(cli_value: Any, config_value: Any, default_value: Any) -> Any:
    """Resolve a setting with CLI overrides taking priority over config and defaults."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def _configure_vllm_runtime() -> None:
    """Force spawn-based multiprocessing for vLLM worker processes."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    current_method = mp.get_start_method(allow_none=True)
    if current_method != "spawn":
        mp.set_start_method("spawn", force=True)


def _sanitize_name(value: str) -> str:
    """Convert a model id into a filesystem-safe cache directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "tokenizer"


def _prepare_tokenizer_path(model_name_or_path: str) -> str:
    """Return a tokenizer path, sanitizing incompatible tokenizer configs if needed."""
    tokenizer_config_path: Path | None = None
    local_path = Path(model_name_or_path).expanduser()
    tokenizer_source_root: Path | None = None
    if local_path.exists():
        tokenizer_source_root = local_path
        candidate = local_path / "tokenizer_config.json"
        if candidate.exists():
            tokenizer_config_path = candidate
    else:
        try:
            tokenizer_config_path = Path(
                hf_hub_download(model_name_or_path, filename="tokenizer_config.json")
            )
            tokenizer_source_root = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=[
                        "tokenizer*",
                        "chat_template*",
                        "special_tokens_map.json",
                        "added_tokens.json",
                        "vocab.json",
                        "merges.txt",
                        "*.model",
                        "*.tiktoken",
                    ],
                    ignore_patterns=["*.bin", "*.safetensors", "*.pt"],
                )
            )
        except Exception:
            return model_name_or_path

    if (
        tokenizer_config_path is None
        or not tokenizer_config_path.exists()
        or tokenizer_source_root is None
    ):
        return model_name_or_path

    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    needs_sanitization = False
    if isinstance(tokenizer_config.get("extra_special_tokens"), list):
        tokenizer_config["extra_special_tokens"] = {}
        needs_sanitization = True
    if tokenizer_config.get("tokenizer_class") == "TokenizersBackend":
        tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
        needs_sanitization = True

    if not needs_sanitization:
        return model_name_or_path

    cache_root = Path.home() / ".cache" / "dynamic-dpo-v3" / "tokenizers"
    cache_key = hashlib.sha256(model_name_or_path.encode("utf-8")).hexdigest()[:12]
    sanitized_dir = cache_root / f"{_sanitize_name(model_name_or_path)}-{cache_key}"
    sanitized_config_path = sanitized_dir / "tokenizer_config.json"
    if sanitized_config_path.exists():
        return str(sanitized_dir)

    sanitized_dir.mkdir(parents=True, exist_ok=True)
    for source_path in tokenizer_source_root.rglob("*"):
        if not source_path.is_file():
            continue
        target_path = sanitized_dir / source_path.relative_to(tokenizer_source_root)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    sanitized_config_path.write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )
    return str(sanitized_dir)


def _load_tokenizer(
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
) -> Any:
    """Load a tokenizer with repo-specific fallbacks for broken tokenizer metadata."""
    tokenizer_kwargs: dict[str, Any] = {
        "use_fast": True,
        "trust_remote_code": trust_remote_code,
    }
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
            extra_special_tokens={},
        )
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(
            _prepare_tokenizer_path(model_name_or_path),
            **tokenizer_kwargs,
        )


def _resolve_tokenizer_source(
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
) -> str:
    """Return the original tokenizer source unless direct loading fails."""
    tokenizer_kwargs: dict[str, Any] = {
        "use_fast": True,
        "trust_remote_code": trust_remote_code,
    }
    try:
        AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        return model_name_or_path
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
            raise
    return _prepare_tokenizer_path(model_name_or_path)


def _format_prompt(
    tokenizer: AutoTokenizer,
    prompt_messages: list[dict[str, str]],
    apply_chat_template: bool,
) -> str:
    """Render prompt messages into the exact text sent to the generation backend."""
    if apply_chat_template:
        if not getattr(tokenizer, "apply_chat_template", None) or not getattr(
            tokenizer, "chat_template", None
        ):
            raise ValueError(
                "apply_chat_template=True but tokenizer.chat_template is unavailable."
            )
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    rendered_prompt = render_hh_prompt(prompt_messages)
    if not apply_chat_template:
        bos_token = tokenizer.bos_token or ""
        return f"{bos_token}{rendered_prompt}"
    return rendered_prompt


def _resolve_dtype(device: str) -> torch.dtype:
    """Choose a torch dtype for the requested device."""
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _resolve_device(device: str) -> str:
    """Resolve the requested device against the local runtime capabilities."""
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield fixed-size prompt batches while preserving input order."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _lookup_token_id(tokenizer: AutoTokenizer, token: str) -> int | None:
    """Look up a token id across the tokenizer vocab and special tokens, for stop token resolution."""
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
    """Resolve EOS and end-of-turn token ids for generation stopping."""
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
    stop_token_ids: list[int] | None,
) -> list[str]:
    """Generate responses with vLLM for a list of rendered prompts."""
    _configure_vllm_runtime()

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "backend=vllm requires the optional vllm package."
        ) from exc

    llm_kwargs = {
        "model": model_name,
        "tokenizer": _resolve_tokenizer_source(
            model_name,
            trust_remote_code=trust_remote_code,
        ),
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if seed is not None:
        llm_kwargs["seed"] = seed
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    llm = LLM(**llm_kwargs)
    sampling_kwargs = {
        "max_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else 0.0,
        "top_p": top_p if temperature > 0 else 1.0,
    }
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = [int(token_id) for token_id in stop_token_ids]
    else:
        resolved_stop_token_ids = _resolve_eos_token_id(tokenizer)
        if resolved_stop_token_ids is not None:
            if isinstance(resolved_stop_token_ids, int):
                sampling_kwargs["stop_token_ids"] = [resolved_stop_token_ids]
            else:
                sampling_kwargs["stop_token_ids"] = list(resolved_stop_token_ids)

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
    stop_token_ids: list[int] | None,
) -> list[str]:
    """Generate responses with transformers for a list of rendered prompts."""
    resolved_device = _resolve_device(device)
    dtype = _resolve_dtype(resolved_device)

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
        if stop_token_ids:
            eos_token_id = [int(token_id) for token_id in stop_token_ids]
            gen_kwargs["eos_token_id"] = (
                eos_token_id if len(eos_token_id) > 1 else eos_token_id[0]
            )
        else:
            resolved_eos_token_id = _resolve_eos_token_id(tokenizer)
            if resolved_eos_token_id is not None:
                gen_kwargs["eos_token_id"] = resolved_eos_token_id
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
    single_turn_only: bool = True,
    apply_chat_template: bool = True,
    extract_chosen: bool = False,
    backend: str = "transformers",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float | None = None,
    trust_remote_code: bool = False,
    stop_token_ids: list[int] | None = None,
) -> None:
    """Run HH generation or chosen-output export for the configured model and split."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if extract_chosen:
        outputs = load_hh_chosen_outputs(
            repo_id=dataset_repo,
            split=dataset_split,
            data_dir=dataset_data_dir,
            single_turn_only=single_turn_only,
        )
        if max_instances is not None:
            outputs = outputs[:max_instances]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(outputs)} chosen outputs to {output_file}")
        return

    if seed is not None and backend != "vllm":
        set_seed(seed)

    tokenizer = _load_tokenizer(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_hh_eval_examples(
        repo_id=dataset_repo,
        split=dataset_split,
        data_dir=dataset_data_dir,
        single_turn_only=single_turn_only,
    )
    if max_instances is not None:
        examples = examples[:max_instances]

    instructions = [str(example["instruction"]) for example in examples]

    try:
        prompts = [
            _format_prompt(
                tokenizer,
                list(example["prompt_messages"]),
                apply_chat_template,
            )
            for example in examples
        ]
    except ValueError as exc:
        if not apply_chat_template:
            raise
        print(f"[HH-EVAL] {exc} Falling back to raw prompt formatting.")
        prompts = [
            _format_prompt(
                tokenizer,
                list(example["prompt_messages"]),
                False,
            )
            for example in examples
        ]
    if prompts:
        print("[HH-EVAL] Sample formatted prompt:")
        print(prompts[0])
        print("[HH-EVAL] End sample formatted prompt")

    if backend == "vllm":
        _configure_vllm_runtime()
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
            stop_token_ids=stop_token_ids,
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
            stop_token_ids=stop_token_ids,
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
    """Parse CLI/config settings and launch HH output generation."""
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
        single_turn_only=_resolve_config_value(
            None,
            dataset_cfg.get("single_turn_only"),
            True,
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
        stop_token_ids=(
            [int(token_id) for token_id in generation_cfg.get("stop_token_ids")]
            if generation_cfg.get("stop_token_ids")
            else None
        ),
    )


if __name__ == "__main__":
    main()

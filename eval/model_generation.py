"""Shared local generation helpers for benchmark pipelines."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.loader import resolve_torch_dtype

from .benchmark_common import (
    get_block_config,
    get_model_name_or_path,
    load_prompt_template,
    sanitize_name,
    use_custom_chat_template,
)


def _normalize_vllm_dtype(config: Dict[str, Any]) -> str | None:
    torch_dtype = resolve_torch_dtype(config.get("precision", "fp32"))
    if torch_dtype is torch.float16:
        return "float16"
    if torch_dtype is torch.bfloat16:
        return "bfloat16"
    if torch_dtype is torch.float32:
        return "float32"
    return None


def _prepare_tokenizer_path(model_name_or_path: str) -> str:
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
    sanitized_dir = cache_root / f"{sanitize_name(model_name_or_path)}-{cache_key}"
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
    tokenizer_kwargs: Dict[str, Any] = {
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


def load_render_tokenizer(config: Dict[str, Any], block_name: str):
    block_cfg = get_block_config(config, block_name)
    transformers_cfg = block_cfg.get("transformers", {})
    if not isinstance(transformers_cfg, dict):
        raise ValueError(f"{block_name}.transformers must be a mapping.")

    trust_remote_code = bool(transformers_cfg.get("trust_remote_code", False))
    tokenizer = _load_tokenizer(
        get_model_name_or_path(config, block_name),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_chat_prompts(
    config: Dict[str, Any],
    block_name: str,
    *,
    tokenizer: Any,
    conversations: Sequence[Sequence[Dict[str, str]]],
    package_dir: Path,
) -> tuple[list[str], str]:
    if use_custom_chat_template(config, block_name):
        template_path, template_text = load_prompt_template(
            config,
            block_name,
            package_dir=package_dir,
        )
        original_template = getattr(tokenizer, "chat_template", None)
        tokenizer.chat_template = template_text
        try:
            prompts = [
                str(
                    tokenizer.apply_chat_template(
                        list(messages),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                for messages in conversations
            ]
        finally:
            tokenizer.chat_template = original_template
        return prompts, str(template_path)

    if (
        getattr(tokenizer, "chat_template", None) is None
        and getattr(tokenizer, "default_chat_template", None) is None
    ):
        raise ValueError(
            "Model tokenizer does not define a default chat template. "
            f"Set {block_name}.use_custom_chat_template=true and provide "
            f"{block_name}.prompt_template."
        )

    prompts = [
        str(
            tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        for messages in conversations
    ]
    return prompts, "model_default"


def generate_with_vllm(
    config: Dict[str, Any],
    block_name: str,
    prompts: Sequence[str],
    generation_cfg: Dict[str, Any],
) -> list[str]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            f"{block_name}.backend=vllm requires the optional vllm package."
        ) from exc

    block_cfg = get_block_config(config, block_name)
    vllm_cfg = block_cfg.get("vllm", {})
    if not isinstance(vllm_cfg, dict):
        raise ValueError(f"{block_name}.vllm must be a mapping.")

    llm_kwargs: Dict[str, Any] = {
        "model": get_model_name_or_path(config, block_name),
        "tokenizer": _prepare_tokenizer_path(get_model_name_or_path(config, block_name)),
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
    return [result.outputs[0].text if result.outputs else "" for result in outputs]


def generate_with_transformers(
    config: Dict[str, Any],
    block_name: str,
    prompts: Sequence[str],
    generation_cfg: Dict[str, Any],
) -> list[str]:
    block_cfg = get_block_config(config, block_name)
    transformers_cfg = block_cfg.get("transformers", {})
    if not isinstance(transformers_cfg, dict):
        raise ValueError(f"{block_name}.transformers must be a mapping.")

    model_name_or_path = get_model_name_or_path(config, block_name)
    trust_remote_code = bool(transformers_cfg.get("trust_remote_code", False))
    tokenizer = _load_tokenizer(
        model_name_or_path,
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
        raise ValueError(f"{block_name}.transformers.device requests CUDA but CUDA is unavailable.")
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
    for start_idx in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start_idx : start_idx + batch_size])
        tokenized = tokenizer(
            prompt_batch,
            add_special_tokens=False,
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

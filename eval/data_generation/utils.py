from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import LLAMA3_CHAT_TEMPLATE


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_precision(precision: Optional[str]) -> Optional[torch.dtype]:
    if not precision:
        return None
    precision = precision.lower()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def load_tokenizer(
    model_name: str,
    *,
    padding_side: str = "left",
    add_chat_template: bool = True,
    use_fast: bool = True,
) -> AutoTokenizer:
    """Load tokenizer with padding and template defaults for chat models."""
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if add_chat_template and not tok.chat_template:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    return tok


def load_model(
    model_name: str,
    *,
    precision: Optional[str] = None,
    device_map: Optional[str] = None,
) -> AutoModelForCausalLM:
    """Load a causal LM with optional dtype/device_map settings."""
    dtype = _resolve_precision(precision)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

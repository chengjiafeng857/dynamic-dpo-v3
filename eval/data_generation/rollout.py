from __future__ import annotations
import tokenize

import random
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_STOP_STRINGS = ("\n\nHuman:",)
DEFAULT_STOP_TOKENS = ("<|eot_id|>",)


class BaseJudge:
    def rank(self, prompt: str, candidates: List[str]) -> Tuple[int, int]:
        """Return (best_idx, worst_idx) for the given prompt and candidates."""
        raise NotImplementedError


class RMJudge(BaseJudge):
    """Reward-model judge that ranks candidates by score."""

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
        precision: str | None = None,
        device_map: str | None = None,
        load_in_8bit: bool = False,
        batch_size: int = 4,
        seed: int = 42,
        max_length: int | None = None,
    ):
        self._rng = random.Random(seed)
        self.batch_size = int(batch_size)
        self.max_length = max_length

        # ArmoRM custom modeling expects these symbols; newer transformers dropped them.
        self._ensure_llama_docstring()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = None
        if precision and not load_in_8bit:
            prec = precision.lower()
            if prec == "fp16":
                dtype = torch.float16
            elif prec == "bf16":
                dtype = torch.bfloat16

        kwargs = {"trust_remote_code": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes quantization requested but BitsAndBytesConfig is unavailable."
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if device_map is None:
                device_map = "auto"
        if device_map is not None:
            kwargs["device_map"] = device_map

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        except ImportError as exc:
            msg = str(exc)
            if "LLAMA_INPUTS_DOCSTRING" in msg or "LLAMA_START_DOCSTRING" in msg:
                self._ensure_llama_docstring()
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
            else:
                raise
        self.model.eval()

        if device_map is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self._device = device
        else:
            self._device = None

        if self._device is None:
            try:
                self._device = next(self.model.parameters()).device
            except StopIteration:
                self._device = None

    def _build_texts(self, prompt: str | List[dict], candidates: List[str]) -> List[str]:
        texts: List[str] = []
        if isinstance(prompt, list):
            for cand in candidates:
                messages = prompt + [{"role": "assistant", "content": cand}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return texts

        suffix = self.tokenizer.eos_token or ""
        for cand in candidates:
            text = f"{prompt}{cand}"
            if suffix and not text.endswith(suffix):
                text = f"{text}{suffix}"
            texts.append(text)
        return texts

    @staticmethod
    def _ensure_llama_docstring() -> None:
        try:
            from transformers.models.llama import modeling_llama

            if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
                modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
            if not hasattr(modeling_llama, "LLAMA_START_DOCSTRING"):
                modeling_llama.LLAMA_START_DOCSTRING = ""
        except Exception:
            # If llama module moves again, we fall back to letting HF raise.
            pass

    def _score_texts(self, texts: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=self.max_length is not None,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self._device is not None:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_scores = logits.squeeze(-1).detach().float().cpu().tolist()
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        return scores

    def rank(self, prompt: str | List[dict], candidates: List[str]) -> Tuple[int, int]:
        if not candidates:
            raise ValueError("No candidates to rank.")
        texts = self._build_texts(prompt, candidates)
        scores = self._score_texts(texts)
        max_score = max(scores)
        min_score = min(scores)
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        worst_indices = [i for i, s in enumerate(scores) if s == min_score]
        best_idx = self._rng.choice(best_indices)
        worst_idx = self._rng.choice(worst_indices)
        return best_idx, worst_idx

    def rank_batch(
        self,
        prompts: List[str | List[dict]],
        candidates_list: List[List[str]],
    ) -> List[Tuple[int, int]]:
        if len(prompts) != len(candidates_list):
            raise ValueError("prompts and candidates_list must have the same length.")
        if not prompts:
            return []

        texts: List[str] = []
        spans: List[Tuple[int, int]] = []
        for prompt, candidates in zip(prompts, candidates_list):
            if not candidates:
                raise ValueError("No candidates to rank.")
            start = len(texts)
            built = self._build_texts(prompt, candidates)
            texts.extend(built)
            spans.append((start, len(built)))

        scores = self._score_texts(texts)
        results: List[Tuple[int, int]] = []
        for (start, count), candidates in zip(spans, candidates_list):
            sub_scores = scores[start : start + count]
            max_score = max(sub_scores)
            min_score = min(sub_scores)
            best_indices = [i for i, s in enumerate(sub_scores) if s == max_score]
            worst_indices = [i for i, s in enumerate(sub_scores) if s == min_score]
            best_idx = self._rng.choice(best_indices)
            worst_idx = self._rng.choice(worst_indices)
            results.append((best_idx, worst_idx))
        return results


class RolloutGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        num_return_sequences: int,
        stop_strings: Iterable[str] = DEFAULT_STOP_STRINGS,
        stop_tokens: Iterable[str | int] = DEFAULT_STOP_TOKENS,
        **generation_kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_return_sequences = int(num_return_sequences)
        self.stop_strings = tuple(stop_strings)
        self.stop_token_sequences = self._build_stop_sequences(stop_tokens)
        self.generation_kwargs = generation_kwargs

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _truncate_at_stop_strings(self, text: str) -> str:
        stop_positions = [text.find(s) for s in self.stop_strings if s in text]
        if stop_positions:
            text = text[: min(stop_positions)]
        return text.strip()

    def _build_stop_sequences(self, stop_tokens: Iterable[str | int]) -> List[List[int]]:
        sequences: List[List[int]] = []
        for token in stop_tokens:
            if isinstance(token, int):
                sequences.append([token])
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                sequences.append([token_id])
                continue
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                sequences.append(encoded)

        if self.tokenizer.eos_token_id is not None:
            sequences.append([self.tokenizer.eos_token_id])

        unique: List[List[int]] = []
        seen = set()
        for seq in sequences:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            unique.append(seq)
        return unique

    def generate_batch(
        self,
        prompt_texts: List[str],
        *,
        return_raw: bool = False,
        return_token_counts: bool = False,
    ):
        if not prompt_texts:
            if return_raw and return_token_counts:
                return [], [], []
            if return_raw:
                return [], []
            if return_token_counts:
                return [], []
            return []

        encoded = self.tokenizer(prompt_texts, padding=True, return_tensors="pt")
        input_len = encoded["input_ids"].shape[1]
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        eos_ids = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            eos_ids.append(eot_id)
        eos_ids = [eid for eid in eos_ids if eid is not None]
        if len(eos_ids) == 1:
            eos_ids = eos_ids[0]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=self.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                **self.generation_kwargs,
            )

        # Strip the padded prompt portion; generated tokens start after input_len.
        generated_ids = outputs[:, input_len:]
        token_counts = None
        if return_token_counts:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                token_counts = [len(seq) for seq in generated_ids]
            else:
                token_counts = (generated_ids != pad_id).sum(dim=1).tolist()
        raw_decoded = None
        if return_raw:
            raw_decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [self._truncate_at_stop_strings(text) for text in decoded]

        grouped: List[List[str]] = []
        raw_grouped: List[List[str]] = []
        count_grouped: List[List[int]] = []
        for i in range(0, len(decoded), self.num_return_sequences):
            grouped.append(decoded[i : i + self.num_return_sequences])
            if raw_decoded is not None:
                raw_grouped.append(raw_decoded[i : i + self.num_return_sequences])
            if token_counts is not None:
                count_grouped.append(token_counts[i : i + self.num_return_sequences])

        if return_raw and return_token_counts:
            return grouped, raw_grouped, count_grouped
        if return_raw:
            return grouped, raw_grouped
        if return_token_counts:
            return grouped, count_grouped
        return grouped


class VLLMRolloutGenerator:
    """High-throughput generator using vLLM."""

    def __init__(
        self,
        model_name: str,
        seed: int,
        device_map: str | None = None,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Please install it with `pip install vllm` to use this generator."
            ) from exc

        self.llm = LLM(
            model=model_name,
            seed=seed,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            # device_map is handled internally by vLLM (tensor_parallel_size, etc.)
            # We assume single-node for now or standard vllm distributed setup.
            **kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_batch(
        self,
        prompt_texts: List[str],
        *,
        return_raw: bool = False,
        return_token_counts: bool = False,
        **generation_kwargs,
    ):
        if not prompt_texts:
            if return_raw and return_token_counts:
                return [], [], []
            if return_raw:
                return [], []
            if return_token_counts:
                return [], []
            return []

        from vllm import SamplingParams

        # Map HF configs to vLLM SamplingParams
        n = int(generation_kwargs.get("num_return_sequences", 1))
        # vLLM expects float for temp/top_p, etc.
        temperature = float(generation_kwargs.get("temperature", 1.0))
        top_p = float(generation_kwargs.get("top_p", 1.0))
        do_sample = generation_kwargs.get("do_sample", True)
        if not do_sample:
            # Match HF greedy decoding semantics when sampling is disabled.
            temperature = 0.0
            top_p = 1.0
        max_tokens = int(generation_kwargs.get("max_new_tokens", 512))
        min_tokens = generation_kwargs.get("min_new_tokens", 0)
        min_tokens = max(0, int(min_tokens)) if min_tokens is not None else 0
        
        # Stop tokens
        # Match HF: stop on EOS/EOT during generation, then truncate stop strings after decoding.
        stop_strings = generation_kwargs.get("stop_strings", DEFAULT_STOP_STRINGS)
        stop_token_ids = self._resolve_stop_token_ids(generation_kwargs.get("eos_token_id", []))

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            # Stop strings are applied post-decode to mirror HF truncation behavior.
            stop=None,
            stop_token_ids=list(set(stop_token_ids)) if stop_token_ids else None,
        )

        # Generate
        # use_tqdm=False to keep logs clean or let caller handle progress
        outputs = self.llm.generate(prompt_texts, sampling_params, use_tqdm=True)

        grouped: List[List[str]] = []
        raw_grouped: List[List[str]] = []
        count_grouped: List[List[int]] = []
        for request_output in outputs:
            cands: List[str] = []
            raw_cands: List[str] = []
            counts: List[int] = []
            for completion in request_output.outputs:
                token_ids = getattr(completion, "token_ids", None)
                if token_ids:
                    # Decode from token IDs to strip special tokens like HF.
                    raw_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                    decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    token_count = len(token_ids)
                else:
                    raw_text = completion.text
                    decoded = completion.text
                    token_count = len(self.tokenizer.encode(decoded, add_special_tokens=False))
                decoded = self._truncate_at_stop_strings(decoded, stop_strings)
                cands.append(decoded)
                if return_raw:
                    raw_cands.append(raw_text)
                if return_token_counts:
                    counts.append(token_count)
            grouped.append(cands)
            if return_raw:
                raw_grouped.append(raw_cands)
            if return_token_counts:
                count_grouped.append(counts)

        if return_raw and return_token_counts:
            return grouped, raw_grouped, count_grouped
        if return_raw:
            return grouped, raw_grouped
        if return_token_counts:
            return grouped, count_grouped
        return grouped

    @staticmethod
    def _truncate_at_stop_strings(text: str, stop_strings: Iterable[str]) -> str:
        if not stop_strings:
            return text.strip()
        stop_positions = [text.find(s) for s in stop_strings if s in text]
        if stop_positions:
            text = text[: min(stop_positions)]
        return text.strip()
# stop_tokens handling
    def _resolve_stop_token_ids(self, stop_token_ids) -> List[int]:
        resolved: List[int] = []

        for token in DEFAULT_STOP_TOKENS:
            if isinstance(token, int):
                resolved.append(token)
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                resolved.append(token_id)
                continue
        # fallback for custom stop tokens
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) == 1:
                resolved.append(encoded[0])

        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        if stop_token_ids:
            for token_id in stop_token_ids:
                if isinstance(token_id, int):
                    resolved.append(token_id)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            resolved.append(eos_id)
        return list(dict.fromkeys(resolved))

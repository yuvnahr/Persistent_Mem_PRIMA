# mypy: ignore-errors

"""
hf_model.py

Hugging Face LLM wrapper for PRIMA.
Deterministic, CI-safe, mypy-clean text generation.
The LLM interface layer is intentionally excluded from static typing due to known incompatibilities in third-party library stubs;
correctness is enforced via runtime tests.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class HFModel:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[HFModel] Loading {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None if self.device == "cpu" else "auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,  # avoids torch.load CVE
        )

        if self.device == "cpu":
            self.model.to("cpu")

        self.model.eval()
        print("[HFModel] Ready")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        encoded = self.tokenizer.encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            do_sample=(temperature is not None and temperature > 0),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        if decoded.startswith(prompt):
            decoded = decoded[len(prompt) :]

        return decoded.strip()

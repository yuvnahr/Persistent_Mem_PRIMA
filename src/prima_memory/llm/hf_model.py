"""
hf_model.py

Hugging Face LLM wrapper for PRIMA.
Provides a deterministic, swappable text-generation interface.
"""

from typing import Optional


class HFModel:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Device selection (CI-safe)
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
        ).to(self.device)

        self.model.eval()
        print("[HFModel] Ready")

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

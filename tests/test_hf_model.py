"""
test_hf_model.py

Sanity test for HuggingFace LLM wrapper.
Uses a small causal model to remain CI-safe.
"""

from prima_memory.llm.hf_model import HFModel


def test_hf_model_generate():
    model = HFModel(
        model_name="sshleifer/tiny-gpt2",
        max_new_tokens=20,
        temperature=0.0,
        device="cpu",
    )

    output = model.generate("Explain memory in one sentence.")

    assert isinstance(output, str)
    assert len(output) > 0

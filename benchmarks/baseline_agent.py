"""
baseline_agent.py

Baseline agent for LoCoMo evaluation (no memory system).
Adapted from A-mem repository.
"""

import torch

from prima_memory.llm.hf_model import HFModel


class BaselineAgent:
    """
    Baseline agent without memory system.
    Just uses LLM to answer questions directly.
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.llm = HFModel(model_name=model_name, device=device)

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """
        Answer a question without memory context.

        Args:
            question: The question to answer
            category: Question category (1-5)
            answer: Ground truth answer (for adversarial questions)

        Returns:
            Predicted answer
        """
        if category == 1:
            # Multi-hop
            user_prompt = f"""
Based on general knowledge, answer the following question. Provide a short answer.

Question: {question}
Short answer:"""

        elif category == 2:
            # Temporal
            user_prompt = f"""
Based on general knowledge, answer the following question. Use approximate dates where possible.
Provide a short answer.

Question: {question}
Short answer:"""

        elif category == 3:
            # Open-domain
            user_prompt = f"""
Based on general knowledge, answer the following question. Provide a short answer.

Question: {question}
Short answer:"""

        elif category == 4:
            # Single-hop
            user_prompt = f"""
Based on general knowledge, answer the following question. Provide a short answer.

Question: {question}
Short answer:"""

        elif category == 5:
            # Adversarial
            import random

            answer_tmp = []
            if random.random() < 0.5:
                answer_tmp.append("Not mentioned in the conversation")
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append("Not mentioned in the conversation")

            user_prompt = f"""
Based on general knowledge, answer the following question.

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}
Short answer:"""

        else:
            raise ValueError(f"Unknown category: {category}")

        # Get response from LLM
        response = self.llm.generate(user_prompt)

        return response.strip()

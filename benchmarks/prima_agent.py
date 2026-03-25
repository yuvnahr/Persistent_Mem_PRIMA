"""
prima_agent.py

PRIMA agent for LoCoMo evaluation (with memory system).
Adapted from A-mem repository.
"""

from typing import List, Tuple

import torch

from prima_memory.core.agentic_memory_system import AgenticMemorySystem
from prima_memory.llm.hf_model import HFModel


class PrimaAgent:
    """
    PRIMA agent with memory system.
    Uses AgenticMemorySystem to maintain and retrieve memories.
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.llm = HFModel(model_name=model_name, device=device)
        self.memory_system = AgenticMemorySystem(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            llm_model_name=model_name,
            device=device,
        )

    def process_conversation(self, conversation: List[Tuple[str, str]]):
        """
        Process conversation history and add to memory.

        Args:
            conversation: List of (speaker, utterance) tuples
        """
        for speaker, utterance in conversation:
            # Add each utterance as a memory note
            self.memory_system.add_note(
                content=utterance, metadata={"speaker": speaker}
            )

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """
        Answer a question using memory context.

        Args:
            question: The question to answer
            category: Question category (1-5)
            answer: Ground truth answer (for adversarial questions)

        Returns:
            Predicted answer
        """
        # Retrieve relevant memories
        relevant_memories = self.memory_system.search(question, top_k=5)
        memory_context = "\n".join([mem.content for mem in relevant_memories])

        if category == 1:
            # Multi-hop
            user_prompt = f"""
Based on the following conversation history and general knowledge, answer the following question.
Provide a short answer.

Conversation history:
{memory_context}

Question: {question}
Short answer:"""

        elif category == 2:
            # Temporal
            user_prompt = f"""
Based on the following conversation history and general knowledge, answer the following question.
Use approximate dates where possible. Provide a short answer.

Conversation history:
{memory_context}

Question: {question}
Short answer:"""

        elif category == 3:
            # Open-domain
            user_prompt = f"""
Based on the following conversation history and general knowledge, answer the following question.
Provide a short answer.

Conversation history:
{memory_context}

Question: {question}
Short answer:"""

        elif category == 4:
            # Single-hop
            user_prompt = f"""
Based on the following conversation history and general knowledge, answer the following question.
Provide a short answer.

Conversation history:
{memory_context}

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
Based on the following conversation history and general knowledge, answer the following question.

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}

Conversation history:
{memory_context}

Short answer:"""

        else:
            raise ValueError(f"Unknown category: {category}")

        # Get response from LLM
        response = self.llm.generate(user_prompt)

        return response.strip()

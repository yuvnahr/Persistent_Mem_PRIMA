"""
load_dataset.py

Dataset loading utilities for LoCoMo benchmark.
Adapted from A-mem repository.
"""

import json
from dataclasses import dataclass
from typing import List


@dataclass
class Turn:
    """A single turn in a conversation."""

    speaker: str
    text: str
    timestamp: str


@dataclass
class Session:
    """A conversation session."""

    turns: List[Turn]


@dataclass
class QA:
    """A question-answer pair."""

    question: str
    category: int
    final_answer: str


@dataclass
class Conversation:
    """A complete conversation with QA pairs."""

    sessions: List[Session]
    qa: List[QA]


def load_locomo_dataset(dataset_path: str) -> List[Conversation]:
    """
    Load LoCoMo dataset from JSON file.

    Args:
        dataset_path: Path to the dataset JSON file

    Returns:
        List of Conversation objects
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []

    for conv_data in data:
        # Parse sessions
        sessions = []
        for session_data in conv_data.get("sessions", []):
            turns = []
            for turn_data in session_data.get("turns", []):
                turn = Turn(
                    speaker=turn_data["speaker"],
                    text=turn_data["text"],
                    timestamp=turn_data["timestamp"],
                )
                turns.append(turn)
            session = Session(turns=turns)
            sessions.append(session)

        # Parse QA pairs
        qa_pairs = []
        for qa_data in conv_data.get("qa", []):
            qa = QA(
                question=qa_data["question"],
                category=qa_data["category"],
                final_answer=qa_data.get("final_answer", ""),
            )
            qa_pairs.append(qa)

        conversation = Conversation(sessions=sessions, qa=qa_pairs)
        conversations.append(conversation)

    return conversations

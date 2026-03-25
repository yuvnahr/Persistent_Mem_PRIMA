"""
evaluate_benchmark.py

Main evaluation script for LoCoMo benchmark.
Compares baseline vs PRIMA agents.
Adapted from A-mem repository.
"""

import json
from typing import Any, Dict, List

import torch
from baseline_agent import BaselineAgent
from load_dataset import Conversation, load_locomo_dataset
from prima_agent import PrimaAgent
from utils import aggregate_metrics, calculate_metrics


def evaluate_dataset(
    dataset: List[Conversation],
    agent_class,
    model_name: str = "microsoft/DialoGPT-medium",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict[str, Any]]:
    """
    Evaluate an agent on the LoCoMo dataset.

    Args:
        dataset: List of Conversation objects
        agent_class: Agent class to evaluate (BaselineAgent or PrimaAgent)
        model_name: HuggingFace model name
        device: Device to run on

    Returns:
        Dictionary with evaluation results
    """
    agent = agent_class(model_name=model_name, device=device)

    results = []

    for conv_idx, conversation in enumerate(dataset):
        print(f"Processing conversation {conv_idx + 1}/{len(dataset)}")

        # Initialize agent for new conversation
        if hasattr(agent, "memory_system"):
            # Reset memory for PRIMA agent
            agent.memory_system = type(agent.memory_system)(
                llm=agent.llm,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                device=device,
            )

        # Process conversation history
        if hasattr(agent, "process_conversation"):
            # Convert sessions to (speaker, utterance) format
            conv_history = []
            for session in conversation.sessions:
                for turn in session.turns:
                    conv_history.append((turn.speaker, turn.text))
            agent.process_conversation(conv_history)

        # Answer questions
        for qa in conversation.qa:
            try:
                predicted_answer = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )

                result = {
                    "conversation_id": conv_idx,  # Use index as ID
                    "question": qa.question,
                    "ground_truth": qa.final_answer,
                    "prediction": predicted_answer,
                    "category": qa.category,
                }
                results.append(result)

            except Exception as e:
                print(f"Error processing QA: {e}")
                result = {
                    "conversation_id": conv_idx,
                    "question": qa.question,
                    "ground_truth": qa.final_answer,
                    "prediction": "",
                    "category": qa.category,
                }
                results.append(result)

    return results


def main():
    """Main evaluation function."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print("Loading LoCoMo dataset...")
    dataset = load_locomo_dataset("data/locomo10.json")

    # Model name
    model_name = "microsoft/DialoGPT-medium"

    # Evaluate baseline
    print("Evaluating baseline agent...")
    baseline_results = evaluate_dataset(dataset, BaselineAgent, model_name, device)

    # Save baseline results
    with open("experiments_outputs/baseline/baseline_outputs.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Evaluate PRIMA
    print("Evaluating PRIMA agent...")
    prima_results = evaluate_dataset(dataset, PrimaAgent, model_name, device)

    # Save PRIMA results
    with open("experiments_outputs/prima/prima_outputs.json", "w") as f:
        json.dump(prima_results, f, indent=2)

    # Calculate metrics
    print("Calculating metrics...")

    baseline_metrics = calculate_metrics(baseline_results)
    prima_metrics = calculate_metrics(prima_results)

    # Aggregate metrics
    baseline_agg = aggregate_metrics(baseline_metrics)
    prima_agg = aggregate_metrics(prima_metrics)

    # Print results
    print("\n=== BASELINE RESULTS ===")
    print(json.dumps(baseline_agg, indent=2))

    print("\n=== PRIMA RESULTS ===")
    print(json.dumps(prima_agg, indent=2))

    # Save aggregated results
    results_summary = {"baseline": baseline_agg, "prima": prima_agg}

    with open("benchmark_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()

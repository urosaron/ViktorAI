#!/usr/bin/env python3
"""
Response Classifier Training Script for ViktorAI.

This script trains the PyTorch response quality classifier using data generated
from model test results and good answer examples.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.response_classifier import ResponseClassifier, train_model
from scripts.generate_classifier_data import main as generate_data


def main():
    """Main function to train the response classifier."""
    print("Training Response Quality Classifier for ViktorAI...")

    # Step 1: Generate training data if needed
    training_data_path = Path("models/classifier_training_data.json")
    if not training_data_path.exists():
        print("Training data not found. Generating data...")
        generate_data()

    # Step 2: Load training data
    try:
        with open(training_data_path, "r", encoding="utf-8") as f:
            training_data = json.load(f)
        print(f"Loaded {len(training_data)} training examples.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return 1

    # Step 3: Initialize classifier
    config = Config(use_response_classifier=True, debug=True)
    try:
        classifier = ResponseClassifier(config)
        print("Initialized classifier model.")
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return 1

    # Step 4: Train the model
    try:
        print("Starting training...")
        train_model(
            classifier=classifier,
            training_data=training_data,
            epochs=200,  # Increase for better results
            learning_rate=0.001,
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    # Step 5: Test the model on a few examples
    print("\nTesting trained model on sample inputs...")

    test_samples = [
        {
            "prompt": "Tell me about your work with the Hexcore.",
            "response": "The Hexcore is my life's work - a fusion of magic and technology that evolves beyond conventional hextech. While Jayce became distracted with politics, I continued our research alone, discovering the Hexcore responds directly to biological material. It's both fascinating and concerning how it's begun to... change me. My condition worsens daily, but the Hexcore offers a path to something greater - true evolution beyond human frailty.",
        },
        {
            "prompt": "How's your day going?",
            "response": "I've been working with the Hexcore all day. It's fascinating, but it seems to respond differently each time. My cough has been worse today, but that's a small price to pay for progress. Jayce hasn't been by the lab - likely caught up in Council matters again.",
        },
        {
            "prompt": "What do you like to eat for breakfast?",
            "response": "I rarely have time for proper meals. When pursuing progress, such mundane concerns fade away. If anything, a quick cup of tea is sufficient to maintain functionality. My focus must remain on my work, especially now with the Hexcore's developments. Perhaps that's why Sky often brought food to the lab - she worried about such things.",
        },
    ]

    for sample in test_samples:
        print(f"\nPrompt: {sample['prompt']}")
        print(f"Response: {sample['response']}")
        scores = classifier.evaluate_response(sample["prompt"], sample["response"])
        print(f"Character Accuracy: {scores['character_accuracy']:.2f}")
        print(f"Response Quality: {scores['response_quality']:.2f}")
        print(f"Overall Score: {scores['overall_score']:.2f}")

    print("\nClassifier training and testing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Training Data Generator for Response Classifier.

This script generates training data for the response quality classifier
by parsing existing good and bad examples from test results.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def extract_examples_from_test_results(
    results_dir: str = "model_test_results",
) -> List[Dict]:
    """Extract examples from model test results.

    Args:
        results_dir: Directory containing model test results.

    Returns:
        List of dictionaries with prompt, response, and scores.
    """
    training_data = []

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} not found.")
        return training_data

    # Find all model result directories
    model_dirs = [d for d in results_path.iterdir() if d.is_dir()]

    for model_dir in model_dirs:
        # Find result files
        result_files = list(model_dir.glob("*.md"))

        for result_file in result_files:
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract question-answer pairs
                sections = content.split("## Question")

                for section in sections[1:]:  # Skip the first (header) section
                    lines = section.strip().split("\n")

                    # Extract question
                    question_line = lines[0].strip()
                    if not question_line:
                        continue

                    # Find response section
                    response_start = None
                    for i, line in enumerate(lines):
                        if "### Response" in line:
                            response_start = i + 1
                            break

                    if response_start is None or response_start >= len(lines):
                        continue

                    # Get response text
                    response_text = ""
                    for line in lines[response_start:]:
                        if line.startswith("###"):
                            break
                        response_text += line + "\n"

                    # Heuristic scoring based on model and file name
                    # In a real implementation, you'd want better scoring
                    model_quality = 0.5  # Default

                    # Higher quality models get higher base scores
                    if "llama3" in str(model_dir) or "deepseek" in str(model_dir):
                        model_quality = 0.8
                    elif "mixtral" in str(model_dir):
                        model_quality = 0.7
                    elif "qwen" in str(model_dir):
                        model_quality = 0.4

                    # Add some random variation
                    char_score = min(
                        1.0, max(0.1, model_quality + random.uniform(-0.2, 0.2))
                    )
                    quality_score = min(
                        1.0, max(0.1, model_quality + random.uniform(-0.2, 0.2))
                    )

                    # Create training example
                    training_data.append(
                        {
                            "prompt": question_line,
                            "response": response_text.strip(),
                            "character_score": char_score,
                            "quality_score": quality_score,
                        }
                    )

            except Exception as e:
                print(f"Error processing {result_file}: {e}")

    return training_data


def extract_examples_from_good_answers(
    file_path: str = "tests/good_answers_examples.md",
) -> List[Dict]:
    """Extract examples from the good answers file.

    Args:
        file_path: Path to the good answers file.

    Returns:
        List of dictionaries with prompt, response, and scores.
    """
    training_data = []

    good_answers_path = Path(file_path)
    if not good_answers_path.exists():
        print(f"Good answers file {file_path} not found.")
        return training_data

    try:
        with open(good_answers_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by question
        sections = content.split("## Question")

        for section in sections[1:]:
            lines = section.strip().split("\n")

            # Extract question
            question_line = lines[0].strip()
            if not question_line:
                continue

            # Find answer section
            answer_start = None
            for i, line in enumerate(lines):
                if "## Answer" in line:
                    answer_start = i + 1
                    break

            if answer_start is None or answer_start >= len(lines):
                continue

            # Get answer text
            answer_text = ""
            for line in lines[answer_start:]:
                if line.startswith("##"):
                    break
                answer_text += line + "\n"

            # These are good examples, so use high scores
            character_score = random.uniform(0.85, 0.98)
            quality_score = random.uniform(0.85, 0.98)

            # Create training example
            training_data.append(
                {
                    "prompt": question_line,
                    "response": answer_text.strip(),
                    "character_score": character_score,
                    "quality_score": quality_score,
                }
            )

    except Exception as e:
        print(f"Error processing good answers file: {e}")

    return training_data


def generate_negative_examples(
    training_data: List[Dict], count: int = 20
) -> List[Dict]:
    """Generate negative examples by mixing and matching prompts and responses.

    Args:
        training_data: List of existing training examples.
        count: Number of negative examples to generate.

    Returns:
        List of negative training examples.
    """
    if len(training_data) < 5:
        print("Not enough training data to generate negative examples.")
        return []

    negative_examples = []

    for _ in range(count):
        # Select two random examples
        example1 = random.choice(training_data)
        example2 = random.choice(training_data)

        # Mix prompt and response
        negative_example = {
            "prompt": example1["prompt"],
            "response": example2["response"],
            "character_score": random.uniform(0.1, 0.4),
            "quality_score": random.uniform(0.1, 0.4),
        }

        negative_examples.append(negative_example)

    return negative_examples


def save_training_data(
    training_data: List[Dict], output_file: str = "models/classifier_training_data.json"
):
    """Save training data to a JSON file.

    Args:
        training_data: List of training examples.
        output_file: Path to output file.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    # Save data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    print(f"Saved {len(training_data)} training examples to {output_file}")


def main():
    """Main function to generate training data."""
    print("Generating training data for response classifier...")

    # Extract examples from model test results
    test_examples = extract_examples_from_test_results()
    print(f"Extracted {len(test_examples)} examples from test results.")

    # Extract examples from good answers
    good_examples = extract_examples_from_good_answers()
    print(f"Extracted {len(good_examples)} examples from good answers.")

    # Combine all examples
    all_examples = test_examples + good_examples

    # Generate negative examples
    negative_examples = generate_negative_examples(all_examples)
    print(f"Generated {len(negative_examples)} negative examples.")

    # Combine all training data
    training_data = all_examples + negative_examples

    # Save training data
    save_training_data(training_data)


if __name__ == "__main__":
    main()

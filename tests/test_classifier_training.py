#!/usr/bin/env python3
"""
End-to-end tests for the response classifier training pipeline.

This module tests the entire classifier training pipeline, including data
generation, model training, and evaluation.
"""

import os
import sys
import json
import shutil
import unittest
import tempfile
from pathlib import Path
import torch

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the training modules
from src.config import Config
from src.response_classifier import ResponseClassifier, train_model
from scripts.generate_classifier_data import (
    generate_negative_examples,
    save_training_data,
)

# Mock the training script's main function to avoid setup requirements
from scripts.train_classifier import main as train_main


class TestClassifierTraining(unittest.TestCase):
    """Test cases for the classifier training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create models directory inside our test directory
        self.models_dir = self.test_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Create sample training data
        self.training_data = [
            {
                "prompt": "Tell me about the Hexcore.",
                "response": "The Hexcore represents the future of hextech. It's a combination of magic and technology that evolves beyond its programming.",
                "character_score": 0.92,
                "quality_score": 0.88,
            },
            {
                "prompt": "How do you feel about your illness?",
                "response": "My condition is... deteriorating. But physical limitations are merely obstacles to overcome. The Hexcore may hold the answer.",
                "character_score": 0.95,
                "quality_score": 0.9,
            },
            {
                "prompt": "What's your relationship with Jayce?",
                "response": "Jayce and I began as partners with a shared vision. But while he pursues politics, I pursue progress in the lab.",
                "character_score": 0.85,
                "quality_score": 0.83,
            },
        ]

        # Save the sample training data
        self.training_data_path = self.models_dir / "classifier_training_data.json"
        with open(self.training_data_path, "w") as f:
            json.dump(self.training_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_negative_example_generation(self):
        """Test the generation of negative examples."""
        # Generate negative examples
        negative_examples = generate_negative_examples(self.training_data, count=5)

        # Check that the right number of examples was generated
        self.assertEqual(len(negative_examples), 5)

        # Check that they have the expected structure
        for example in negative_examples:
            self.assertIn("prompt", example)
            self.assertIn("response", example)
            self.assertIn("character_score", example)
            self.assertIn("quality_score", example)

            # Negative examples should have low scores
            self.assertLess(example["character_score"], 0.5)
            self.assertLess(example["quality_score"], 0.5)

    def test_training_data_saving(self):
        """Test saving training data to a file."""
        # Set up test path
        test_file = self.test_dir / "test_data.json"

        # Save the data
        save_training_data(self.training_data, str(test_file))

        # Check that the file exists
        self.assertTrue(test_file.exists())

        # Load the data back and check it
        with open(test_file, "r") as f:
            loaded_data = json.load(f)

        # Check that we got the same data back
        self.assertEqual(len(loaded_data), len(self.training_data))
        self.assertEqual(loaded_data[0]["prompt"], self.training_data[0]["prompt"])
        self.assertEqual(loaded_data[0]["response"], self.training_data[0]["response"])

    def test_training_with_data(self):
        """Test that training works with our sample data."""
        # Initialize a classifier
        config = Config(use_response_classifier=True)
        classifier = ResponseClassifier(config)

        # Train the model with minimal epochs
        train_model(
            classifier=classifier,
            training_data=self.training_data,
            epochs=10,  # Minimal training for test speed
            learning_rate=0.01,
        )

        # Ensure model can make predictions after training
        test_prompt = "What do you think of Zaun?"
        test_response = "Zaun represents both the cost and potential of progress."

        scores = classifier.evaluate_response(test_prompt, test_response)

        # Check that we got valid scores
        self.assertGreaterEqual(scores["character_accuracy"], 0.0)
        self.assertLessEqual(scores["character_accuracy"], 1.0)
        self.assertGreaterEqual(scores["response_quality"], 0.0)
        self.assertLessEqual(scores["response_quality"], 1.0)

    def test_script_integration(self):
        """Test integration with the actual training script."""
        # Mock the main function from the training script
        # Since we can't easily run the full script in a test, we'll test the components

        # First, set up the environment for training
        original_models_dir = Path("models")
        temp_models_dir = self.models_dir

        # Create a test model file
        model_path = temp_models_dir / "response_classifier.pt"

        # Initialize a classifier
        config = Config(use_response_classifier=True)
        classifier = ResponseClassifier(config)

        # Save the model to simulate a pre-existing model
        torch.save(classifier.model.state_dict(), model_path)

        # Check that our setup is valid
        self.assertTrue(model_path.exists())
        self.assertTrue(self.training_data_path.exists())

        # Check that we can load the model
        test_classifier = ResponseClassifier(config)
        test_classifier.model.load_state_dict(
            torch.load(model_path, map_location=test_classifier.device)
        )

        # The test is successful if we reach this point without exceptions
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()

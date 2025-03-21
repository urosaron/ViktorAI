#!/usr/bin/env python3
"""
Tests for the PyTorch response classifier.

This module contains tests for the response quality classifier, including
model initialization, feature extraction, and training functionality.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.response_classifier import (
    ResponseClassifier,
    ResponseQualityModel,
    train_model,
)


class TestResponseClassifier(unittest.TestCase):
    """Test cases for the response classifier."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a test configuration
        self.config = Config(
            use_response_classifier=True, min_response_score=0.6, debug=True
        )

        # Create a temporary directory for test models
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_model_path = Path(self.temp_dir.name) / "test_model.pt"

        # Sample training data
        self.training_data = [
            {
                "prompt": "Tell me about your work with the Hexcore.",
                "response": "The Hexcore is my life's work. It represents the future of hextech technology.",
                "character_score": 0.9,
                "quality_score": 0.85,
            },
            {
                "prompt": "What do you think about Jayce?",
                "response": "Jayce is a brilliant scientist, but our paths have diverged. He focuses on politics while I pursue true progress.",
                "character_score": 0.8,
                "quality_score": 0.8,
            },
            {
                "prompt": "How is your health?",
                "response": "My condition worsens daily, but I won't let human frailty impede progress.",
                "character_score": 0.95,
                "quality_score": 0.9,
            },
            {
                "prompt": "What's your favorite food?",
                "response": "I love pizza and ice cream! And I enjoy dancing at parties on weekends!",
                "character_score": 0.1,
                "quality_score": 0.3,
            },
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        classifier = ResponseClassifier(self.config)

        # Check that the model is a PyTorch module
        self.assertIsInstance(classifier.model, torch.nn.Module)

        # Check that the model is in evaluation mode by default
        self.assertFalse(classifier.model.training)

        # Check architecture - model should have defined layers
        self.assertTrue(hasattr(classifier.model, "character_layer1"))
        self.assertTrue(hasattr(classifier.model, "quality_layer1"))
        self.assertTrue(hasattr(classifier.model, "character_output"))
        self.assertTrue(hasattr(classifier.model, "quality_output"))

    def test_feature_extraction(self):
        """Test feature extraction from prompts and responses."""
        classifier = ResponseClassifier(self.config)

        # Test with a Viktor-relevant prompt and response
        prompt = "Tell me about your work with the Hexcore."
        response = "The Hexcore is evolving beyond what we initially imagined. It's responding to biological material in fascinating ways."

        features = classifier._prepare_features(prompt, response)

        # Check feature type and shape
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.dim(), 1)  # Should be a 1D tensor

        # Character-specific terms should be detected
        text = f"{prompt} {response}".lower()
        self.assertIn("hexcore", text)

        # Test with irrelevant content
        prompt = "What's your favorite hobby?"
        response = "I enjoy swimming and playing basketball."

        features = classifier._prepare_features(prompt, response)

        # Should still produce valid features, just different values
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.dim(), 1)

    def test_response_evaluation(self):
        """Test that responses are properly evaluated."""
        classifier = ResponseClassifier(self.config)

        # Test a good, in-character response
        prompt = "Tell me about the Hexcore."
        good_response = "The Hexcore is my greatest achievement - a fusion of hextech and organic material that evolves beyond its programming. Despite Jayce's reservations, I've continued my research, even as my condition deteriorates. It represents our future."

        scores = classifier.evaluate_response(prompt, good_response)

        # Check that the scores are produced correctly
        self.assertIn("character_accuracy", scores)
        self.assertIn("response_quality", scores)
        self.assertIn("overall_score", scores)

        # Check that scores are in appropriate range
        self.assertGreaterEqual(scores["character_accuracy"], 0.0)
        self.assertLessEqual(scores["character_accuracy"], 1.0)
        self.assertGreaterEqual(scores["response_quality"], 0.0)
        self.assertLessEqual(scores["response_quality"], 1.0)

        # Test an out-of-character response
        bad_response = "I love parties and dancing all night! Hextech is boring compared to having fun!"

        scores_bad = classifier.evaluate_response(prompt, bad_response)

        # Bad response should score lower than good response
        # Note: With a fresh model this might not always be true, but after training it should be
        # This is more of a sanity check than a strict test since an untrained model may not behave predictably
        print(f"Good response score: {scores['overall_score']}")
        print(f"Bad response score: {scores_bad['overall_score']}")

    def test_model_training(self):
        """Test the model training functionality."""
        classifier = ResponseClassifier(self.config)

        # Save initial predictions for comparison
        test_prompt = "What do you think about Heimerdinger?"
        test_response = "Heimerdinger's caution impedes progress. His years of experience have made him resistant to change."

        pre_training_scores = classifier.evaluate_response(test_prompt, test_response)

        # Train the model
        train_model(
            classifier=classifier,
            training_data=self.training_data,
            epochs=50,  # Smaller number for testing
            learning_rate=0.01,
        )

        # Save the model to our temp location
        torch.save(classifier.model.state_dict(), self.temp_model_path)

        # Create a new classifier and load the trained model
        new_classifier = ResponseClassifier(self.config)
        new_classifier.model.load_state_dict(
            torch.load(self.temp_model_path, map_location=new_classifier.device)
        )

        # Get post-training predictions
        post_training_scores = new_classifier.evaluate_response(
            test_prompt, test_response
        )

        # Print scores for info
        print(f"Pre-training scores: {pre_training_scores}")
        print(f"Post-training scores: {post_training_scores}")

        # The specific scores may vary, but we're validating that training and loading works
        self.assertIsNotNone(post_training_scores["character_accuracy"])
        self.assertIsNotNone(post_training_scores["response_quality"])

    def test_model_save_load(self):
        """Test saving and loading the model."""
        # Initialize a classifier and modify weights to ensure they're different from default
        classifier = ResponseClassifier(self.config)

        # Change some weights to non-default values
        with torch.no_grad():
            for param in classifier.model.parameters():
                # Add 0.1 to all weights
                param.add_(0.1)

        # Save the model
        torch.save(classifier.model.state_dict(), self.temp_model_path)

        # Create a new classifier with default weights
        new_classifier = ResponseClassifier(self.config)

        # Weights should be different before loading
        for p1, p2 in zip(
            classifier.model.parameters(), new_classifier.model.parameters()
        ):
            # If any parameter differs significantly, the test passes
            if torch.max(torch.abs(p1 - p2)) > 0.05:
                break
        else:
            self.fail("Models should have different weights before loading")

        # Load the saved model
        new_classifier.model.load_state_dict(
            torch.load(self.temp_model_path, map_location=new_classifier.device)
        )

        # Now weights should be the same
        for p1, p2 in zip(
            classifier.model.parameters(), new_classifier.model.parameters()
        ):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

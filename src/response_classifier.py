"""
Response Quality Classifier for ViktorAI.

This module implements a PyTorch-based classifier to evaluate whether
responses accurately reflect Viktor's character from Arcane.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ResponseClassifier:
    """PyTorch classifier for evaluating response quality and character accuracy."""

    def __init__(self, config):
        """Initialize the response classifier.

        Args:
            config: Configuration object containing settings.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        self.model.to(self.device)

    def _initialize_model(self) -> nn.Module:
        """Initialize the PyTorch model.

        Returns:
            The initialized PyTorch model.
        """
        model = ResponseQualityModel()

        # Check if a pre-trained model exists
        model_path = Path("models/response_classifier.pt")
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Loaded pre-trained response classifier model.")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                print("Initializing new model.")
        else:
            print("No pre-trained model found. Initializing new model.")

        return model

    def evaluate_response(self, prompt: str, response: str) -> Dict[str, float]:
        """Evaluate the quality and character accuracy of a response.

        Args:
            prompt: The user prompt.
            response: The generated response to evaluate.

        Returns:
            Dictionary with evaluation scores.
        """
        # Create input features from prompt and response
        features = self._prepare_features(prompt, response)

        # Set model to evaluation mode
        self.model.eval()

        # Generate evaluation scores
        with torch.no_grad():
            character_score, quality_score = self.model(features)

        return {
            "character_accuracy": float(character_score),
            "response_quality": float(quality_score),
            "overall_score": float((character_score + quality_score) / 2),
        }

    def _prepare_features(self, prompt: str, response: str) -> torch.Tensor:
        """Prepare input features for the model.

        In a real implementation, this would use a text embedding model.
        For now, we'll use a simple bag-of-words approach with limited vocabulary.

        Args:
            prompt: The user prompt.
            response: The generated response.

        Returns:
            Tensor containing the input features.
        """
        # For demonstration purposes, we'll use a simplified approach
        # In production, you would use proper embeddings from a model like BERT

        # Key character terms that represent Viktor's character
        character_terms = [
            "hextech",
            "hexcore",
            "progress",
            "evolution",
            "science",
            "research",
            "viktor",
            "cough",
            "disability",
            "illness",
            "zaun",
            "piltover",
            "jayce",
            "sky",
            "heimerdinger",
            "future",
        ]

        # Create a simple feature vector based on term presence
        text = (prompt + " " + response).lower()

        # Count occurrences of character terms
        term_counts = [text.count(term) for term in character_terms]

        # Additional features
        response_length = len(response.split())
        prompt_length = len(prompt.split())

        # Combine all features
        features = term_counts + [response_length, prompt_length]

        return torch.tensor(features, dtype=torch.float32, device=self.device)


class ResponseQualityModel(nn.Module):
    """PyTorch model for evaluating response quality and character accuracy."""

    def __init__(self, input_size: int = 18, hidden_size: int = 32):
        """Initialize the model.

        Args:
            input_size: Size of input features.
            hidden_size: Size of hidden layer.
        """
        super(ResponseQualityModel, self).__init__()

        # Layers for character accuracy score
        self.character_layer1 = nn.Linear(input_size, hidden_size)
        self.character_layer2 = nn.Linear(hidden_size, hidden_size)
        self.character_output = nn.Linear(hidden_size, 1)

        # Layers for response quality score
        self.quality_layer1 = nn.Linear(input_size, hidden_size)
        self.quality_layer2 = nn.Linear(hidden_size, hidden_size)
        self.quality_output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input tensor with features.

        Returns:
            Tuple of (character_accuracy_score, response_quality_score).
        """
        # Character accuracy branch
        character = F.relu(self.character_layer1(x))
        character = F.relu(self.character_layer2(character))
        character_score = torch.sigmoid(self.character_output(character))

        # Response quality branch
        quality = F.relu(self.quality_layer1(x))
        quality = F.relu(self.quality_layer2(quality))
        quality_score = torch.sigmoid(self.quality_output(quality))

        return character_score.squeeze(), quality_score.squeeze()


def train_model(
    classifier: ResponseClassifier,
    training_data: List[Dict],
    epochs: int = 100,
    learning_rate: float = 0.001,
):
    """Train the response classifier model.

    Args:
        classifier: ResponseClassifier instance.
        training_data: List of dictionaries with 'prompt', 'response',
                      'character_score', and 'quality_score'.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
    """
    # Set model to training mode
    classifier.model.train()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(classifier.model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for sample in training_data:
            # Prepare features and targets
            features = classifier._prepare_features(
                sample["prompt"], sample["response"]
            )
            character_target = torch.tensor(
                sample["character_score"], dtype=torch.float32, device=classifier.device
            )
            quality_target = torch.tensor(
                sample["quality_score"], dtype=torch.float32, device=classifier.device
            )

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            character_pred, quality_pred = classifier.model(features)

            # Calculate loss
            character_loss = criterion(character_pred, character_target)
            quality_loss = criterion(quality_pred, quality_target)
            loss = character_loss + quality_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(training_data)}")

    # Save the trained model
    save_path = Path("models")
    save_path.mkdir(exist_ok=True)
    torch.save(classifier.model.state_dict(), save_path / "response_classifier.pt")
    print(f"Model saved to {save_path / 'response_classifier.pt'}")

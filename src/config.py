"""
Configuration module for ViktorAI.

This module defines the configuration class that holds settings for ViktorAI.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Dict, List, Optional, Any


@dataclass
class Config:
    """Configuration settings for ViktorAI.

    This class holds all configuration settings for the ViktorAI system,
    including model parameters, response classification settings, and
    character data paths.
    """

    # Ollama model settings
    model_name: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 500

    # Response classifier settings
    use_response_classifier: bool = False
    min_response_score: float = 0.6
    debug: bool = False

    # Character data settings
    character_data_dir: str = "character_data"
    main_prompt_file: str = "viktor_main_prompt.md"
    core_profile_file: str = "viktor_core_profile.md"
    technical_knowledge_file: str = "viktor_technical_knowledge.md"
    relationships_file: str = "viktor_relationships.md"
    world_context_file: str = "viktor_world_context.md"
    response_guidelines_file: str = "viktor_response_guidelines.md"
    character_analysis_file: str = "viktor_scenes_and_events.md"
    character_files: List[str] = field(
        default_factory=lambda: [
            "viktor_core_profile.md",
            "viktor_technical_knowledge.md",
            "viktor_relationships.md",
            "viktor_world_context.md",
            "viktor_response_guidelines.md",
        ]
    )

    # ViktorBrain integration settings
    brain_api_url: str = "http://localhost:8000"
    brain_neurons: int = 1000
    brain_connection_density: float = 0.1
    brain_spontaneous_activity: float = 0.02
    use_brain: bool = True
    
    def __post_init__(self):
        """Run validation and setup after initialization."""
        # Validate temperature
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {self.temperature}")

        # Validate max_tokens
        if self.max_tokens < 10:
            raise ValueError(f"max_tokens must be at least 10, got {self.max_tokens}")

        # Validate min_response_score
        if self.min_response_score < 0.0 or self.min_response_score > 1.0:
            raise ValueError(
                f"min_response_score must be between 0.0 and 1.0, got {self.min_response_score}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.

        Returns:
            A dictionary representation of the config.
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_response_classifier": self.use_response_classifier,
            "min_response_score": self.min_response_score,
            "debug": self.debug,
            "character_data_dir": self.character_data_dir,
            "main_prompt_file": self.main_prompt_file,
            "core_profile_file": self.core_profile_file,
            "technical_knowledge_file": self.technical_knowledge_file,
            "relationships_file": self.relationships_file,
            "world_context_file": self.world_context_file,
            "response_guidelines_file": self.response_guidelines_file,
            "character_analysis_file": self.character_analysis_file,
            "character_files": self.character_files,
            "brain_api_url": self.brain_api_url,
            "brain_neurons": self.brain_neurons,
            "brain_connection_density": self.brain_connection_density,
            "brain_spontaneous_activity": self.brain_spontaneous_activity,
            "use_brain": self.use_brain,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary.

        Args:
            config_dict: A dictionary containing configuration values.

        Returns:
            A Config instance with the provided values.
        """
        return cls(**config_dict)

    def get_character_file_path(self, filename: str) -> Path:
        """Get the full path to a character data file."""
        return Path(self.character_data_dir) / filename

    def get_all_character_files(self) -> List[Path]:
        """Get a list of all character data files."""
        return [self.get_character_file_path(filename) for filename in self.character_files]

    def get_model_params(self) -> Dict:
        """Get model parameters as a dictionary."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

"""
Configuration module for ViktorAI.

This module contains the configuration settings for the ViktorAI chatbot,
including model settings and file paths.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class Config:
    """Configuration class for ViktorAI."""
    
    # Model settings
    model_name: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 500
    
    # File paths
    character_data_dir: Path = None  # We'll set this in __post_init__
    
    # Character data files
    core_profile_file: str = "viktor_core_profile.md"
    technical_knowledge_file: str = "viktor_technical_knowledge.md"
    relationships_file: str = "viktor_relationships.md"
    world_context_file: str = "viktor_world_context.md"
    response_guidelines_file: str = "viktor_response_guidelines.md"
    test_scenarios_file: str = "viktor_test_scenarios.md"
    main_prompt_file: str = "viktor_main_prompt.md"
    character_analysis_file: str = "viktor_scenes_and_events.md"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set character_data_dir if not already set
        if self.character_data_dir is None:
            # Try to find character_data directory in current or parent directory
            current_dir_path = Path("character_data")
            parent_dir_path = Path("..") / "character_data"
            project_root_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "character_data"
            
            if current_dir_path.exists():
                self.character_data_dir = current_dir_path
            elif parent_dir_path.exists():
                self.character_data_dir = parent_dir_path
            elif project_root_path.exists():
                self.character_data_dir = project_root_path
            else:
                raise FileNotFoundError(f"Character data directory not found in either {current_dir_path}, {parent_dir_path}, or {project_root_path}")
        
        # Ensure character_data_dir is a Path object
        if isinstance(self.character_data_dir, str):
            self.character_data_dir = Path(self.character_data_dir)
        
        # Validate that character data directory exists
        if not self.character_data_dir.exists():
            raise FileNotFoundError(f"Character data directory not found: {self.character_data_dir}")
    
    def get_character_file_path(self, filename: str) -> Path:
        """Get the full path to a character data file."""
        return self.character_data_dir / filename
    
    def get_all_character_files(self) -> List[Path]:
        """Get a list of all character data files."""
        return [
            self.get_character_file_path(self.core_profile_file),
            self.get_character_file_path(self.technical_knowledge_file),
            self.get_character_file_path(self.relationships_file),
            self.get_character_file_path(self.world_context_file),
            self.get_character_file_path(self.response_guidelines_file),
            self.get_character_file_path(self.main_prompt_file),
        ]
    
    def get_model_params(self) -> Dict:
        """Get model parameters as a dictionary."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        } 
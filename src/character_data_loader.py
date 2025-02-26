"""
Character Data Loader for ViktorAI.

This module handles loading and processing character data from markdown files.
It provides functions to read, parse, and organize the character information
for use in generating responses.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class CharacterDataLoader:
    """Loads and processes character data from markdown files."""
    
    def __init__(self, config):
        """Initialize the CharacterDataLoader.
        
        Args:
            config: Configuration object containing file paths and settings.
        """
        self.config = config
        self.character_data = {}
        self.main_prompt = ""
        self.character_analysis = ""
    
    def load_all_data(self) -> None:
        """Load all character data files."""
        # Load main character data files
        self.character_data = {
            "core_profile": self._load_file(self.config.core_profile_file),
            "technical_knowledge": self._load_file(self.config.technical_knowledge_file),
            "relationships": self._load_file(self.config.relationships_file),
            "world_context": self._load_file(self.config.world_context_file),
            "response_guidelines": self._load_file(self.config.response_guidelines_file),
        }
        
        # Load main prompt
        self.main_prompt = self._load_file(self.config.main_prompt_file)
        
        # Load character analysis (this is a large file, so we load it separately)
        self.character_analysis = self._load_file(self.config.character_analysis_file)
        
        print(f"Loaded {len(self.character_data) + 2} character data files.")
    
    def _load_file(self, filename: str) -> str:
        """Load a single file and return its contents as a string.
        
        Args:
            filename: Name of the file to load.
            
        Returns:
            The contents of the file as a string.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = self.config.get_character_file_path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Character data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_combined_character_data(self) -> str:
        """Combine all character data into a single string.
        
        Returns:
            A string containing all character data.
        """
        combined = ""
        
        # Add core profile
        combined += "# Viktor's Core Profile\n\n"
        combined += self.character_data["core_profile"]
        combined += "\n\n"
        
        # Add technical knowledge
        combined += "# Viktor's Technical Knowledge\n\n"
        combined += self.character_data["technical_knowledge"]
        combined += "\n\n"
        
        # Add relationships
        combined += "# Viktor's Relationships\n\n"
        combined += self.character_data["relationships"]
        combined += "\n\n"
        
        # Add world context
        combined += "# Viktor's World Context\n\n"
        combined += self.character_data["world_context"]
        combined += "\n\n"
        
        # Add response guidelines
        combined += "# Response Guidelines\n\n"
        combined += self.character_data["response_guidelines"]
        combined += "\n\n"
        
        return combined
    
    def get_main_prompt(self) -> str:
        """Get the main prompt for the chatbot.
        
        Returns:
            The main prompt as a string.
        """
        return self.main_prompt
    
    def get_character_analysis(self) -> str:
        """Get the character analysis.
        
        Returns:
            The character analysis as a string.
        """
        return self.character_analysis
    
    def search_character_analysis(self, query: str) -> List[Tuple[str, str]]:
        """Search the character analysis for relevant information.
        
        Args:
            query: The search query.
            
        Returns:
            A list of tuples containing section titles and content.
        """
        # Split the character analysis into sections
        sections = []
        current_section = ""
        current_content = []
        
        for line in self.character_analysis.split('\n'):
            if line.startswith('# ') or line.startswith('## '):
                # Save the previous section
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                
                # Start a new section
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        # Search for the query in each section
        results = []
        query = query.lower()
        
        for section_title, section_content in sections:
            if query in section_title.lower() or query in section_content.lower():
                results.append((section_title, section_content))
        
        return results 
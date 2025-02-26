"""
Tests for the ViktorAI character data loading components.

This module contains tests for the CharacterDataLoader class, which is responsible for
loading and processing character data files for the ViktorAI chatbot.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.character_data_loader import CharacterDataLoader

class TestCharacterDataLoader(unittest.TestCase):
    """Tests for the CharacterDataLoader class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = Config()
    
    def test_load_file(self):
        """Test loading a single file."""
        loader = CharacterDataLoader(self.config)
        content = loader._load_file(self.config.core_profile_file)
        self.assertIsNotNone(content)
        self.assertGreater(len(content), 0)
    
    def test_load_all_data(self):
        """Test loading all character data files."""
        loader = CharacterDataLoader(self.config)
        loader.load_all_data()
        
        # Check that all data was loaded
        self.assertIsNotNone(loader.character_data.get("core_profile"))
        self.assertIsNotNone(loader.character_data.get("technical_knowledge"))
        self.assertIsNotNone(loader.character_data.get("relationships"))
        self.assertIsNotNone(loader.character_data.get("world_context"))
        self.assertIsNotNone(loader.character_data.get("response_guidelines"))
        self.assertIsNotNone(loader.main_prompt)
        self.assertIsNotNone(loader.character_analysis)
    
    def test_get_combined_character_data(self):
        """Test getting combined character data."""
        loader = CharacterDataLoader(self.config)
        loader.load_all_data()
        
        combined_data = loader.get_combined_character_data()
        self.assertIsNotNone(combined_data)
        self.assertGreater(len(combined_data), 0)
        
        # Check that all sections are included
        self.assertIn("Viktor's Core Profile", combined_data)
        self.assertIn("Viktor's Technical Knowledge", combined_data)
        self.assertIn("Viktor's Relationships", combined_data)
        self.assertIn("Viktor's World Context", combined_data)
        self.assertIn("Response Guidelines", combined_data)
    
    def test_search_character_analysis(self):
        """Test searching the character analysis."""
        loader = CharacterDataLoader(self.config)
        loader.load_all_data()
        
        # Search for a keyword that should be in the analysis
        results = loader.search_character_analysis("hextech")
        self.assertGreater(len(results), 0)
        
        # Search for a keyword that should not be in the analysis
        results = loader.search_character_analysis("nonexistentterm12345")
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main() 
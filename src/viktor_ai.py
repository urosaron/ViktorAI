"""
ViktorAI - Main chatbot implementation.

This module implements the main ViktorAI chatbot functionality, including
loading character data, generating prompts, and handling conversations.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from src.character_data_loader import CharacterDataLoader
from src.llm_interface import OllamaInterface
from src.vector_store import VectorStore
from src.response_classifier import ResponseClassifier


class ViktorAI:
    """Main ViktorAI chatbot implementation."""

    def __init__(self, config):
        """Initialize the ViktorAI chatbot.

        Args:
            config: Configuration object containing settings.
        """
        self.config = config

        # Initialize character data loader
        self.data_loader = CharacterDataLoader(config)
        self.data_loader.load_all_data()

        # Initialize LLM interface
        self.llm = OllamaInterface(config)

        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()

        # Initialize response classifier (PyTorch)
        self.response_classifier = None
        if config.use_response_classifier:
            try:
                self.response_classifier = ResponseClassifier(config)
                print("Response classifier initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize response classifier: {e}")
                print("Continuing without response validation.")

        # Prepare system prompt
        self.system_prompt = self._prepare_system_prompt()

        print("ViktorAI initialized successfully.")

    def _initialize_vector_store(self):
        """Initialize the vector store for RAG."""
        vector_store_path = "vector_store"

        # Check if vector store exists
        if os.path.exists(vector_store_path):
            try:
                self.vector_store = VectorStore.load_local(vector_store_path)
                print(f"Loaded vector store from {vector_store_path}")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None
        else:
            print("Vector store not found. Run 'python -m src.indexer' to create it.")
            self.vector_store = None

    def _prepare_system_prompt(self) -> str:
        """Prepare the system prompt for the LLM.

        Returns:
            The system prompt as a string.
        """
        # Get the main prompt
        main_prompt = self.data_loader.get_main_prompt()

        # Get the combined character data
        character_data = self.data_loader.get_combined_character_data()

        # Combine them into a system prompt
        system_prompt = f"{main_prompt}\n\n{character_data}"

        return system_prompt

    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input.

        Args:
            user_input: The user's input message.

        Returns:
            Viktor's response as a string.
        """
        # Retrieve relevant context from the vector store
        context = self._retrieve_context(user_input)

        # Prepare the prompt with the retrieved context
        prompt = self._prepare_rag_prompt(user_input, context)

        # Get the current conversation history
        history = self.llm.get_history()

        # Initialize response variables
        response = None
        max_attempts = 3  # Maximum number of attempts to generate a good response
        attempts = 0

        while attempts < max_attempts:
            # If we have history, use it to generate a response
            if history:
                # Add the new user message
                messages = history + [{"role": "user", "content": prompt}]

                # Generate a response using the chat history
                response = self.llm.generate_with_chat_history(
                    messages=messages, system_prompt=self.system_prompt
                )
            else:
                # Generate a response without history
                response = self.llm.generate(
                    prompt=prompt, system_prompt=self.system_prompt
                )

            # If response classifier is not available or disabled, return the response
            if not self.response_classifier or not self.config.use_response_classifier:
                break

            # Evaluate the response quality
            evaluation = self.response_classifier.evaluate_response(
                user_input, response
            )

            # Check if the response meets quality thresholds
            if evaluation["overall_score"] >= self.config.min_response_score:
                if self.config.debug:
                    print(f"Response scores: {evaluation}")
                break

            # If the response doesn't meet quality thresholds, try again
            attempts += 1

            if self.config.debug:
                print(f"Response attempt {attempts} didn't meet quality thresholds.")
                print(f"Scores: {evaluation}")
                print(f"Trying again...")

            # If we've reached the maximum number of attempts, use the best response
            if attempts >= max_attempts:
                if self.config.debug:
                    print(
                        f"Couldn't generate a high-quality response after {max_attempts} attempts."
                    )
                break

        return response

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from the vector store.

        Args:
            query: The user's query.

        Returns:
            Retrieved context as a string.
        """
        if self.vector_store is None:
            # Fall back to the old method if vector store is not available
            if self._is_scene_query(query):
                return self._get_relevant_scene_info(query)
            return ""

        # Search the vector store for relevant documents
        results = self.vector_store.similarity_search_with_metadata(query, k=5)

        if not results:
            return ""

        # Format the results
        context = "Here is relevant information from my knowledge base:\n\n"

        for doc, score, metadata in results:
            # Add metadata information
            source = metadata.get("source", "unknown")
            section = metadata.get("section", "")
            doc_type = metadata.get("type", "")

            # Format based on document type
            if doc_type == "scene_analysis":
                context += f"--- Scene Information: {section} ---\n{doc}\n\n"
            elif doc_type == "character_trait":
                context += f"--- Character Trait: {section} ---\n{doc}\n\n"
            elif doc_type == "knowledge":
                context += f"--- Technical Knowledge: {section} ---\n{doc}\n\n"
            elif doc_type == "relationship":
                context += f"--- Relationship: {section} ---\n{doc}\n\n"
            elif doc_type == "world_knowledge":
                context += f"--- World Context: {section} ---\n{doc}\n\n"
            elif doc_type == "guideline":
                context += f"--- Response Guideline: {section} ---\n{doc}\n\n"
            else:
                context += f"--- {section} ---\n{doc}\n\n"

        return context

    def _prepare_rag_prompt(self, user_input: str, context: str) -> str:
        """Prepare a RAG prompt with retrieved context.

        Args:
            user_input: The user's input message.
            context: Retrieved context.

        Returns:
            The prepared prompt.
        """
        if not context:
            return user_input

        # Create a prompt that includes the retrieved context
        prompt = f"""I need to respond to the user as Viktor from Arcane Season 1.

{context}

Based on the information above and my character knowledge, I should respond to:

User: {user_input}

Remember to maintain Viktor's voice, personality, and knowledge boundaries. Only reference events that happened in Season 1. Do not make up events that didn't happen in the show."""

        return prompt

    # Legacy methods for backward compatibility

    def _is_scene_query(self, user_input: str) -> bool:
        """Check if the user is asking about a specific scene or event.

        Args:
            user_input: The user's input message.

        Returns:
            True if the user is asking about a specific scene, False otherwise.
        """
        # List of keywords that might indicate a scene query
        scene_keywords = [
            "scene",
            "episode",
            "moment",
            "when",
            "what happened",
            "how did you",
            "how did viktor",
            "what did you do",
            "what did viktor do",
            "how did you feel",
            "how did viktor feel",
            "jayce",
            "heimerdinger",
            "sky",
            "singed",
            "hexcore",
            "hextech",
        ]

        # Check if any of the keywords are in the user input
        user_input_lower = user_input.lower()
        for keyword in scene_keywords:
            if keyword.lower() in user_input_lower:
                return True

        return False

    def _get_relevant_scene_info(self, user_input: str) -> str:
        """Get relevant scene information from the character analysis.

        Args:
            user_input: The user's input message.

        Returns:
            Relevant scene information as a string.
        """
        # Extract keywords from the user input
        keywords = self._extract_keywords(user_input)

        # Search the character analysis for each keyword
        all_results = []
        for keyword in keywords:
            results = self.data_loader.search_character_analysis(keyword)
            all_results.extend(results)

        # Deduplicate results
        unique_results = []
        seen_sections = set()
        for section_title, section_content in all_results:
            if section_title not in seen_sections:
                unique_results.append((section_title, section_content))
                seen_sections.add(section_title)

        # Format the results
        if unique_results:
            formatted_results = (
                "Here is relevant information from my character analysis:\n\n"
            )
            for section_title, section_content in unique_results[
                :3
            ]:  # Limit to top 3 results
                formatted_results += f"{section_title}\n{section_content}\n\n"
            return formatted_results
        else:
            return "No specific scene information found."

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.

        Args:
            text: The text to extract keywords from.

        Returns:
            A list of keywords.
        """
        # List of common words to exclude
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "because",
            "as",
            "what",
            "when",
            "where",
            "how",
            "why",
            "who",
            "which",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "to",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "of",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "just",
            "don",
            "now",
        }

        # Split the text into words
        words = text.lower().split()

        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Add specific character names that might be important
        character_names = ["viktor", "jayce", "heimerdinger", "sky", "singed", "mel"]
        for name in character_names:
            if name in text.lower() and name not in keywords:
                keywords.append(name)

        # Add specific terms related to the show
        show_terms = ["hextech", "hexcore", "piltover", "zaun", "undercity", "shimmer"]
        for term in show_terms:
            if term in text.lower() and term not in keywords:
                keywords.append(term)

        return keywords

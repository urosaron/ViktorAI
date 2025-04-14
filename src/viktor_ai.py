"""
ViktorAI - Main chatbot implementation.

This module implements the main ViktorAI chatbot functionality, including
loading character data, generating prompts, and handling conversations.
"""

import os
from typing import List
import requests
import json

from src.character_data_loader import CharacterDataLoader
from src.llm_interface import OllamaInterface
from src.vector_store import VectorStore
from src.response_classifier import ResponseClassifier
from src.brain_client import BrainClient


class ViktorAI:
    """Main ViktorAI chatbot implementation."""

    def __init__(self, config):
        """Initialize the ViktorAI chatbot.

        Args:
            config: Configuration object containing settings.
        """
        self.config = config

        # Initialize brain client
        self.brain = BrainClient(
            api_url=config.brain_api_url,
            auto_initialize=True,
            neurons=config.brain_neurons,
            connection_density=config.brain_connection_density,
            spontaneous_activity=config.brain_spontaneous_activity
        )

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
        print(f"Brain client connected: {self.brain.is_connected}")

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

        return f"{main_prompt}\n\n{character_data}"

    def _process_through_brain(self, user_input: str) -> str:
        """Process the user input through ViktorBrain.

        Args:
            user_input: The user's input message.

        Returns:
            Processed input with brain metrics.
        """
        try:
            # Check if the brain client is connected
            if not self.brain.is_connected:
                if not self.brain.check_connection():
                    print("Warning: ViktorBrain API is not accessible")
                    return user_input

            # Process the input through the brain
            brain_analysis = self.brain.process_input(
                user_input=user_input,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            if not brain_analysis:
                print("Warning: No brain analysis received")
                return user_input
            
            # Format the brain analysis into the prompt
            brain_context = f"""
Brain State Analysis:
- Processing Mode: {brain_analysis.get('processing_mode', 'N/A')}
- Brain State: {brain_analysis.get('brain_state', 'N/A')}
- Activation Level: {brain_analysis.get('activation_level', 'N/A'):.2f}
- Dominant Cluster: {brain_analysis.get('dominant_cluster', 'N/A')}
- Technical/Emotional Balance: {brain_analysis.get('technical_ratio', 'N/A'):.2f}/{brain_analysis.get('emotional_ratio', 'N/A'):.2f}

Cluster Distribution:
{json.dumps(brain_analysis.get('cluster_distribution', {}), indent=2)}
"""
            
            return f"{user_input}\n\n{brain_context}"
            
        except Exception as e:
            print(f"Warning: Error processing through ViktorBrain: {e}")
            return user_input

    def _process_response_feedback(self, response: str) -> None:
        """Process response as feedback to the brain.

        Args:
            response: Viktor's response to the user.
        """
        try:
            # Check if the brain client is connected
            if not self.brain.is_connected:
                if not self.brain.check_connection():
                    print("Warning: ViktorBrain API is not accessible for feedback")
                    return

            # Send the response back to the brain as feedback
            self.brain.process_feedback(response)
            
        except Exception as e:
            print(f"Warning: Error sending feedback to ViktorBrain: {e}")

    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input.

        Args:
            user_input: The user's input message.

        Returns:
            Viktor's response as a string.
        """
        # Process the input through ViktorBrain
        processed_input = self._process_through_brain(user_input)
        
        # Retrieve relevant context from the vector store
        context = self._retrieve_context(processed_input)

        # Prepare the prompt with the retrieved context
        prompt = self._prepare_rag_prompt(processed_input, context)

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
                processed_input, response
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
                print("Trying again...")

            # If we've reached the maximum number of attempts, use the best response
            if attempts >= max_attempts:
                if self.config.debug:
                    print("Maximum attempts reached. Using best response.")
                break

        # Process the response as feedback to the brain
        self._process_response_feedback(response)

        # Return the response
        return response

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from the vector store.

        Args:
            query: The query to search for.

        Returns:
            Retrieved context as formatted string.
        """
        try:
            if not self.vector_store:
                return ""
                
            # Add query to get relevant context
            results = self.vector_store.query(query, top_k=3)
            
            if not results:
                return ""

            # Format the results for inclusion in the prompt
            context_parts = []
            
            for doc, score in results:
                # Handle results as tuples of (document, score)
                context_parts.append(f"--- Relevant information ---\n{doc}")
                
            context = "\n\n".join(context_parts)
            
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def _prepare_rag_prompt(self, user_input: str, context: str) -> str:
        """Prepare the prompt with retrieved context for RAG.

        Args:
            user_input: The user's input (possibly processed by brain)
            context: Retrieved context from the vector store

        Returns:
            Complete prompt for the LLM
        """
        # If no context was retrieved, just return the user input
        if not context.strip():
            return user_input

        # Create a prompt with the retrieved context
        prompt = f"""
I'll provide you with some relevant information that might help you respond:

{context}

Now, please respond to this message:

{user_input}
"""

        return prompt

    def _is_scene_query(self, user_input: str) -> bool:
        """Determine if a query is asking about a specific scene.

        Args:
            user_input: The user's input message.

        Returns:
            True if the query is about a scene, False otherwise.
        """
        # Clean the input
        input_lower = user_input.lower()
        
        # Scene-related keywords
        scene_keywords = [
            "scene", "episode", "moment", "when you", "remember when",
            "that time", "that scene", "that moment", "what happened when",
            "council", "hearing", "progress day", "hextech", "gemstone",
            "jayce", "laboratory", "workshop", "heimerdinger", "mel", "accident"
        ]
        
        # Check for scene keywords
        for keyword in scene_keywords:
            if keyword in input_lower:
                return True
                
        return False

    def _get_relevant_scene_info(self, user_input: str) -> str:
        """Get information about relevant scenes based on the query.

        Args:
            user_input: The user's input message.

        Returns:
            Information about relevant scenes.
        """
        # Clean the input
        input_lower = user_input.lower()
        
        # Identify potential scene topics
        topics = []
        
        # Check for specific scene keywords
        scene_topics = {
            "council": ["council", "hearing", "trial", "decision", "vote"],
            "lab": ["laboratory", "lab", "workshop", "invention", "experiment"],
            "hextech": ["hextech", "gemstone", "crystal", "invention", "experiment"],
            "progress day": ["progress day", "celebration", "demonstration"],
            "jayce": ["jayce", "partner", "colleague", "friend", "rivalry"],
            "heimerdinger": ["heimerdinger", "professor", "mentor", "yordle"],
            "mel": ["mel", "councilor", "medarda", "alliance"],
            "accident": ["accident", "explosion", "injury", "laboratory accident"]
        }
        
        # Identify relevant topics
        for topic, keywords in scene_topics.items():
            for keyword in keywords:
                if keyword in input_lower:
                    topics.append(topic)
                    break
        
        # If no specific topics identified, try a general search
        if not topics:
            # Extract keywords from the input
            keywords = self._extract_keywords(user_input)
            
            # Use the vector store to find relevant scene information
            if self.vector_store and keywords:
                search_query = " ".join(keywords[:5])
                results = self.vector_store.query(
                    search_query, 
                    top_k=3,
                    filter_fn=lambda doc: "scene" in doc.lower() if isinstance(doc, str) else False
                )
                
                if results:
                    context_parts = []
                    for doc, score in results:
                        # Handle results as tuples of (document, score)
                        context_parts.append(f"--- Relevant scene information ---\n{doc}")
                    return "\n\n".join(context_parts)
        
        # Use the identified topics to find relevant scenes
        if topics and self.vector_store:
            search_query = " ".join(topics)
            results = self.vector_store.query(
                search_query, 
                top_k=3,
                filter_fn=lambda doc: "scene" in doc.lower() if isinstance(doc, str) else False
            )
            
            if results:
                context_parts = []
                for doc, score in results:
                    # Handle results as tuples of (document, score)
                    context_parts.append(f"--- Relevant scene information ---\n{doc}")
                return "\n\n".join(context_parts)
        
        # If no results found, return empty context
        return ""

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text.

        Args:
            text: The text to extract keywords from.

        Returns:
            List of extracted keywords.
        """
        # Clean the text
        text_lower = text.lower()
        
        # Remove common stop words
        stop_words = [
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "about", "like", "through", "over", "before",
            "between", "after", "since", "without", "under", "within", "along",
            "following", "across", "behind", "beyond", "plus", "except", "but",
            "up", "out", "around", "down", "off", "above", "below", "to", "for",
            "with", "in", "on", "of", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "can", "could",
            "will", "would", "shall", "should", "may", "might", "must", "i", "you",
            "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
        ]
        
        # Tokenize by splitting on spaces and punctuation
        tokens = []
        for word in text_lower.split():
            # Remove punctuation
            word = ''.join(c for c in word if c.isalnum())
            if word and word not in stop_words:
                tokens.append(word)
        
        # Return unique tokens
        return list(dict.fromkeys(tokens))

"""
LLM Interface for ViktorAI.

This module handles interactions with the Ollama LLM, including sending prompts
and receiving responses.
"""

import json
import requests
from typing import Dict, List, Optional, Any

class OllamaInterface:
    """Interface for interacting with Ollama LLMs."""
    
    def __init__(self, config):
        """Initialize the OllamaInterface.
        
        Args:
            config: Configuration object containing model settings.
        """
        self.config = config
        self.api_base = "http://localhost:11434/api"
        self.history = []
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt to provide context.
            
        Returns:
            The generated response as a string.
            
        Raises:
            Exception: If there is an error communicating with the LLM.
        """
        # Prepare the request payload
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            # Send the request to Ollama
            response = requests.post(
                f"{self.api_base}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Update history
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": result["response"]})
            
            return result["response"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama: {e}")
    
    def generate_with_chat_history(self, 
                                  messages: List[Dict[str, str]], 
                                  system_prompt: Optional[str] = None) -> str:
        """Generate a response using chat history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            system_prompt: Optional system prompt to provide context.
            
        Returns:
            The generated response as a string.
            
        Raises:
            Exception: If there is an error communicating with the LLM.
        """
        # Prepare the request payload
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            # Send the request to Ollama
            response = requests.post(
                f"{self.api_base}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Update history with the new message
            self.history = messages.copy()
            self.history.append({"role": "assistant", "content": result["message"]["content"]})
            
            return result["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama: {e}")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.
        
        Returns:
            A list of message dictionaries with 'role' and 'content'.
        """
        return self.history
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = [] 
"""
ViktorBrain API Client

This module provides a client interface for ViktorAI to connect to the ViktorBrain API,
managing brain sessions, processing inputs, and handling feedback.
"""

import os
import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ViktorAI.BrainClient')

class BrainClient:
    """Client for connecting to the ViktorBrain API."""
    
    def __init__(
        self, 
        api_url: str = "http://localhost:8000",
        auto_initialize: bool = True,
        neurons: int = 1000,
        connection_density: float = 0.1,
        spontaneous_activity: float = 0.02
    ):
        """
        Initialize the brain client.
        
        Args:
            api_url: URL of the ViktorBrain API
            auto_initialize: Whether to initialize a brain session immediately
            neurons: Number of neurons for the brain (if auto_initialize is True)
            connection_density: Connection density (if auto_initialize is True)
            spontaneous_activity: Spontaneous activity rate (if auto_initialize is True)
        """
        self.api_url = api_url
        self.session_id = None
        self.config = {
            "neurons": neurons,
            "connection_density": connection_density,
            "spontaneous_activity": spontaneous_activity
        }
        
        # Store metrics from last API call
        self.last_metrics = None
        
        # Connection tracking
        self.is_connected = False
        
        # Auto-initialize if requested
        if auto_initialize:
            success = self.initialize()
            if success:
                logger.info(f"Auto-initialized brain with {neurons} neurons")
            else:
                logger.warning("Failed to auto-initialize brain")
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize a new brain session.
        
        Args:
            **kwargs: Additional configuration parameters to override defaults
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Update config with any provided kwargs
            config = self.config.copy()
            config.update(kwargs)
            
            # Make API request to initialize brain
            response = requests.post(
                f"{self.api_url}/initialize",
                json=config,
                timeout=30  # Longer timeout for initialization
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                self.last_metrics = data.get("metrics")
                self.is_connected = True
                logger.info(f"Initialized brain session: {self.session_id}")
                return True
            else:
                logger.error(f"Failed to initialize brain: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing brain: {e}")
            return False
    
    def process_input(
        self, 
        user_input: str, 
        temperature: float = 0.7, 
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Process user input through the brain.
        
        Args:
            user_input: The user's input to process
            temperature: Temperature parameter
            max_tokens: Maximum tokens parameter
            
        Returns:
            Dict containing brain analysis or None if processing failed
        """
        # Check if we have an active session
        if not self.session_id:
            if not self.initialize():
                logger.error("No active session and initialization failed")
                return self._default_metrics()
        
        try:
            # Make API request to process input
            response = requests.post(
                f"{self.api_url}/process/{self.session_id}",
                json={
                    "prompt": user_input,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                self.last_metrics = data.get("brain_analysis")
                return self.last_metrics
            elif response.status_code == 404:
                # Session expired or not found, try to initialize again
                logger.warning("Session expired, trying to reinitialize")
                if self.initialize():
                    return self.process_input(user_input, temperature, max_tokens)
                else:
                    return self._default_metrics()
            else:
                logger.error(f"Error processing input: {response.text}")
                return self._default_metrics()
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return self._default_metrics()
    
    def process_feedback(self, ai_response: str) -> bool:
        """
        Process AI response as feedback to the brain.
        
        Args:
            ai_response: The AI response to process
            
        Returns:
            bool: True if feedback was successfully processed, False otherwise
        """
        # Check if we have an active session
        if not self.session_id:
            logger.error("No active session for feedback")
            return False
        
        try:
            # Make API request to process feedback
            response = requests.post(
                f"{self.api_url}/feedback/{self.session_id}",
                json={"response": ai_response},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.last_metrics = data.get("brain_analysis")
                return True
            else:
                logger.error(f"Error processing feedback: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False
    
    def stimulate(
        self, 
        region: Optional[List[float]] = None, 
        intensity: float = 1.0,
        target_type: Optional[str] = None
    ) -> bool:
        """
        Directly stimulate a region of the brain.
        
        Args:
            region: Target region as [x, y, z, radius]
            intensity: Stimulation intensity (0-1)
            target_type: Target neuron type (e.g., "EMOTIONAL", "TECHNICAL")
            
        Returns:
            bool: True if stimulation was successful, False otherwise
        """
        # Check if we have an active session
        if not self.session_id:
            logger.error("No active session for stimulation")
            return False
        
        try:
            # Prepare stimulation data
            stim_data = {"intensity": intensity}
            
            if region:
                stim_data["region"] = region
            
            if target_type:
                stim_data["target_type"] = target_type
            
            # Make API request for stimulation
            response = requests.post(
                f"{self.api_url}/stimulate/{self.session_id}",
                json=stim_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.last_metrics = data.get("brain_analysis")
                return True
            else:
                logger.error(f"Error stimulating brain: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error stimulating brain: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about the current brain state.
        
        Returns:
            Dict containing brain metrics or default metrics if request failed
        """
        # Check if we have an active session
        if not self.session_id:
            logger.error("No active session for metrics")
            return self._default_metrics()
        
        try:
            # Make API request for metrics
            response = requests.get(
                f"{self.api_url}/metrics/{self.session_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.last_metrics = data.get("brain_analysis")
                return data
            else:
                logger.error(f"Error getting metrics: {response.text}")
                return {"brain_analysis": self._default_metrics()}
                
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"brain_analysis": self._default_metrics()}
    
    def _default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when API calls fail."""
        return {
            "processing_mode": "balanced",
            "brain_state": "moderately_active",
            "activation_level": 0.5,
            "dominant_cluster": "none",
            "cluster_distribution": {},
            "technical_ratio": 0.5,
            "emotional_ratio": 0.5,
            "error": "Could not retrieve metrics from ViktorBrain API"
        }
    
    def check_connection(self) -> bool:
        """
        Check if the API is accessible and operational.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                return True
            else:
                self.is_connected = False
                return False
        except Exception:
            self.is_connected = False
            return False
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizing the brain state.
        
        Returns:
            Dict containing visualization data or empty dict if request failed
        """
        # Check if we have an active session
        if not self.session_id:
            logger.error("No active session for visualization")
            return {}
        
        try:
            # Make API request for visualization data
            response = requests.get(
                f"{self.api_url}/visualization/{self.session_id}",
                timeout=15  # Longer timeout for visualization data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting visualization data: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            return {}
    
    def close(self) -> None:
        """Close the current session, cleaning up resources."""
        if not self.session_id:
            return
        
        try:
            # Make API request to delete session
            response = requests.delete(
                f"{self.api_url}/session/{self.session_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"Closed session: {self.session_id}")
            else:
                logger.warning(f"Error closing session: {response.text}")
                
        except Exception as e:
            logger.error(f"Error closing session: {e}")
        
        # Clear session info even if API call failed
        self.session_id = None
        self.is_connected = False 
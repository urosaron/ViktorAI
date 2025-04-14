"""
ViktorAI API

This module provides a FastAPI-based API for interacting with ViktorAI.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import Dict, List, Optional, Any, Union
import os

from src.viktor_ai import ViktorAI
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='viktorai_api.log'
)
logger = logging.getLogger('ViktorAI.API')

# Create FastAPI app
app = FastAPI(title="ViktorAI API", description="API for interacting with ViktorAI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="The user's message")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(default=500, description="Maximum tokens for response generation")
    use_brain: Optional[bool] = Field(default=True, description="Whether to use ViktorBrain for processing")

class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Viktor's response")
    brain_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Brain metrics if brain was used")

# Initialize ViktorAI with default config
# Read from environment variables if available
config = Config(
    model_name=os.getenv("MODEL_NAME", "llama3"),
    temperature=float(os.getenv("TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("MAX_TOKENS", "500")),
    use_response_classifier=os.getenv("USE_CLASSIFIER", "false").lower() == "true",
    min_response_score=float(os.getenv("MIN_SCORE", "0.6")),
    debug=os.getenv("DEBUG", "false").lower() == "true",
    brain_api_url=os.getenv("BRAIN_API_URL", "http://localhost:8000"),
    brain_neurons=int(os.getenv("BRAIN_NEURONS", "1000")),
    brain_connection_density=float(os.getenv("BRAIN_DENSITY", "0.1")),
    brain_spontaneous_activity=float(os.getenv("BRAIN_ACTIVITY", "0.02")),
    use_brain=os.getenv("USE_BRAIN", "true").lower() == "true"
)

# Initialize ViktorAI
viktor_ai = ViktorAI(config)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "status": "operational",
        "service": "ViktorAI API",
        "version": "1.0.0",
        "model": config.model_name,
        "brain_connected": viktor_ai.brain.is_connected if config.use_brain else False
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request and return Viktor's response."""
    try:
        # Log the request
        logger.info(f"Received chat request: {request.message[:50]}...")
        
        # Override config settings if specified in request
        original_temperature = config.temperature
        original_max_tokens = config.max_tokens
        original_use_brain = config.use_brain
        
        if request.temperature is not None:
            config.temperature = request.temperature
            
        if request.max_tokens is not None:
            config.max_tokens = request.max_tokens
            
        if request.use_brain is not None:
            config.use_brain = request.use_brain
        
        # Generate response
        response = viktor_ai.generate_response(request.message)
        
        # Reset config to original values
        config.temperature = original_temperature
        config.max_tokens = original_max_tokens
        config.use_brain = original_use_brain
        
        # Return response
        brain_metrics = None
        if config.use_brain and viktor_ai.brain.last_metrics:
            brain_metrics = viktor_ai.brain.last_metrics
            
        return {
            "response": response,
            "brain_metrics": brain_metrics
        }
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/brain_status")
async def brain_status():
    """Get the status of the ViktorBrain connection."""
    if not config.use_brain:
        return {
            "status": "disabled",
            "message": "ViktorBrain integration is disabled"
        }
        
    try:
        is_connected = viktor_ai.brain.check_connection()
        
        if is_connected:
            return {
                "status": "connected",
                "session_id": viktor_ai.brain.session_id,
                "metrics": viktor_ai.brain.last_metrics
            }
        else:
            return {
                "status": "disconnected",
                "message": "ViktorBrain API is not accessible"
            }
            
    except Exception as e:
        logger.error(f"Error checking brain status: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/reset_brain")
async def reset_brain():
    """Reset the ViktorBrain connection by initializing a new session."""
    if not config.use_brain:
        return {
            "status": "disabled",
            "message": "ViktorBrain integration is disabled"
        }
        
    try:
        # Close existing session
        if viktor_ai.brain.session_id:
            viktor_ai.brain.close()
            
        # Initialize new session
        success = viktor_ai.brain.initialize()
        
        if success:
            return {
                "status": "success",
                "message": "ViktorBrain session reset successfully",
                "session_id": viktor_ai.brain.session_id,
                "metrics": viktor_ai.brain.last_metrics
            }
        else:
            return {
                "status": "error",
                "message": "Failed to initialize new ViktorBrain session"
            }
            
    except Exception as e:
        logger.error(f"Error resetting brain: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/model_info")
async def model_info():
    """Get information about the current model configuration."""
    return {
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "use_response_classifier": config.use_response_classifier,
        "min_response_score": config.min_response_score,
        "debug": config.debug,
        "brain_integration": {
            "enabled": config.use_brain,
            "api_url": config.brain_api_url,
            "neurons": config.brain_neurons,
            "connection_density": config.brain_connection_density,
            "spontaneous_activity": config.brain_spontaneous_activity
        }
    }

@app.on_event("startup")
async def startup_event():
    """Run when the API starts up."""
    logger.info("ViktorAI API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Run when the API shuts down."""
    # Close ViktorBrain session if active
    if config.use_brain and viktor_ai.brain.session_id:
        try:
            viktor_ai.brain.close()
            logger.info("ViktorBrain session closed")
        except Exception as e:
            logger.error(f"Error closing ViktorBrain session: {e}")
    
    logger.info("ViktorAI API shutdown") 
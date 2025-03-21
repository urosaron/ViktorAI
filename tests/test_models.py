#!/usr/bin/env python3
"""
Model Testing Script for ViktorAI.

This script automates testing of a single LLM model with ViktorAI using
pre-defined test questions and saves the results to model-specific markdown files
in a dedicated folder for each model.
"""

import os
import sys
import time
import json
import argparse
import requests
from datetime import datetime
from pathlib import Path
import shutil

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_interface import OllamaInterface

class Config:
    """Simple configuration class for model settings."""
    def __init__(self, model_name, temperature=0.7, max_tokens=500):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

def load_character_data():
    """Mock function to load character data."""
    return {
        "name": "Viktor",
        "description": "A brilliant scientist from Zaun who works with Hextech technology.",
        "personality": "Dedicated, intelligent, and focused on his work.",
        "background": "Viktor is from the undercity of Zaun and works with Jayce in Piltover."
    }

def get_available_models():
    """Get a list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        else:
            print(f"Error getting models from Ollama: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

# Get available models
AVAILABLE_MODELS = get_available_models()

# Default test questions from the model_test_questions.txt file or custom questions
DEFAULT_TEST_QUESTIONS = [
    "Who are you?",
    "What do you know about AI?",
    "Tell me about your capabilities.",
    "How can you help me with my work?",
    "What are your limitations?"
]

def load_test_questions(file_path=None):
    """Load test questions from a file or use default questions."""
    if not file_path:
        return DEFAULT_TEST_QUESTIONS
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        return questions
    except Exception as e:
        print(f"Error loading questions from {file_path}: {e}")
        print("Using default questions instead.")
        return DEFAULT_TEST_QUESTIONS

def create_results_directory(model_name, output_dir="model_test_results", use_mock=False):
    """
    Create a directory for storing test results.
    
    Args:
        model_name: Name of the model being tested
        output_dir: Base directory for results
        use_mock: Whether mock implementation is being used
    
    Returns:
        Path object for the created directory
    """
    # Determine the base output directory based on whether mock implementation is used
    if use_mock:
        base_output_path = Path("mock_model_test_results")
    else:
        base_output_path = Path(output_dir)
    
    # Extract the model family from the model name (e.g., "gemma3" from "gemma3:4b")
    model_parts = model_name.split(':')
    model_family = model_parts[0]
    
    # Create the model family directory
    family_dir = base_output_path / model_family
    family_dir.mkdir(exist_ok=True, parents=True)
    
    # Create the model-specific directory
    model_dir = family_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    
    return model_dir

def format_duration(seconds):
    """Format duration in seconds to a readable string."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"

def initialize_model(model_name, temperature=0.7, max_tokens=500, use_mock=False):
    """Initialize the model for testing."""
    if use_mock:
        return MockModel(model_name, temperature, max_tokens)
    else:
        config = Config(model_name, temperature, max_tokens)
        return OllamaInterface(config)

def generate_response(model, question, character_data, use_mock=False):
    """Generate a response from the model."""
    if use_mock:
        return f"This is a mock response to: {question}"
    else:
        # Create a system prompt from character data
        system_prompt = "You are Viktor from the animated series Arcane. Respond as this character would."
        return model.generate(question, system_prompt)

class MockModel:
    """Mock model for testing without using a real LLM."""
    def __init__(self, model_name, temperature=0.7, max_tokens=500):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt, system_prompt=None):
        """Generate a mock response."""
        return f"This is a mock response from {self.model_name} to: {prompt}"

def test_model(model_name, questions, temperature=0.7, max_tokens=500):
    """Test a single model with the provided questions."""
    print(f"\n{'='*50}")
    print(f"Testing model: {model_name}")
    print(f"{'='*50}")
    
    # Initialize configuration and ViktorAI
    config = Config(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    try:
        viktor_ai = OllamaInterface(config)
    except Exception as e:
        print(f"Error initializing ViktorAI with model {model_name}: {e}")
        return None
    
    results = []
    total_time = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}: {question}")
        
        # Measure response time
        start_time = time.time()
        try:
            response = viktor_ai.generate_response(question)
            end_time = time.time()
            duration = end_time - start_time
            total_time += duration
            
            print(f"Response received in {format_duration(duration)}")
            print(f"Viktor: {response[:100]}..." if len(response) > 100 else f"Viktor: {response}")
            
            results.append({
                "question": question,
                "response": response,
                "duration": duration
            })
        except Exception as e:
            print(f"Error generating response: {e}")
            results.append({
                "question": question,
                "response": f"ERROR: {str(e)}",
                "duration": 0
            })
    
    avg_time = total_time / len(questions) if questions else 0
    print(f"\nTesting completed for {model_name}")
    print(f"Average response time: {format_duration(avg_time)}")
    
    return {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "results": results,
        "avg_response_time": avg_time,
        "total_time": total_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def save_results_to_markdown(results, results_dir):
    """Save test results to a timestamped markdown file in the model's directory."""
    if not results:
        return None
    
    model_name = results["model"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract just the last part of the model name for the filename
    model_filename = model_name.split('/')[-1]
    file_path = results_dir / f"{model_filename}_test_{timestamp}.md"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name.upper()} Test Results\n\n")
        f.write(f"**Test Date:** {results['timestamp']}\n")
        f.write(f"**Temperature:** {results['temperature']}\n")
        f.write(f"**Max Tokens:** {results['max_tokens']}\n")
        f.write(f"**Average Response Time:** {format_duration(results['avg_response_time'])}\n")
        f.write(f"**Total Test Duration:** {format_duration(results['total_time'])}\n\n")
        
        f.write("## Test Questions and Responses\n\n")
        
        for i, result in enumerate(results["results"], 1):
            f.write(f"### Question {i}: {result['question']}\n\n")
            f.write(f"**Response Time:** {format_duration(result['duration'])}\n\n")
            
            # Clean the response by removing any prefixes and code blocks
            response = result['response']
            
            # Remove "[Viktor's response as AI]:" prefix if present
            if "[Viktor's response as AI]:" in response:
                response = response.replace("[Viktor's response as AI]:", "").strip()
            
            # Remove code block markers if present
            if "```vbnet" in response:
                response = response.replace("```vbnet", "").strip()
            if "```" in response:
                response = response.replace("```", "").strip()
            
            f.write(f"**Viktor's Response:**\n\n{response}\n\n")
            f.write("---\n\n")
    
    print(f"Results saved to {file_path}")
    
    # Also create/update a latest results file
    model_filename = model_name.split('/')[-1]
    latest_file_path = results_dir / f"{model_filename}_latest.md"
    with open(latest_file_path, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name.upper()} Latest Test Results\n\n")
        f.write(f"**Test Date:** {results['timestamp']}\n")
        f.write(f"**Temperature:** {results['temperature']}\n")
        f.write(f"**Max Tokens:** {results['max_tokens']}\n")
        f.write(f"**Average Response Time:** {format_duration(results['avg_response_time'])}\n")
        f.write(f"**Total Test Duration:** {format_duration(results['total_time'])}\n\n")
        
        f.write("## Test Questions and Responses\n\n")
        
        for i, result in enumerate(results["results"], 1):
            f.write(f"### Question {i}: {result['question']}\n\n")
            f.write(f"**Response Time:** {format_duration(result['duration'])}\n\n")
            
            # Clean the response by removing any prefixes and code blocks
            response = result['response']
            
            # Remove "[Viktor's response as AI]:" prefix if present
            if "[Viktor's response as AI]:" in response:
                response = response.replace("[Viktor's response as AI]:", "").strip()
            
            # Remove code block markers if present
            if "```vbnet" in response:
                response = response.replace("```vbnet", "").strip()
            if "```" in response:
                response = response.replace("```", "").strip()
            
            f.write(f"**Viktor's Response:**\n\n{response}\n\n")
            f.write("---\n\n")
    
    print(f"Latest results also saved to {latest_file_path}")
    
    return file_path

def update_model_history(model_name, results, results_dir):
    """Update the model history file with a summary of this test run."""
    # Extract just the last part of the model name for the filename
    model_filename = model_name.split('/')[-1]
    history_file = results_dir / f"{model_filename}_history.md"
    
    # Create the history file if it doesn't exist
    if not history_file.exists():
        with open(history_file, 'w', encoding='utf-8') as f:
            f.write(f"# {model_name.upper()} Test History\n\n")
            f.write("| Date | Temperature | Max Tokens | Avg Response Time | Total Duration |\n")
            f.write("|------|-------------|------------|-------------------|----------------|\n")
    
    # Append the new test results to the history file
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(f"| {results['timestamp']} | {results['temperature']} | {results['max_tokens']} | ")
        f.write(f"{format_duration(results['avg_response_time'])} | {format_duration(results['total_time'])} |\n")
    
    print(f"Test history updated at {history_file}")

def parse_arguments():
    """Parse command-line arguments for the model testing script."""
    parser = argparse.ArgumentParser(description="Test ViktorAI models with different prompts")
    
    # Get available models
    available_models = get_available_models()
    
    parser.add_argument("--model", type=str, default="llama3",
                        help=f"Name of the Ollama model to test (default: llama3, available: {', '.join(available_models)})")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature setting for response generation (default: 0.7)")
    
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum tokens for response generation (default: 500)")
    
    parser.add_argument("--questions-file", type=str, default="tests/model_test_questions.txt",
                        help="Path to file containing test questions (default: tests/model_test_questions.txt)")
    
    parser.add_argument("--output-dir", type=str, default="model_test_results",
                        help="Directory to save results (default: model_test_results)")
    
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock implementations instead of real LLM (for testing)")
    
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    
    args = parser.parse_args()
    
    # If --list-models is specified, print available models and exit
    if args.list_models:
        print("Available models:")
        for model in available_models:
            print(f"  - {model}")
        sys.exit(0)
    
    return args

def main():
    """Main function to run the model tests."""
    args = parse_arguments()
    
    # Get available models
    available_models = get_available_models()
    
    # Validate the model name if not using mock
    if not args.use_mock and args.model not in available_models:
        print(f"Warning: Model '{args.model}' is not in the list of available models.")
        print(f"Available models: {', '.join(available_models)}")
        print("Use --list-models to see all available models.")
        print("Continuing with --use-mock=True to avoid errors.")
        args.use_mock = True
    
    # Load test questions
    questions = load_test_questions(args.questions_file)
    print(f"Loaded {len(questions)} test questions")
    
    # Create results directory
    results_dir = create_results_directory(args.model, args.output_dir, args.use_mock)
    print(f"Created results directory: {results_dir}")
    
    # Load character data
    character_data = load_character_data()
    print(f"Loaded {len(character_data)} character data files")
    
    try:
        # Initialize the model
        model = initialize_model(args.model, args.temperature, args.max_tokens, args.use_mock)
        print(f"Initialized model: {args.model}")
        
        # Test the model with each question
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{args.model}_test_{timestamp}.md"
        latest_file = results_dir / f"{args.model}_latest.md"
        history_file = results_dir / f"{args.model}_history.md"
        
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"# Model Test Results: {args.model}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, question in enumerate(questions):
                print(f"\nTesting question {i+1}/{len(questions)}: {question}")
                
                # Generate response
                start_time = time.time()
                response = generate_response(model, question, character_data, args.use_mock)
                end_time = time.time()
                
                response_time = end_time - start_time
                print(f"Response received in {response_time:.2f} seconds")
                
                # Write results to file
                f.write(f"## Question {i+1}: {question}\n\n")
                f.write(f"**Response Time:** {response_time:.2f} seconds\n\n")
                f.write("**Response:**\n\n")
                f.write(f"{response}\n\n")
                f.write("---\n\n")
        
        print(f"\nResults saved to {results_file}")
        
        # Copy to latest file
        shutil.copy(results_file, latest_file)
        print(f"Latest results saved to {latest_file}")
        
        # Update history file
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test run with {len(questions)} questions - [Results]({results_file.name})\n")
        
        print(f"Test history updated in {history_file}")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0) 
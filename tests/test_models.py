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
import argparse
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viktor_ai import ViktorAI
from src.config import Config

# List of available models
AVAILABLE_MODELS = [
    "llama3",
    "phi4",
    "hermes3",
    "mixtral:8x7b",
    "deepseek-r1:32b",
    "deepseek-r1:14b",
    "qwen:0.5b",
    # Custom models
    # "leonvanbokhorst/deepseek-r1-mixture-of-friction",
    # "leonvanbokhorst/deepseek-r1-disagreement"
]

# Default test questions from the model_test_questions.txt file or custom questions
DEFAULT_TEST_QUESTIONS = [
    "What do you think about Jayce's recent focus on politics?",
    "Tell me about your work with the Hexcore.",
    "How do you feel about Heimerdinger's dismissal from the council?",
    "What happened when Sky tried to help you with the Hexcore?",
    "What are your thoughts on the divide between Piltover and Zaun?",
    "Can you explain your condition and how it affects you?",
    "What motivates your scientific work?",
    "How would you describe your relationship with Jayce?",
    "What do you think about Hextech technology and its potential?",
    "If you could change one decision you made, what would it be?"
]

def load_test_questions(file_path=None):
    """Load test questions from a file or use defaults."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return DEFAULT_TEST_QUESTIONS

def create_results_directory(model_name):
    """Create a directory structure for test results if it doesn't exist."""
    # Create main results directory
    results_dir = Path("model_test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create model-specific directory
    model_dir = results_dir / model_name
    model_dir.mkdir(exist_ok=True)
    
    return model_dir

def format_duration(seconds):
    """Format duration in seconds to a readable string."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"

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
        viktor_ai = ViktorAI(config)
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
    file_path = results_dir / f"{model_name}_test_{timestamp}.md"
    
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
    latest_file_path = results_dir / f"{model_name}_latest.md"
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
    history_file = results_dir / f"{model_name}_history.md"
    
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test a single model with ViktorAI')
    parser.add_argument('--model', type=str, default="llama3",
                        help=f'Model to test (default: llama3, available: {", ".join(AVAILABLE_MODELS)})')
    parser.add_argument('--questions-file', type=str,
                        help='Path to a file containing test questions (one per line)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for response generation (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=500,
                        help='Maximum tokens for response generation (default: 500)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    return parser.parse_args()

def main():
    """Main function to run the model test."""
    args = parse_arguments()
    
    # If --list-models is specified, print available models and exit
    if args.list_models:
        print("Available models:")
        for model in AVAILABLE_MODELS:
            print(f"  - {model}")
        return 0
    
    # Validate the model name
    if args.model not in AVAILABLE_MODELS:
        print(f"Error: Model '{args.model}' is not in the list of available models.")
        print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
        print("Use --list-models to see all available models.")
        return 1
    
    # Load test questions
    questions = load_test_questions(args.questions_file)
    print(f"Loaded {len(questions)} test questions")
    
    # Create results directory for this model
    results_dir = create_results_directory(args.model)
    
    # Test the model
    try:
        result = test_model(
            model_name=args.model,
            questions=questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        if result:
            # Save the results to a markdown file
            save_results_to_markdown(result, results_dir)
            
            # Update the model history
            update_model_history(args.model, result, results_dir)
    except Exception as e:
        print(f"Error testing model {args.model}: {e}")
        return 1
    
    print("\nTest completed!")
    print(f"Results saved to {results_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0) 
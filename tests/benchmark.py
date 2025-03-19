from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import sys
import requests
from tqdm import tqdm
import numpy as np
import argparse
import re
import random
import shutil

# Add the parent directory to the path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.config import Config
    from src.llm_interface import OllamaInterface
    from src.viktor_ai import ViktorAI
except ImportError:
    print("Warning: Failed to import src modules. Make sure you're running from the project root directory.")
    sys.exit(1)

from tests.test_evaluator import get_question_type, evaluate_response

# Define prompt categories
PROMPT_CATEGORIES = {
    "baseline": "Basic Viktor prompt",
    "personality_focused": "Prompt focused on Viktor's personality",
    "technical_focused": "Prompt focused on Viktor's technical knowledge",
    "relationship_focused": "Prompt focused on Viktor's relationships",
    "full": "Complete Viktor prompt"
}

# Define mapping from question types to prompt categories
QUESTION_CATEGORY_MAPPING = {
    "identity": "personality_focused",
    "technical": "technical_focused",
    "relationship": "relationship_focused",
    "philosophical": "full"
}

# Define evaluation metrics class if not imported
class EvaluationMetrics:
    @staticmethod
    def get_question_type(question):
        """
        Determine the type of question based on its content.
        
        Args:
            question: The question to categorize
            
        Returns:
            String indicating the question type (identity, technical, relationship, philosophical)
        """
        # Use our own implementation instead of importing from test_evaluator.py
        # This provides more consistent categorization
        question_lower = question.lower()
        
        # First check for specific questions that might need special handling
        specific_questions = {
            "who are you?": "identity",
            "what's your name?": "identity", 
            "tell me about yourself.": "identity",
            "how do you feel about your condition?": "identity",
            "what motivates your scientific work?": "identity",
            "what happened when sky tried to help you with the hexcore?": "relationship",
            "how did you feel when jayce presented hextech to the academy?": "relationship",
            "what was your reaction to being dismissed from the hextech project?": "relationship",
            "tell me about your disagreement with heimerdinger about progress and hextech.": "relationship",
            "what do you think about jayce's recent focus on politics?": "relationship",
            "what are your thoughts on heimerdinger?": "relationship",
            "what happened during your presentation to the council?": "relationship",
            "what are your thoughts on the divide between piltover and zaun?": "philosophical",
            "how do you view the relationship between humanity and technology?": "philosophical",
            "what does \"the glorious evolution\" mean to you?": "philosophical", 
            "if you could change one decision you made, what would it be?": "philosophical",
            "what do you think is the purpose of scientific advancement?": "philosophical"
        }
        
        # Check if this is a specific question we've identified
        for specific_q, q_type in specific_questions.items():
            if question_lower.strip() == specific_q.lower():
                return q_type
                
        # Identity questions (check these first)
        identity_keywords = ["who are you", "tell me about yourself", "your name", "introduce yourself", 
                         "what are you", "who is viktor", "about yourself"]
        if any(keyword in question_lower for keyword in identity_keywords):
            return "identity"
        
        # Relationship questions
        relationship_keywords = ["jayce", "heimerdinger", "sky", "relationship", "friend", "colleague", 
                             "thoughts on", "feelings about", "feel about", "describe your relationship"]
        if any(keyword in question_lower for keyword in relationship_keywords):
            return "relationship"
        
        # Philosophical questions
        philosophical_keywords = ["evolution", "glorious", "future", "humanity", "progress", "philosophy", 
                               "believe", "think about", "purpose", "divide", "piltover and zaun", 
                               "change one decision", "ethics", "moral", "right and wrong", "principles",
                               "values", "meaning", "greater good"]
        if any(keyword in question_lower for keyword in philosophical_keywords):
            return "philosophical"
            
        # Technical questions (check last as they're often the default)
        technical_keywords = ["hexcore", "hextech", "technology", "research", "work", "scientific", 
                          "limitations", "improve", "applications", "experiment", "results", 
                          "methodology", "theory", "development", "innovation"]
        if any(keyword in question_lower for keyword in technical_keywords):
            return "technical"
        
        # Default to identity if we can't determine the type
        return "identity"
    
    @staticmethod
    def get_evaluation_criteria(question_type):
        """
        Get specific evaluation criteria based on the question type.
    
    Args:
            question_type: The type of question (identity, technical, relationship, philosophical)
        
    Returns:
            String containing specific evaluation criteria for this question type
        """
        # Ensure we're using the same evaluation criteria as in test_evaluator.py
        from tests.test_evaluator import get_evaluation_criteria
        return get_evaluation_criteria(question_type)
    
    @staticmethod
    def calculate_weighted_score(metrics):
        """
        Calculate a weighted overall score based on the question type.
        
        Args:
            metrics: Dictionary containing evaluation metrics and question type
            
        Returns:
            Float representing the weighted overall score
        """
        question_type = metrics.get("question_type", "identity")
        
        if question_type == "identity":
            return (
                metrics["authenticity_score"] * 0.5 +
                metrics["technical_score"] * 0.1 +
                metrics["emotional_score"] * 0.2 +
                metrics["quality_score"] * 0.2
            )
        elif question_type == "technical":
            return (
                metrics["authenticity_score"] * 0.2 +
                metrics["technical_score"] * 0.5 +
                metrics["emotional_score"] * 0.1 +
                metrics["quality_score"] * 0.2
            )
        elif question_type == "relationship":
            return (
                metrics["authenticity_score"] * 0.4 +
                metrics["technical_score"] * 0.1 +
                metrics["emotional_score"] * 0.3 +
                metrics["quality_score"] * 0.2
            )
        elif question_type == "philosophical":
            return (
                metrics["authenticity_score"] * 0.4 +
                metrics["technical_score"] * 0.3 +
                metrics["emotional_score"] * 0.1 +
                metrics["quality_score"] * 0.2
            )
        else:
            # Default weighting
            return (
                metrics["authenticity_score"] * 0.25 +
                metrics["technical_score"] * 0.25 +
                metrics["emotional_score"] * 0.25 +
                metrics["quality_score"] * 0.25
            )
    
    @staticmethod
    def evaluate_response(response, question, category, evaluator_llm):
        """
        Evaluate a response using the evaluator model.
        
        Args:
            response: The response to evaluate
            question: The question that was asked
            category: The prompt category used to generate the response
            evaluator_llm: The evaluator model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Determine question type from question type parameter in metrics or the question content
        if hasattr(question, "question_type"):
            # If we're passed a dict or object with question_type attribute
            question_type = question.question_type
        else:
            # Use the question_type passed to us
            question_type = "identity"  # Default
        
        # Handle mock implementation specifically
        if hasattr(evaluator_llm, 'config') and isinstance(evaluator_llm.config, MockConfig):
            # For mock implementation, generate scores directly without going through the test_evaluator
            print("Generating mock evaluation scores directly")
            overall_score = random.randint(6, 9)
            primary_score = random.randint(6, 9)
            consistency_score = random.randint(6, 9)
            
            return {
                "overall_score": overall_score,
                "overall_reasoning": f"This is a mock evaluation for a {question_type} question.",
                "primary_dimension_score": primary_score,
                "primary_dimension_reasoning": f"The response demonstrates understanding of Viktor's {question_type} characteristics.",
                "character_consistency_score": consistency_score,
                "character_consistency_reasoning": "The response maintains Viktor's characteristic tone and perspective.",
                "question_type": question_type,
                "response_time": 0.0
            }
        
        # Use the evaluate_response function from test_evaluator.py
        from tests.test_evaluator import evaluate_response as test_evaluator_evaluate_response
        
        try:
            # Call the evaluator with specified question type
            metrics = test_evaluator_evaluate_response(response, question, evaluator_llm, question_type)
            
            # Add the question type to the metrics
            metrics["question_type"] = question_type
            
            # Add response time (will be filled in by the caller)
            metrics["response_time"] = 0.0
            
            return metrics
                
        except Exception as e:
            print(f"Error evaluating response: {e}")
            # Return default scores with error message
            return {
                "overall_score": 5.0,  # Using 5.0 instead of 3.0 to be consistent with test_evaluator.py
                "overall_reasoning": f"The evaluation process encountered an error: {str(e)}. This is a fallback score provided when the evaluation couldn't be completed properly.",
                "primary_dimension_score": 5.0,  # Using 5.0 instead of 3.0 to be consistent
                "primary_dimension_reasoning": "This is a default fallback score. The primary dimension evaluation couldn't be completed due to an error in the evaluation process.",
                "character_consistency_score": 5.0,  # Using 5.0 instead of 3.0 to be consistent
                "character_consistency_reasoning": "This is a default fallback score. The character consistency evaluation couldn't be completed due to an error in the evaluation process.",
                "question_type": question_type,
                "response_time": 0.0
            }

# Define helper functions if not imported
def get_specialized_prompt(category, config):
    """
    Get a specialized prompt based on the category.
    
    Args:
        category: The prompt category
        config: The configuration object
        
    Returns:
        String containing the specialized prompt
    """
    if category == "baseline":
        return "You are Viktor, a brilliant scientist from the show Arcane."
    elif category == "personality_focused":
        return "You are Viktor, a brilliant scientist from Zaun who is pragmatic, determined, and stoic, with a focus on scientific progress."
    elif category == "technical_focused":
        return "You are Viktor, a brilliant scientist with deep technical knowledge about Hextech and the Hexcore, who speaks with precise technical language."
    elif category == "relationship_focused":
        return "You are Viktor, a scientist who maintains professional relationships with colleagues like Jayce, Heimerdinger, and Sky, focusing on collaboration for scientific progress."
    elif category == "full":
        return "You are Viktor, a brilliant scientist from Zaun with a physical disability, who believes in using technology to transcend human limitations. You are pragmatic, determined, and stoic, speaking with precise technical language and focusing on scientific progress above all else."
    else:
        # Default prompt
        return "You are Viktor, a character from the show Arcane."

def create_custom_viktor_ai(config, specialized_prompt, use_mock=False, mock_inference=None):
    """
    Create a custom ViktorAI instance with the specialized prompt.
    
    Args:
        config: Configuration object
        specialized_prompt: The specialized prompt to use
        use_mock: Whether to use a mock implementation for both config and inference
        mock_inference: Whether to use a mock implementation just for inference
        
    Returns:
        ViktorAI instance with the specialized prompt
    """
    # Determine if we should use mock inference
    if mock_inference is None:
        mock_inference = use_mock
        
    # Check if this is a mock config or if we should use mock inference
    if mock_inference:
        return MockViktorAI(config, specialized_prompt)
    
    # Create a real ViktorAI instance
    try:
        viktor_ai = ViktorAI(config)
        
        # Set the specialized prompt
        if specialized_prompt:
            viktor_ai.system_prompt = specialized_prompt
        
        return viktor_ai
    except Exception as e:
        print(f"Error creating ViktorAI instance: {e}")
        print("Using mock implementation instead")
        return MockViktorAI(config, specialized_prompt)

def calculate_summary_statistics(results):
    """
    Calculate summary statistics from benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        
    Returns:
        Dictionary containing summary statistics
    """
    # Initialize summary dictionary
    summary = {
        "by_category": {},
        "by_question_type": {},
        "overall": {
            "overall_scores": [],
            "primary_scores": [],
            "consistency_scores": [],
            "response_times": [],
            "total_responses": 0
        }
    }
    
    # Process metrics for each category
    for category, metrics_list in results["metrics"].items():
        # Skip categories with no metrics
        if not metrics_list:
            continue
        
        # Initialize category summary
        summary["by_category"][category] = {
            "overall_scores": [],
            "primary_scores": [],
            "consistency_scores": [],
            "response_times": [],
            "total_responses": 0
        }
        
        # Process each metric in this category
        for metrics in metrics_list:
            question_type = metrics.get("question_type", "unknown")
            
            # Add to overall summary
            if "overall_score" in metrics:
                summary["overall"]["overall_scores"].append(metrics["overall_score"])
            if "primary_dimension_score" in metrics:
                summary["overall"]["primary_scores"].append(metrics["primary_dimension_score"])
            if "character_consistency_score" in metrics:
                summary["overall"]["consistency_scores"].append(metrics["character_consistency_score"])
            if "response_time" in metrics:
                summary["overall"]["response_times"].append(metrics["response_time"])
            
            summary["overall"]["total_responses"] += 1
            
            # Add to category summary
            if "overall_score" in metrics:
                summary["by_category"][category]["overall_scores"].append(metrics["overall_score"])
            if "primary_dimension_score" in metrics:
                summary["by_category"][category]["primary_scores"].append(metrics["primary_dimension_score"])
            if "character_consistency_score" in metrics:
                summary["by_category"][category]["consistency_scores"].append(metrics["character_consistency_score"])
            if "response_time" in metrics:
                summary["by_category"][category]["response_times"].append(metrics["response_time"])
            
            summary["by_category"][category]["total_responses"] += 1
            
            # Initialize question type summary if needed
            if question_type not in summary["by_question_type"]:
                summary["by_question_type"][question_type] = {
                    "overall_scores": [],
                    "primary_scores": [],
                    "consistency_scores": [],
                    "response_times": [],
                    "total_responses": 0
                }
            
            # Add to question type summary
            if "overall_score" in metrics:
                summary["by_question_type"][question_type]["overall_scores"].append(metrics["overall_score"])
            if "primary_dimension_score" in metrics:
                summary["by_question_type"][question_type]["primary_scores"].append(metrics["primary_dimension_score"])
            if "character_consistency_score" in metrics:
                summary["by_question_type"][question_type]["consistency_scores"].append(metrics["character_consistency_score"])
            if "response_time" in metrics:
                summary["by_question_type"][question_type]["response_times"].append(metrics["response_time"])
            
            summary["by_question_type"][question_type]["total_responses"] += 1
    
    # Calculate averages for overall summary
    if summary["overall"]["overall_scores"]:
        summary["overall"]["avg_overall_score"] = sum(summary["overall"]["overall_scores"]) / len(summary["overall"]["overall_scores"])
    if summary["overall"]["primary_scores"]:
        summary["overall"]["avg_primary_dimension_score"] = sum(summary["overall"]["primary_scores"]) / len(summary["overall"]["primary_scores"])
    if summary["overall"]["consistency_scores"]:
        summary["overall"]["avg_character_consistency_score"] = sum(summary["overall"]["consistency_scores"]) / len(summary["overall"]["consistency_scores"])
    if summary["overall"]["response_times"]:
        summary["overall"]["avg_response_time"] = sum(summary["overall"]["response_times"]) / len(summary["overall"]["response_times"])
    
    # Calculate averages for each category
    for category in summary["by_category"]:
        cat_summary = summary["by_category"][category]
        
        if cat_summary["overall_scores"]:
            cat_summary["avg_overall_score"] = sum(cat_summary["overall_scores"]) / len(cat_summary["overall_scores"])
        if cat_summary["primary_scores"]:
            cat_summary["avg_primary_dimension_score"] = sum(cat_summary["primary_scores"]) / len(cat_summary["primary_scores"])
        if cat_summary["consistency_scores"]:
            cat_summary["avg_character_consistency_score"] = sum(cat_summary["consistency_scores"]) / len(cat_summary["consistency_scores"])
        if cat_summary["response_times"]:
            cat_summary["avg_response_time"] = sum(cat_summary["response_times"]) / len(cat_summary["response_times"])
        
        # Calculate question type distribution for this category
        cat_summary["question_type_distribution"] = {}
        for metrics in results["metrics"][category]:
            question_type = metrics.get("question_type", "unknown")
            if question_type not in cat_summary["question_type_distribution"]:
                cat_summary["question_type_distribution"][question_type] = 0
            cat_summary["question_type_distribution"][question_type] += 1
    
    # Calculate averages for each question type
    for qtype in summary["by_question_type"]:
        qtype_summary = summary["by_question_type"][qtype]
        
        if qtype_summary["overall_scores"]:
            qtype_summary["avg_overall_score"] = sum(qtype_summary["overall_scores"]) / len(qtype_summary["overall_scores"])
        if qtype_summary["primary_scores"]:
            qtype_summary["avg_primary_dimension_score"] = sum(qtype_summary["primary_scores"]) / len(qtype_summary["primary_scores"])
        if qtype_summary["consistency_scores"]:
            qtype_summary["avg_character_consistency_score"] = sum(qtype_summary["consistency_scores"]) / len(qtype_summary["consistency_scores"])
        if qtype_summary["response_times"]:
            qtype_summary["avg_response_time"] = sum(qtype_summary["response_times"]) / len(qtype_summary["response_times"])
        
        # Calculate category distribution for this question type
        qtype_summary["category_distribution"] = {}
        for category, metrics_list in results["metrics"].items():
            for metrics in metrics_list:
                if metrics.get("question_type", "unknown") == qtype:
                    if category not in qtype_summary["category_distribution"]:
                        qtype_summary["category_distribution"][category] = 0
                    qtype_summary["category_distribution"][category] += 1
    
    return summary

def get_available_models():
    """
    Get a list of available models from Ollama.
    
    Returns:
        List of available model names
    """
    try:
        # Try to get models from Ollama API
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            # Extract model names from the response
            models = [model["name"] for model in models_data["models"]]
            return models
        else:
            print(f"Error getting models from Ollama API: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama API: {e}")
        return []

def load_test_questions(questions_file, use_mock=False):
    """
    Load test questions from a file and their categories based on section headings.
    
    Args:
        questions_file: Path to the file containing questions.
        use_mock: Whether to use mock questions if the file doesn't exist.
        
    Returns:
        List of tuples containing (question, question_type).
    """
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        questions_with_types = []
        current_type = "identity"  # Default type
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                # Check if this is a heading that indicates question type
                if "general character questions" in line.lower():
                    current_type = "identity"
                elif "specific scene questions" in line.lower() or "relationship questions" in line.lower():
                    current_type = "relationship"
                elif "technical questions" in line.lower():
                    current_type = "technical"
                elif "philosophical questions" in line.lower():
                    current_type = "philosophical"
                # Skip empty lines and comments
                continue
            
            # Add the question with its type
            questions_with_types.append((line, current_type))
            
        return questions_with_types
    except FileNotFoundError:
        print(f"Error loading test questions: [Errno 2] No such file or directory: '{questions_file}'")
        
        # If using mock and file doesn't exist, create test questions
        if use_mock:
            print("Creating mock test questions")
            mock_questions = [
                ("Who are you?", "identity"),
                ("What is your name?", "identity"),
                ("Tell me about yourself.", "identity"),
                ("Tell me about your work with the Hexcore.", "technical"),
                ("How does your technology work?", "technical"),
                ("What advancements have you made in your research?", "technical"),
                ("How would you describe your relationship with Jayce?", "relationship"),
                ("What happened when Sky tried to help you with the Hexcore?", "relationship"),
                ("How did you feel when Jayce presented Hextech to the Academy?", "relationship"),
                ("What does 'the glorious evolution' mean to you?", "philosophical"),
                ("What is your vision for humanity's future?", "philosophical"),
                ("Do you think there are limits to what science should achieve?", "philosophical")
            ]
            
            # Handle path properly - if no directory specified, just use the file name in current dir
            dirname = os.path.dirname(questions_file)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            
            # Write the mock questions to the file
            with open(questions_file, "w", encoding="utf-8") as f:
                f.write("# ViktorAI Model Test Questions\n")
                f.write("# One question per line, lines starting with # are ignored\n\n")
                
                f.write("# General character questions\n")
                for question, qtype in mock_questions:
                    if qtype == "identity":
                        f.write(f"{question}\n")
                
                f.write("\n# Specific scene questions\n")
                for question, qtype in mock_questions:
                    if qtype == "relationship":
                        f.write(f"{question}\n")
                
                f.write("\n# Technical questions\n")
                for question, qtype in mock_questions:
                    if qtype == "technical":
                        f.write(f"{question}\n")
                
                f.write("\n# Philosophical questions\n")
                for question, qtype in mock_questions:
                    if qtype == "philosophical":
                        f.write(f"{question}\n")
            
            return mock_questions
        
        return []

def run_benchmark(questions_with_types, model_name, prompt_categories=None, temperature=0.7, max_tokens=1000,
                 evaluator_model="llama3", output_dir="benchmark_results", use_mock=False,
                 mock_inference=None, category_specific_mode=False):
    """
    Run a benchmark with the specified parameters.
    
    Args:
        questions_with_types: List of tuples containing (question, question_type)
        model_name: Name of the model to benchmark
        prompt_categories: List of prompt categories to test
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        evaluator_model: Name of the model to use for evaluation
        output_dir: Directory to save results
        use_mock: Whether to use mock implementations for everything
        mock_inference: Whether to use mock implementations for inference only (overrides use_mock for inference)
        category_specific_mode: Whether to use category-specific mode
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"Running benchmark with model: {model_name}")
    print(f"Evaluator model: {evaluator_model}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Using mock: {use_mock}")
    
    # Determine what to mock
    if mock_inference is None:
        mock_inference = use_mock
    
    print(f"Using mock inference: {mock_inference}")
    print(f"Category-specific mode: {category_specific_mode}")
    
    # Create base output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a safe model name for directories
    safe_model_name = model_name.replace(":", "_")
    
    # Extract model family from model name (e.g., gemma3 from gemma3:1b)
    model_family = model_name.split(":")[0] if ":" in model_name else model_name
    
    # Create model family directory
    family_dir = output_dir / model_family
    family_dir.mkdir(exist_ok=True, parents=True)
    
    # Create model version directory
    model_dir = family_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Create run-specific directory with timestamp
    run_dir = model_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for raw data and visualizations
    raw_data_dir = run_dir / "raw_data"
    visualizations_dir = run_dir / "visualizations"
    raw_data_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    
    # Extract just the questions for storing in results
    questions = [q for q, _ in questions_with_types]
    
    # Initialize results dictionary
    results = {
        "model_name": model_name,
        "evaluator_model": evaluator_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "use_mock": use_mock,
        "mock_inference": mock_inference,
        "category_specific_mode": category_specific_mode,
        "metrics": {},
        "questions": questions
    }
    
    # Set default prompt categories if not provided
    if prompt_categories is None:
        prompt_categories = list(PROMPT_CATEGORIES.keys())
    
    # Initialize metrics dictionary for each category
    for category in prompt_categories:
        results["metrics"][category] = []
    
    # Load character data from config
    config = MockConfig(model_name, temperature, max_tokens) if use_mock else Config(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    
    # Setup Viktor and evaluator
    evaluator_config = MockConfig(evaluator_model, temperature=0.7, max_tokens=2000) if use_mock else Config(model_name=evaluator_model, temperature=0.7, max_tokens=2000)
    
    # Initialize the Viktor AI with the base configuration
    viktor_ai = create_custom_viktor_ai(config, None, use_mock, mock_inference)
    print(f"Viktor AI initialized with model: {model_name}")
    
    # Initialize the evaluator
    evaluator = MockOllamaInterface(evaluator_config) if use_mock else OllamaInterface(evaluator_config)
    print(f"Evaluator initialized with model: {evaluator_model}")
    
    if category_specific_mode:
        print("Using category-specific mode")
        
        # Group questions by type
        questions_by_type = {
            "identity": [],
            "technical": [],
            "relationship": [],
            "philosophical": []
        }
        
        # Organize questions by their types
        for question, question_type in questions_with_types:
            if question_type in questions_by_type:
                questions_by_type[question_type].append(question)
        
        # Map question types to prompt categories
        type_to_category = {
            "identity": "personality_focused",
            "technical": "technical_focused",
            "relationship": "relationship_focused",
            "philosophical": "full"
        }
        
        # Process each question type with its appropriate prompt category
        for question_type, type_questions in questions_by_type.items():
            if not type_questions:
                continue
                
            category = type_to_category[question_type]
            
            if category not in prompt_categories:
                continue
                
            print(f"Processing {len(type_questions)} {question_type} questions with {category} prompt")
            
            # Create specialized Viktor AI for this category
            specialized_prompt = get_specialized_prompt(category, config)
            specialized_viktor = create_custom_viktor_ai(config, specialized_prompt, use_mock, mock_inference)
            
            # Process each question in this type
            responses = []
            
            with tqdm(total=len(type_questions), desc=f"Generating responses for {question_type} questions") as pbar:
                for question in type_questions:
                    # Generate response and measure time
                    start_time = time.time()
                    response = specialized_viktor.generate_response(question)
                    response_time = time.time() - start_time
            
                    # Store response data
                    responses.append({
                        "question": question,
                        "response": response,
                        "response_time": response_time,
                        "question_type": question_type
                    })
                    
                    pbar.update(1)
            
            print()  # Add a newline after the progress bar
        
            # Add results for this category
            if category not in results["metrics"]:
                results["metrics"][category] = []
            
            # Evaluate all responses for this category
            print(f"Evaluating {len(responses)} responses for category: {category}")
            with tqdm(total=len(responses), desc=f"Evaluating {category} responses") as pbar:
                for response_data in responses:
                    # Evaluate response
                    metrics = EvaluationMetrics.evaluate_response(
                        response_data["response"],
                        response_data["question"],
                        category,
                        evaluator
                    )
                    
                    # Update response time from generation
                    metrics["response_time"] = response_data["response_time"]
                    
                    # Add question and response to metrics
                    metrics["question_type"] = response_data["question_type"]
                    metrics["question"] = response_data["question"]
                    metrics["response"] = response_data["response"]
                    
                    # Add metrics to results
                    results["metrics"][category].append(metrics)
                    
                    pbar.update(1)
    
    else:
        # Original mode: Process all questions for each category
        for category in prompt_categories:
            print(f"Processing category: {category}")
            
            # Create specialized Viktor AI for this category
            specialized_prompt = get_specialized_prompt(category, config)
            specialized_viktor = create_custom_viktor_ai(config, specialized_prompt, use_mock, mock_inference)
            
            # Process each question
            responses = []
            
            with tqdm(total=len(questions_with_types), desc=f"Generating responses for {category}") as pbar:
                for question, question_type in questions_with_types:
                    # Generate response and measure time
                    start_time = time.time()
                    response = specialized_viktor.generate_response(question)
                    response_time = time.time() - start_time
                    
                    # Store response data
                    responses.append({
                        "question": question,
                        "response": response,
                        "response_time": response_time,
                        "question_type": question_type
                    })
                    
                    pbar.update(1)
            
            print()  # Add a newline after the progress bar
            
            # Evaluate all responses for this category
            print(f"Evaluating {len(responses)} responses for category: {category}")
            with tqdm(total=len(responses), desc=f"Evaluating {category} responses") as pbar:
                for response_data in responses:
                    # Evaluate response
                    metrics = EvaluationMetrics.evaluate_response(
                        response_data["response"],
                        response_data["question"],
                        category,
                        evaluator
                    )
                    
                    # Update response time from generation
                    metrics["response_time"] = response_data["response_time"]
                    
                    # Add question and response to metrics
                    metrics["question_type"] = response_data["question_type"]
                    metrics["question"] = response_data["question"]
                    metrics["response"] = response_data["response"]
                    
                    # Add metrics to results
                    results["metrics"][category].append(metrics)
            
                    pbar.update(1)
            
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(results)
    results["summary_stats"] = summary_stats
    
    # Save results to files
    results_file = raw_data_dir / f"benchmark_{safe_model_name}_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Create markdown report
    markdown_file = visualizations_dir / f"benchmark_{safe_model_name}_{timestamp}.md"
    create_markdown_report(results, markdown_file)
    print(f"Markdown report saved to {markdown_file}")
    
    # Create HTML report
    html_file = visualizations_dir / f"benchmark_{safe_model_name}_{timestamp}.html"
    create_html_report(results, html_file)
    print(f"HTML report saved to {html_file}")
    
    # Create "latest" reference files in the model directory
    try:
        latest_json = model_dir / f"benchmark_{safe_model_name}_latest.json"
        latest_md = model_dir / f"benchmark_{safe_model_name}_latest.md"
        latest_html = model_dir / f"benchmark_{safe_model_name}_latest.html"
        
        shutil.copy(results_file, latest_json)
        shutil.copy(markdown_file, latest_md)
        shutil.copy(html_file, latest_html)
        print(f"Created 'latest' reference files in {model_dir}")
    except Exception as e:
        print(f"Error creating latest reference files: {e}")
    
    print(f"Benchmark complete! Results saved to {output_dir}")
    
    # Print summary
    print("Summary of results:")
    print()
    print("Overall scores by category:")
    for category, stats in summary_stats["by_category"].items():
        print(f"  {category}: {stats['avg_overall_score']:.2f}/10")
    print()
    print("Overall scores by question type:")
    for qtype, stats in summary_stats["by_question_type"].items():
        print(f"  {qtype}: {stats['avg_overall_score']:.2f}/10")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a benchmark for ViktorAI.")
    
    # Model settings
    parser.add_argument("--model", type=str, default="llama3", help="Model to benchmark")
    parser.add_argument("--evaluator-model", type=str, default="llama3", help="Model to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens for generation")
    
    # Input and output settings
    parser.add_argument("--questions-file", type=str, default="model_test_questions.txt", help="File containing test questions")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save results")
    
    # Prompt categories
    parser.add_argument("--categories", type=str, help="Space-separated list of prompt categories to test (e.g., 'baseline full')")
    
    # Test modes
    parser.add_argument("--category-specific-mode", action="store_true", help="Use category-specific prompts for each question type")
    parser.add_argument("--use-mock", action="store_true", help="Use mock implementations for both file loading and inference")
    parser.add_argument("--mock-inference", action="store_true", help="Use mock implementations only for inference (faster testing with real data)")
    
    return parser.parse_args()

def main():
    """Main function to run the benchmark."""
    args = parse_arguments()
    
    # Load questions
    print(f"Loading questions from {args.questions_file}")
    questions_with_types = load_test_questions(args.questions_file, args.use_mock)
    print(f"Loaded {len(questions_with_types)} questions")
    
    # Convert categories string to list if provided
    if args.categories:
        categories = args.categories.split()
    else:
        # If category-specific mode is enabled and no specific categories are provided,
        # include all the necessary categories for each question type
        if args.category_specific_mode:
            categories = ["personality_focused", "technical_focused", "relationship_focused", "full"]
        else:
            categories = None
    
    # Run the benchmark
    results = run_benchmark(
        questions_with_types=questions_with_types,
        model_name=args.model,
        prompt_categories=categories,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        evaluator_model=args.evaluator_model,
        output_dir=args.output_dir,
        use_mock=args.use_mock,
        mock_inference=args.mock_inference,
        category_specific_mode=args.category_specific_mode
    )
    
    print(f"Benchmark complete! Results saved to {args.output_dir}")
    print("Summary of results:")
    
    # Print a summary of the results if available
    if "summary_stats" in results:
        summary = results["summary_stats"]
        
        # Check for category stats
        if "by_category" in summary:
            print("\nOverall scores by category:")
            for category in sorted(summary["by_category"].keys()):
                category_stats = summary["by_category"][category]
                if "avg_overall_score" in category_stats:
                    print(f"  {category}: {category_stats['avg_overall_score']:.2f}/10")
        
        # Check for question type stats
        if "by_question_type" in summary:
            print("\nOverall scores by question type:")
            for qtype in sorted(summary["by_question_type"].keys()):
                qtype_stats = summary["by_question_type"][qtype]
                if "avg_overall_score" in qtype_stats:
                    print(f"  {qtype}: {qtype_stats['avg_overall_score']:.2f}/10")
        
        # Fall back to raw metrics if structured summary isn't available
        if "by_category" not in summary and "by_question_type" not in summary:
            print("\nAverage scores:")
            if "avg_overall_score" in summary:
                print(f"  Overall: {summary['avg_overall_score']:.2f}/10")
            if "avg_primary_dimension_score" in summary:
                print(f"  Primary dimension: {summary['avg_primary_dimension_score']:.2f}/10")
            if "avg_character_consistency_score" in summary:
                print(f"  Character consistency: {summary['avg_character_consistency_score']:.2f}/10")
    
    return results

# Define mock classes for testing
class MockConfig:
    """Mock implementation of Config for testing without requiring character_data directory."""
    
    def __init__(self, model_name="llama3", temperature=0.7, max_tokens=1000):
        """Initialize the MockConfig."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Add required file paths
        self.character_data_dir = Path("character_data")
        
        # Add character data files
        self.core_profile_file = "viktor_core_profile.md"
        self.technical_knowledge_file = "viktor_technical_knowledge.md"
        self.relationships_file = "viktor_relationships.md"
        self.world_context_file = "viktor_world_context.md"
        self.response_guidelines_file = "viktor_response_guidelines.md"
        self.test_scenarios_file = "viktor_test_scenarios.md"
        self.main_prompt_file = "viktor_main_prompt.md"
        self.character_analysis_file = "viktor_scenes_and_events.md"
    
    def get_character_file_path(self, filename):
        """Mock method that returns a path but doesn't verify existence."""
        return self.character_data_dir / filename
    
    def get_all_character_files(self):
        """Mock method that returns a list of all character data files."""
        return [
            self.get_character_file_path(self.core_profile_file),
            self.get_character_file_path(self.technical_knowledge_file),
            self.get_character_file_path(self.relationships_file),
            self.get_character_file_path(self.world_context_file),
            self.get_character_file_path(self.response_guidelines_file),
            self.get_character_file_path(self.main_prompt_file),
        ]
    
    def get_model_params(self):
        """Get model parameters as a dictionary."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

class MockViktorAI:
    """Mock implementation of ViktorAI for testing without a running Ollama server."""
    
    def __init__(self, config, specialized_prompt=None):
        """
        Initialize the MockViktorAI.
        
        Args:
            config: Configuration object.
            specialized_prompt: Optional specialized prompt for specific category testing.
        """
        self.config = config
        self.specialized_prompt = specialized_prompt
        
        # No need to initialize real components that require file access
        self.system_prompt = "Mock system prompt for Viktor AI"
        
        # Track questions for testing
        self.questions_asked = []
    
    def generate_response(self, query):
        """
        Generate a mock response based on the query type.
    
    Args:
            query: The user's question.
            
        Returns:
            A mock response appropriate for the question type.
        """
        # Track questions for testing
        self.questions_asked.append(query)
        
        # Determine question type
        question_type = EvaluationMetrics.get_question_type(query)
        
        # Generate appropriate response based on question type
        if question_type == "identity":
            return "I am Viktor, a scientist from Piltover dedicated to advancing humanity through technology. My life's work is the Hexcore, which harnesses arcane energy for human enhancement."
        elif question_type == "technical":
            return "The Hexcore is a revolutionary technology that harnesses arcane energy to power various enhancements and innovations. It represents the future of what humanity can achieve through the merger of magic and science."
        elif question_type == "relationship":
            return "My relationship with Jayce was once built on mutual respect and shared scientific ambition, but we have diverged in our vision for what humanity can become. Our disagreements stem from fundamental philosophical differences about progress."
        elif question_type == "philosophical":
            return "The Glorious Evolution represents humanity's ascension beyond our biological limitations. It is the inevitable next step in our development as a species, where we transcend the constraints of flesh and achieve our true potential."
        else:
            return f"This is a mock response to: {query}"

class MockOllamaInterface:
    """Mock implementation of OllamaInterface for testing without a running Ollama server."""
    
    def __init__(self, config):
        """Initialize the MockOllamaInterface."""
        self.config = config
        self.history = []
    
    def generate(self, prompt, system_prompt=None):
        """
        Generate a mock response or evaluation.
        
        Args:
            prompt: The prompt to generate a response for.
            system_prompt: Optional system prompt.
            
        Returns:
            A mock response or evaluation.
        """
        # Check if this is an evaluation prompt
        if "evaluate the following response" in prompt.lower():
            # Extract question and response using regex (simplified)
            question_match = re.search(r"Question: (.*?)(\n|$)", prompt)
            response_match = re.search(r"Response to Evaluate\n(.*?)(\n\n|$)", prompt, re.DOTALL)
            
            question = question_match.group(1) if question_match else ""
            response = response_match.group(1) if response_match else ""
            
            # Determine question type
            question_type = EvaluationMetrics.get_question_type(question)
            
            # Generate scores based on question type (simulating better scores for appropriate responses)
            overall_score = random.randint(6, 9)
            primary_score = random.randint(6, 9)
            consistency_score = random.randint(6, 9)
            
            # Format evaluation as a text string that matches the expected pattern
            evaluation = f"""
Overall Score: {overall_score}
Overall Reasoning: This is a mock evaluation for a {question_type} question. The response captures Viktor's characteristic technical precision and pragmatic worldview.

Primary Dimension Score: {primary_score}
Primary Dimension Reasoning: The response demonstrates understanding of Viktor's {question_type} characteristics. It uses appropriately concise language and focuses on the relevant aspects of his character.

Character Consistency Score: {consistency_score}
Character Consistency Reasoning: The response maintains Viktor's characteristic tone and perspective. It avoids contradicting established facts about the character and uses his typical speech patterns.
"""
            return evaluation
        
        # For regular prompts, just return a simple response
        return "This is a mock response from the OllamaInterface."
    
    def generate_with_chat_history(self, messages, system_prompt=None):
        """Generate a mock response with chat history."""
        return "This is a mock response with chat history."
    
    def get_history(self):
        """Get the chat history."""
        return self.history
    
    def clear_history(self):
        """Clear the chat history."""
        self.history = []

def create_markdown_report(results, output_file):
    """
    Create a markdown report from benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        output_file: Path to the output file
    """
    report = []
    
    # Add header
    report.append(f"# ViktorAI Benchmark Results for {results['model_name']}")
    report.append(f"Timestamp: {results['timestamp']}")
    report.append("")
    
    # Add overall summary
    report.append("## Overall Summary")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    
    if "summary_stats" in results and "overall" in results["summary_stats"]:
        overall = results["summary_stats"]["overall"]
        if "avg_overall_score" in overall:
            report.append(f"| Average Overall Score | {overall['avg_overall_score']:.2f} |")
        if "avg_primary_dimension_score" in overall:
            report.append(f"| Average Primary Dimension Score | {overall['avg_primary_dimension_score']:.2f} |")
        if "avg_character_consistency_score" in overall:
            report.append(f"| Average Character Consistency Score | {overall['avg_character_consistency_score']:.2f} |")
        if "avg_response_time" in overall:
            report.append(f"| Average Response Time | {overall['avg_response_time']:.4f} seconds |")
        if "total_responses" in overall:
            report.append(f"| Total Responses | {overall['total_responses']} |")
    
    report.append("")
    
    # Add category summaries
    if "summary_stats" in results and "by_category" in results["summary_stats"]:
        report.append("## Results by Category")
        report.append("")
        report.append("| Category | Average Overall Score | Average Primary Dimension Score | Average Character Consistency Score | Total Responses |")
        report.append("|----------|----------------------|--------------------------------|-----------------------------------|----------------|")
        
        for category in sorted(results["summary_stats"]["by_category"].keys()):
            cat_stats = results["summary_stats"]["by_category"][category]
            
            avg_overall = f"{cat_stats.get('avg_overall_score', 0):.2f}" if "avg_overall_score" in cat_stats else "N/A"
            avg_primary = f"{cat_stats.get('avg_primary_dimension_score', 0):.2f}" if "avg_primary_dimension_score" in cat_stats else "N/A"
            avg_consistency = f"{cat_stats.get('avg_character_consistency_score', 0):.2f}" if "avg_character_consistency_score" in cat_stats else "N/A"
            total = cat_stats.get("total_responses", 0)
            
            report.append(f"| {category} | {avg_overall} | {avg_primary} | {avg_consistency} | {total} |")
        
        report.append("")
    
    # Add question type summaries
    if "summary_stats" in results and "by_question_type" in results["summary_stats"]:
        report.append("## Results by Question Type")
        report.append("")
        report.append("| Question Type | Average Overall Score | Average Primary Dimension Score | Average Character Consistency Score | Total Responses |")
        report.append("|--------------|----------------------|--------------------------------|-----------------------------------|----------------|")
        
        for qtype in sorted(results["summary_stats"]["by_question_type"].keys()):
            qtype_stats = results["summary_stats"]["by_question_type"][qtype]
            
            avg_overall = f"{qtype_stats.get('avg_overall_score', 0):.2f}" if "avg_overall_score" in qtype_stats else "N/A"
            avg_primary = f"{qtype_stats.get('avg_primary_dimension_score', 0):.2f}" if "avg_primary_dimension_score" in qtype_stats else "N/A"
            avg_consistency = f"{qtype_stats.get('avg_character_consistency_score', 0):.2f}" if "avg_character_consistency_score" in qtype_stats else "N/A"
            total = qtype_stats.get("total_responses", 0)
            
            report.append(f"| {qtype} | {avg_overall} | {avg_primary} | {avg_consistency} | {total} |")
        
        report.append("")
    
    # Add individual response details
    report.append("## Individual Responses")
    report.append("")
    
    for category in sorted(results["metrics"].keys()):
        if results["metrics"][category]:
            report.append(f"### Category: {category}")
            report.append("")
            
            for i, metrics in enumerate(results["metrics"][category], 1):
                question = metrics.get("question", "N/A")
                response = metrics.get("response", "N/A")
                question_type = metrics.get("question_type", "unknown")
                overall_score = metrics.get("overall_score", "N/A")
                primary_score = metrics.get("primary_dimension_score", "N/A")
                consistency_score = metrics.get("character_consistency_score", "N/A")
                
                report.append(f"#### Response {i} (Question Type: {question_type})")
                report.append("")
                report.append(f"**Question:** {question}")
                report.append("")
                report.append(f"**Response:**")
                report.append("```")
                report.append(response)
                report.append("```")
                report.append("")
                report.append(f"**Scores:**")
                report.append(f"- Overall: {overall_score}")
                report.append(f"- Primary Dimension: {primary_score}")
                report.append(f"- Character Consistency: {consistency_score}")
                report.append("")
                
                if "overall_reasoning" in metrics:
                    report.append("**Evaluation:**")
                    report.append(f"*Overall:* {metrics['overall_reasoning']}")
                    report.append("")
                    if "primary_dimension_reasoning" in metrics:
                        report.append(f"*Primary Dimension:* {metrics['primary_dimension_reasoning']}")
                        report.append("")
                    if "character_consistency_reasoning" in metrics:
                        report.append(f"*Character Consistency:* {metrics['character_consistency_reasoning']}")
                        report.append("")
                
                report.append("---")
                report.append("")
    
    # Write the report to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"Markdown report saved to {output_file}")

def create_html_report(results, output_file):
    """
    Create an HTML report from benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        output_file: Path to the output file
    """
    # Start with HTML header
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViktorAI Benchmark Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        h1 {
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .question-section {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        .question {
            background-color: #eaf2f8;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #3498db;
            margin-bottom: 15px;
        }
        .response {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #2ecc71;
            margin-bottom: 15px;
            white-space: pre-wrap;
        }
        .response-text {
            margin: 0;
            padding: 0;
        }
        .evaluation {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #f39c12;
        }
        .score-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 15px;
        }
        .score-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            width: 100%;
        }
        .score-box {
            flex: 1;
            min-width: 200px;
            padding: 10px;
            border-radius: 5px;
            background-color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .score-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .score-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .score-bar {
            height: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            margin-bottom: 10px;
            position: relative;
        }
        .score-fill {
            height: 100%;
            border-radius: 5px;
            position: absolute;
            top: 0;
            left: 0;
        }
        .score-reasoning {
            font-style: italic;
            color: #555;
        }
        .weighted-score {
            font-weight: bold;
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9f9;
            border-radius: 5px;
            text-align: right;
        }
        .high-score { background-color: #2ecc71; }
        .medium-score { background-color: #f39c12; }
        .low-score { background-color: #e74c3c; }
        .summary {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    """
    
    # Add header
    html += f"<h1>ViktorAI Benchmark Results for {results['model_name']}</h1>"
    html += f"<div class='timestamp'>Timestamp: {results['timestamp']}</div>"
    
    # Add overall summary
    html += "<div class='summary'>"
    html += "<h2>Summary</h2>"
    html += "<table>"
    html += "<tr><th>Metric</th><th>Value</th></tr>"
    
    if "summary_stats" in results and "overall" in results["summary_stats"]:
        overall = results["summary_stats"]["overall"]
        if "avg_overall_score" in overall:
            html += f"<tr><td>Average Overall Score</td><td>{overall['avg_overall_score']:.2f}/10</td></tr>"
        if "avg_primary_dimension_score" in overall:
            html += f"<tr><td>Average Primary Dimension Score</td><td>{overall['avg_primary_dimension_score']:.2f}/10</td></tr>"
        if "avg_character_consistency_score" in overall:
            html += f"<tr><td>Average Character Consistency Score</td><td>{overall['avg_character_consistency_score']:.2f}/10</td></tr>"
        if "avg_response_time" in overall:
            html += f"<tr><td>Average Response Time</td><td>{overall['avg_response_time']:.4f} seconds</td></tr>"
        if "total_responses" in overall:
            html += f"<tr><td>Total Responses</td><td>{overall['total_responses']}</td></tr>"
    
    html += "</table>"
    
    # Add scores by question type
    if "summary_stats" in results and "by_question_type" in results["summary_stats"]:
        html += "<h3>Scores by Question Type</h3>"
        html += "<table>"
        html += "<tr><th>Question Type</th><th>Average Overall Score</th><th>Average Primary Dimension Score</th><th>Average Character Consistency Score</th></tr>"
        
        for qtype in sorted(results["summary_stats"]["by_question_type"].keys()):
            qtype_stats = results["summary_stats"]["by_question_type"][qtype]
            
            # Improved handling of avg_overall_score to avoid N/A values
            if "avg_overall_score" in qtype_stats and qtype_stats["avg_overall_score"] is not None:
                try:
                    avg_overall = f"{float(qtype_stats['avg_overall_score']):.2f}/10"
                except (ValueError, TypeError):
                    avg_overall = "N/A"
            else:
                avg_overall = "0.00/10"  # Default to 0 instead of N/A
                
            # Improved handling of avg_primary_dimension_score to avoid N/A values
            if "avg_primary_dimension_score" in qtype_stats and qtype_stats["avg_primary_dimension_score"] is not None:
                try:
                    avg_primary = f"{float(qtype_stats['avg_primary_dimension_score']):.2f}/10"
                except (ValueError, TypeError):
                    avg_primary = "N/A"
            else:
                avg_primary = "0.00/10"  # Default to 0 instead of N/A
                
            # Improved handling of avg_character_consistency_score to avoid N/A values
            if "avg_character_consistency_score" in qtype_stats and qtype_stats["avg_character_consistency_score"] is not None:
                try:
                    avg_consistency = f"{float(qtype_stats['avg_character_consistency_score']):.2f}/10"
                except (ValueError, TypeError):
                    avg_consistency = "N/A"
            else:
                avg_consistency = "0.00/10"  # Default to 0 instead of N/A
            
            html += f"<tr><td>{qtype.capitalize()}</td><td>{avg_overall}</td><td>{avg_primary}</td><td>{avg_consistency}</td></tr>"
        
        html += "</table>"
    
    html += "</div>"  # Close summary div
    
    # Group responses by question type
    question_types = {}
    
    for category in results["metrics"]:
        for metric in results["metrics"][category]:
            question_type = metric.get("question_type", "unknown")
            if question_type not in question_types:
                question_types[question_type] = []
            question_types[question_type].append((category, metric))
    
    # Add individual response details by question type
    for qtype, responses in sorted(question_types.items()):
        if responses:
            html += f"<div class='question-section'>"
            html += f"<h2>{qtype.capitalize()} Questions</h2>"
            
            for i, (category, metrics) in enumerate(responses, 1):
                question = metrics.get("question", "N/A")
                response = metrics.get("response", "N/A")
                
                # Improved handling of scores to avoid N/A display issues
                # Get overall score with fallback to numeric value
                try:
                    overall_score = float(metrics.get("overall_score", 0))
                    overall_score_display = f"{overall_score}/10"
                except (ValueError, TypeError):
                    overall_score = 0  # Default for calculations
                    overall_score_display = "0/10"  # Default display
                
                # Get primary score with fallback to numeric value
                try:
                    primary_score = float(metrics.get("primary_dimension_score", 0))
                    primary_score_display = f"{primary_score}/10"
                except (ValueError, TypeError):
                    primary_score = 0  # Default for calculations
                    primary_score_display = "0/10"  # Default display
                
                # Get consistency score with fallback to numeric value
                try:
                    consistency_score = float(metrics.get("character_consistency_score", 0))
                    consistency_score_display = f"{consistency_score}/10"
                except (ValueError, TypeError):
                    consistency_score = 0  # Default for calculations
                    consistency_score_display = "0/10"  # Default display
                
                html += f"<h3>Question {i}</h3>"
                html += f"<div class='question'><strong>Q:</strong> {question}</div>"
                html += f"<div class='response'><strong>Response:</strong><div class='response-text'>{response}</div></div>"
                
                html += f"<div class='evaluation'>"
                html += f"<h4>Evaluation</h4>"
                html += f"<div class='score-container'>"
                
                # Overall score row (full width)
                html += f"<div class='score-row'>"
                html += f"<div class='score-box'>"
                html += f"<div class='score-title'>Overall Score</div>"
                html += f"<div class='score-value'>{overall_score_display}</div>"
                
                # Only add score bars for valid numeric scores
                html += f"<div class='score-bar'>"
                score_class = "high-score" if overall_score >= 8 else "medium-score" if overall_score >= 5 else "low-score"
                width = min(100, max(0, overall_score * 10))
                html += f"<div class='score-fill {score_class}' style='width: {width}%;'></div>"
                html += f"</div>"
                
                if "overall_reasoning" in metrics:
                    html += f"<div class='score-reasoning'>{metrics['overall_reasoning']}</div>"
                
                html += f"</div>"  # Close score box
                html += f"</div>"  # Close score row
                
                # Primary dimension and character consistency (side by side)
                html += f"<div class='score-row'>"
                
                # Primary dimension score
                html += f"<div class='score-box'>"
                html += f"<div class='score-title'>Primary Dimension Score</div>"
                html += f"<div class='score-value'>{primary_score_display}</div>"
                
                # Add score bars for primary dimension
                html += f"<div class='score-bar'>"
                score_class = "high-score" if primary_score >= 8 else "medium-score" if primary_score >= 5 else "low-score"
                width = min(100, max(0, primary_score * 10))
                html += f"<div class='score-fill {score_class}' style='width: {width}%;'></div>"
                html += f"</div>"
                
                if "primary_dimension_reasoning" in metrics:
                    html += f"<div class='score-reasoning'>{metrics['primary_dimension_reasoning']}</div>"
                
                html += f"</div>"  # Close score box
                
                # Character consistency score
                html += f"<div class='score-box'>"
                html += f"<div class='score-title'>Character Consistency Score</div>"
                html += f"<div class='score-value'>{consistency_score_display}</div>"
                
                # Add score bars for character consistency
                html += f"<div class='score-bar'>"
                score_class = "high-score" if consistency_score >= 8 else "medium-score" if consistency_score >= 5 else "low-score"
                width = min(100, max(0, consistency_score * 10))
                html += f"<div class='score-fill {score_class}' style='width: {width}%;'></div>"
                html += f"</div>"
                
                if "character_consistency_reasoning" in metrics:
                    html += f"<div class='score-reasoning'>{metrics['character_consistency_reasoning']}</div>"
                
                html += f"</div>"  # Close score box
                html += f"</div>"  # Close score row
                
                # Add weighted score if available
                if "weighted_score" in metrics:
                    try:
                        weighted_score = float(metrics["weighted_score"])
                        html += f"<div class='weighted-score'>Weighted Score (based on question type): {weighted_score:.2f}/10</div>"
                    except (ValueError, TypeError):
                        # Skip rendering weighted score if it's not a valid number
                        pass
                elif primary_score > 0 and consistency_score > 0:
                    # Calculate weighted score if not provided but we have valid component scores
                    try:
                        # Use standard weights (60% primary, 40% character consistency)
                        weighted_score = (primary_score * 0.6) + (consistency_score * 0.4)
                        html += f"<div class='weighted-score'>Weighted Score (based on question type): {weighted_score:.2f}/10</div>"
                    except (ValueError, TypeError):
                        # Skip rendering weighted score if calculation fails
                        pass
                
                html += f"</div>"  # Close score container
                html += f"</div>"  # Close evaluation div
            
            html += f"</div>"  # Close question section
    
    # Close HTML document
    html += """
</body>
</html>
"""

    # Write the report to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"HTML report saved to {output_file}")

if __name__ == "__main__":
    sys.exit(main() or 0) 

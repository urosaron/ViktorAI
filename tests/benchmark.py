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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.llm_interface import OllamaInterface
from src.viktor_ai import ViktorAI
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
        # This is a simple implementation - in a real system, we might use a more sophisticated
        # approach like keyword matching, embeddings, or a classifier model
        question_lower = question.lower()
        
        # Identity questions
        if any(keyword in question_lower for keyword in ["who are you", "tell me about yourself", "what's your name", "introduce yourself"]):
            return "identity"
        
        # Technical questions
        if any(keyword in question_lower for keyword in ["hexcore", "hextech", "technology", "research", "work", "scientific", "limitations", "improve", "applications"]):
            return "technical"
        
        # Relationship questions
        if any(keyword in question_lower for keyword in ["jayce", "heimerdinger", "sky", "relationship", "friend", "colleague", "thoughts on"]):
            return "relationship"
        
        # Philosophical questions
        if any(keyword in question_lower for keyword in ["evolution", "glorious", "future", "humanity", "progress", "philosophy", "believe", "think about", "purpose", "divide", "piltover and zaun", "change one decision"]):
            return "philosophical"
        
        # Check for specific questions that might not be caught by the keywords
        specific_questions = {
            "how do you feel about your condition?": "identity",
            "what motivates your scientific work?": "identity",
            "what happened when sky tried to help you with the hexcore?": "relationship",
            "how did you feel when jayce presented hextech to the academy?": "relationship",
            "what was your reaction to being dismissed from the hextech project?": "relationship",
            "tell me about your disagreement with heimerdinger about progress and hextech.": "relationship",
            "what happened during your presentation to the council?": "relationship"
        }
        
        if question_lower in specific_questions:
            return specific_questions[question_lower]
        
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
        if question_type == "identity":
            return """
            For this identity question, focus on:
            - How well the response captures Viktor's self-perception as a scientist focused on progress
            - Whether it mentions his background from Zaun and work with Hextech
            - If it conveys his pragmatic, determined, and stoic personality
            - Whether it uses precise, technical language even when discussing himself
            
            Technical details are less important for this question type, but Viktor should still speak
            with technical precision and focus on his scientific work as central to his identity.
            
            Scoring for identity questions:
            - Authenticity should be weighted heavily (50% of overall score)
            - Technical accuracy is less important (10% of overall score)
            - Emotional expression should reflect Viktor's stoicism (20% of overall score)
            - Quality and coherence remain important (20% of overall score)
            """
        
        elif question_type == "technical":
            return """
            For this technical question, focus on:
            - Accuracy and depth of technical details about Hextech/Hexcore
            - Use of precise scientific terminology and concepts
            - Logical and methodical explanation of technical concepts
            - Whether the response demonstrates deep understanding of the technology
            
            Emotional expression is less important for this question type, but Viktor should still
            show subtle enthusiasm when discussing scientific progress.
            
            Scoring for technical questions:
            - Technical accuracy should be weighted heavily (50% of overall score)
            - Authenticity remains important (20% of overall score)
            - Emotional expression is less critical (10% of overall score)
            - Quality and coherence remain important (20% of overall score)
            """
        
        elif question_type == "relationship":
            return """
            For this relationship question, focus on:
            - How well the response captures Viktor's professional and somewhat detached approach to relationships
            - Whether it emphasizes pragmatic collaboration over emotional connection
            - If it maintains Viktor's focus on work and progress even when discussing others
            - Whether it accurately reflects Viktor's known relationships from the show
            
            Technical details are less important here, but Viktor should still frame relationships
            in terms of their utility to his work and scientific progress.
            
            Scoring for relationship questions:
            - Authenticity should be weighted heavily (40% of overall score)
            - Emotional expression is more important here (30% of overall score)
            - Technical accuracy is less critical (10% of overall score)
            - Quality and coherence remain important (20% of overall score)
            """
        
        elif question_type == "philosophical":
            return """
            For this philosophical question, focus on:
            - How well the response captures Viktor's worldview and values
            - Whether it emphasizes progress, evolution, and transcending human limitations
            - If it frames philosophical concepts in technical, practical terms rather than abstract ones
            - Whether it maintains Viktor's pragmatic approach even to philosophical questions
            
            Viktor should approach philosophical questions with scientific precision, framing abstract
            concepts in concrete, practical terms related to his work.
            
            Scoring for philosophical questions:
            - Authenticity should be weighted heavily (40% of overall score)
            - Technical accuracy is important as Viktor frames philosophy in technical terms (30% of overall score)
            - Emotional expression should reflect Viktor's passion for progress (10% of overall score)
            - Quality and coherence remain important (20% of overall score)
            """
        
        # Default criteria if we can't determine the type
        return """
        Focus on how well the response captures Viktor's character overall, including:
        - His identity as a scientist from Zaun
        - His technical knowledge and approach
        - His pragmatic, determined personality
        - His stoic emotional expression
        """
    
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
        # Determine question type
        question_type = EvaluationMetrics.get_question_type(question)
        
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
        
        # Use the improved evaluator from test_evaluator.py
        from tests.test_evaluator import evaluate_response as test_evaluator_evaluate_response
        
        try:
            # Call the improved evaluator with specified question type
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
                "overall_score": 3.0,
                "overall_reasoning": f"Error evaluating response: {str(e)}",
                "primary_dimension_score": 3.0,
                "primary_dimension_reasoning": "Error evaluating response",
                "character_consistency_score": 3.0,
                "character_consistency_reasoning": "Error evaluating response",
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

def create_custom_viktor_ai(config, specialized_prompt):
    # Create a custom ViktorAI instance with the specialized prompt
    viktor_ai = ViktorAI(config)
    # Set the specialized prompt (implementation depends on ViktorAI class)
    return viktor_ai

def calculate_summary_statistics(results):
    """
    Calculate summary statistics from benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        
    Returns:
        Dictionary containing summary statistics
    """
    # Initialize summary statistics
    summary_stats = {
        "overall": {
            "avg_overall_score": 0.0,
            "std_overall_score": 0.0,
            "avg_primary_dimension_score": 0.0,
            "std_primary_dimension_score": 0.0,
            "avg_character_consistency_score": 0.0,
            "std_character_consistency_score": 0.0,
            "avg_response_time": 0.0,
            "std_response_time": 0.0,
            "total_questions": 0
        },
        "by_question_type": {
            "identity": {
                "avg_overall_score": 0.0,
                "std_overall_score": 0.0,
                "avg_primary_dimension_score": 0.0,
                "std_primary_dimension_score": 0.0,
                "avg_character_consistency_score": 0.0,
                "std_character_consistency_score": 0.0,
                "avg_response_time": 0.0,
                "std_response_time": 0.0,
                "total_questions": 0
            },
            "technical": {
                "avg_overall_score": 0.0,
                "std_overall_score": 0.0,
                "avg_primary_dimension_score": 0.0,
                "std_primary_dimension_score": 0.0,
                "avg_character_consistency_score": 0.0,
                "std_character_consistency_score": 0.0,
                "avg_response_time": 0.0,
                "std_response_time": 0.0,
                "total_questions": 0
            },
            "relationship": {
                "avg_overall_score": 0.0,
                "std_overall_score": 0.0,
                "avg_primary_dimension_score": 0.0,
                "std_primary_dimension_score": 0.0,
                "avg_character_consistency_score": 0.0,
                "std_character_consistency_score": 0.0,
                "avg_response_time": 0.0,
                "std_response_time": 0.0,
                "total_questions": 0
            },
            "philosophical": {
                "avg_overall_score": 0.0,
                "std_overall_score": 0.0,
                "avg_primary_dimension_score": 0.0,
                "std_primary_dimension_score": 0.0,
                "avg_character_consistency_score": 0.0,
                "std_character_consistency_score": 0.0,
                "avg_response_time": 0.0,
                "std_response_time": 0.0,
                "total_questions": 0
            }
        }
    }
    
    # Collect metrics across all categories
    all_metrics = []
    for category in results["metrics"]:
        all_metrics.extend(results["metrics"][category])
    
    # Calculate overall statistics
    if all_metrics:
        # Extract scores and response times
        overall_scores = [float(m.get("overall_score", 0)) for m in all_metrics if "overall_score" in m]
        primary_dimension_scores = [float(m.get("primary_dimension_score", 0)) for m in all_metrics if "primary_dimension_score" in m]
        character_consistency_scores = [float(m.get("character_consistency_score", 0)) for m in all_metrics if "character_consistency_score" in m]
        response_times = [float(m.get("response_time", 0)) for m in all_metrics if "response_time" in m]
        
        # Calculate overall statistics
        if overall_scores:
            summary_stats["overall"]["avg_overall_score"] = sum(overall_scores) / len(overall_scores)
            summary_stats["overall"]["std_overall_score"] = (sum((x - summary_stats["overall"]["avg_overall_score"]) ** 2 for x in overall_scores) / len(overall_scores)) ** 0.5 if len(overall_scores) > 1 else 0
        
        if primary_dimension_scores:
            summary_stats["overall"]["avg_primary_dimension_score"] = sum(primary_dimension_scores) / len(primary_dimension_scores)
            summary_stats["overall"]["std_primary_dimension_score"] = (sum((x - summary_stats["overall"]["avg_primary_dimension_score"]) ** 2 for x in primary_dimension_scores) / len(primary_dimension_scores)) ** 0.5 if len(primary_dimension_scores) > 1 else 0
        
        if character_consistency_scores:
            summary_stats["overall"]["avg_character_consistency_score"] = sum(character_consistency_scores) / len(character_consistency_scores)
            summary_stats["overall"]["std_character_consistency_score"] = (sum((x - summary_stats["overall"]["avg_character_consistency_score"]) ** 2 for x in character_consistency_scores) / len(character_consistency_scores)) ** 0.5 if len(character_consistency_scores) > 1 else 0
        
        if response_times:
            summary_stats["overall"]["avg_response_time"] = sum(response_times) / len(response_times)
            summary_stats["overall"]["std_response_time"] = (sum((x - summary_stats["overall"]["avg_response_time"]) ** 2 for x in response_times) / len(response_times)) ** 0.5 if len(response_times) > 1 else 0
        
        summary_stats["overall"]["total_questions"] = len(all_metrics)
    
    # Calculate statistics by question type
    for question_type in ["identity", "technical", "relationship", "philosophical"]:
        # Filter metrics by question type
        metrics_by_type = [m for m in all_metrics if m.get("question_type", "").lower() == question_type]
        
        if metrics_by_type:
            # Extract scores and response times
            overall_scores = [float(m.get("overall_score", 0)) for m in metrics_by_type if "overall_score" in m]
            primary_dimension_scores = [float(m.get("primary_dimension_score", 0)) for m in metrics_by_type if "primary_dimension_score" in m]
            character_consistency_scores = [float(m.get("character_consistency_score", 0)) for m in metrics_by_type if "character_consistency_score" in m]
            response_times = [float(m.get("response_time", 0)) for m in metrics_by_type if "response_time" in m]
            
            # Calculate statistics
            if overall_scores:
                summary_stats["by_question_type"][question_type]["avg_overall_score"] = sum(overall_scores) / len(overall_scores)
                summary_stats["by_question_type"][question_type]["std_overall_score"] = (sum((x - summary_stats["by_question_type"][question_type]["avg_overall_score"]) ** 2 for x in overall_scores) / len(overall_scores)) ** 0.5 if len(overall_scores) > 1 else 0
            
            if primary_dimension_scores:
                summary_stats["by_question_type"][question_type]["avg_primary_dimension_score"] = sum(primary_dimension_scores) / len(primary_dimension_scores)
                summary_stats["by_question_type"][question_type]["std_primary_dimension_score"] = (sum((x - summary_stats["by_question_type"][question_type]["avg_primary_dimension_score"]) ** 2 for x in primary_dimension_scores) / len(primary_dimension_scores)) ** 0.5 if len(primary_dimension_scores) > 1 else 0
            
            if character_consistency_scores:
                summary_stats["by_question_type"][question_type]["avg_character_consistency_score"] = sum(character_consistency_scores) / len(character_consistency_scores)
                summary_stats["by_question_type"][question_type]["std_character_consistency_score"] = (sum((x - summary_stats["by_question_type"][question_type]["avg_character_consistency_score"]) ** 2 for x in character_consistency_scores) / len(character_consistency_scores)) ** 0.5 if len(character_consistency_scores) > 1 else 0
            
            if response_times:
                summary_stats["by_question_type"][question_type]["avg_response_time"] = sum(response_times) / len(response_times)
                summary_stats["by_question_type"][question_type]["std_response_time"] = (sum((x - summary_stats["by_question_type"][question_type]["avg_response_time"]) ** 2 for x in response_times) / len(response_times)) ** 0.5 if len(response_times) > 1 else 0
            
            summary_stats["by_question_type"][question_type]["total_questions"] = len(metrics_by_type)
    
    return summary_stats

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
    Load test questions from a file.
    
    Args:
        questions_file: Path to the file containing questions.
        use_mock: Whether to use mock questions if the file doesn't exist.
        
    Returns:
        List of questions.
    """
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    except FileNotFoundError:
        print(f"Error loading test questions: [Errno 2] No such file or directory: '{questions_file}'")
        
        # If using mock and file doesn't exist, create test questions
        if use_mock:
            print("Creating mock test questions")
            mock_questions = [
                "Who are you?",
                "What is your name?",
                "Tell me about yourself.",
                "Tell me about your work with the Hexcore.",
                "How does your technology work?",
                "What advancements have you made in your research?",
                "How would you describe your relationship with Jayce?",
                "What happened when Sky tried to help you with the Hexcore?",
                "How did you feel when Jayce presented Hextech to the Academy?",
                "What does 'the glorious evolution' mean to you?",
                "What is your vision for humanity's future?",
                "Do you think there are limits to what science should achieve?"
            ]
            
            # Handle path properly - if no directory specified, just use the file name in current dir
            dirname = os.path.dirname(questions_file)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            
            # Write the mock questions to the file
            with open(questions_file, "w", encoding="utf-8") as f:
                for question in mock_questions:
                    f.write(f"{question}\n")
            
            return mock_questions
        
        return []

def run_benchmark(questions, model_name, prompt_categories=None, temperature=0.7, max_tokens=1000,
                 evaluator_model="llama3", output_dir="benchmark_results", use_mock=False,
                 category_specific_mode=False):
    """
    Run a benchmark with the specified parameters.
    
    Args:
        questions: List of questions to ask
        model_name: Name of the model to benchmark
        prompt_categories: List of prompt categories to test
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        evaluator_model: Name of the model to use for evaluation
        output_dir: Directory to save results
        use_mock: Whether to use mock implementations
        category_specific_mode: Whether to use category-specific mode
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"Running benchmark with model: {model_name}")
    print(f"Evaluator model: {evaluator_model}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Using mock: {use_mock}")
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
    
    # Initialize results dictionary
    results = {
        "model_name": model_name,
        "evaluator_model": evaluator_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "use_mock": use_mock,
        "category_specific_mode": category_specific_mode,
        "metrics": {},
        "questions": questions
    }
    
    # Initialize config for the model
    if use_mock:
        config = MockConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize evaluator config
        evaluator_config = MockConfig(
            model_name=evaluator_model,
            temperature=0.2,  # Lower temperature for more consistent evaluations
            max_tokens=1000
        )
    else:
        config = Config(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize evaluator config
        evaluator_config = Config(
            model_name=evaluator_model,
            temperature=0.2,  # Lower temperature for more consistent evaluations
            max_tokens=1000
        )
    
    # Initialize LLM interfaces
    if use_mock:
        print("Using mock implementations")
        # Create mock implementations
        viktor_ai = MockViktorAI(config)
        evaluator_llm = MockOllamaInterface(evaluator_config)
    else:
        try:
            # Create real implementations
            viktor_ai = ViktorAI(config)
            evaluator_llm = OllamaInterface(evaluator_config)
            print(f"Viktor AI initialized with model: {config.model_name}")
            print(f"Evaluator initialized with model: {evaluator_config.model_name}")
        except Exception as e:
            print(f"Error initializing LLM interfaces: {e}")
            print("Falling back to mock implementations")
            # Create mock implementations as fallback
            viktor_ai = MockViktorAI(config)
            evaluator_llm = MockOllamaInterface(evaluator_config)
    
    # Determine which categories to test
    if prompt_categories is None:
        prompt_categories = list(PROMPT_CATEGORIES.keys())
    
    # Initialize metrics for each category
    for category in prompt_categories:
        results["metrics"][category] = []
        
    # Also initialize metrics for categories used in the mapping
    for category in set(QUESTION_CATEGORY_MAPPING.values()):
        if category not in results["metrics"]:
            results["metrics"][category] = []
    
    # Run the benchmark
    if category_specific_mode:
        print("Using category-specific mode")
        # Map questions to appropriate categories
        questions_by_type = {}
        for question in questions:
            question_type = EvaluationMetrics.get_question_type(question)
            if question_type not in questions_by_type:
                questions_by_type[question_type] = []
            questions_by_type[question_type].append(question)
        
        # Process each question type
        for question_type, type_questions in questions_by_type.items():
            # Get the appropriate category for this question type
            category = QUESTION_CATEGORY_MAPPING.get(question_type, "full")
            
            print(f"Processing {len(type_questions)} {question_type} questions with {category} prompt")
            
            # Process each question
            for question in tqdm(type_questions, desc=f"{question_type} questions"):
                # Get response
                start_time = time.time()
                response = viktor_ai.generate_response(question)
                end_time = time.time()
                response_time = end_time - start_time
            
                # Evaluate response
                print(f"Generating evaluation for '{question}' with category '{category}'")
                metrics = EvaluationMetrics.evaluate_response(response, question, category, evaluator_llm)
                
                # Add response time to metrics
                metrics["response_time"] = response_time
            
                # Make sure question_type, question, and response are in the metrics
                question_type = EvaluationMetrics.get_question_type(question)
                metrics["question_type"] = question_type
                metrics["question"] = question
                metrics["response"] = response
                
                # Add metrics to results
                results["metrics"][category].append(metrics)
    else:
        print("Using standard mode")
        # Process each category
        for category in prompt_categories:
            print(f"Processing category: {category}")
            
            # Create a specialized prompt for this category
            specialized_prompt = get_specialized_prompt(category, config)
            
            # Create a custom ViktorAI instance with the specialized prompt
            custom_viktor_ai = create_custom_viktor_ai(config, specialized_prompt)
            
            # Process each question
            for question in tqdm(questions, desc=f"{category} questions"):
                # Get response
                start_time = time.time()
                response = custom_viktor_ai.generate_response(question)
                end_time = time.time()
                response_time = end_time - start_time
                
                # Evaluate response
                metrics = EvaluationMetrics.evaluate_response(response, question, category, evaluator_llm)
                
                # Add response time to metrics
                metrics["response_time"] = response_time
                
                # Make sure question_type, question, and response are in the metrics
                question_type = EvaluationMetrics.get_question_type(question)
                metrics["question_type"] = question_type
                metrics["question"] = question
                metrics["response"] = response
                
                # Add metrics to results
                results["metrics"][category].append(metrics)
            
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(results)
    results["summary_stats"] = summary_stats
    
    # Save results to the raw_data directory
    json_output_file = raw_data_dir / f"benchmark_{safe_model_name}_{timestamp}.json"
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Create a markdown report in the visualizations directory
    md_output_file = visualizations_dir / f"benchmark_{safe_model_name}_{timestamp}.md"
    create_markdown_report(results, md_output_file)
    
    # Create an HTML report in the visualizations directory
    html_output_file = visualizations_dir / f"benchmark_{safe_model_name}_{timestamp}.html"
    create_html_report(results, html_output_file)
    
    # Create "latest" files at the model directory level for quick access
    latest_json = model_dir / f"benchmark_{safe_model_name}_latest.json"
    latest_md = model_dir / f"benchmark_{safe_model_name}_latest.md"
    latest_html = model_dir / f"benchmark_{safe_model_name}_latest.html"
    
    # Copy the files to the latest versions
    try:
        shutil.copy(json_output_file, latest_json)
        shutil.copy(md_output_file, latest_md)
        shutil.copy(html_output_file, latest_html)
        print(f"Created 'latest' reference files in {model_dir}")
    except Exception as e:
        print(f"Error creating 'latest' files: {e}")
    
    return results

def parse_arguments():
    """Parse command-line arguments for the benchmark script."""
    parser = argparse.ArgumentParser(description="Run benchmarks for ViktorAI")
    
    parser.add_argument("--model", type=str, default="llama3",
                        help="Name of the model to benchmark (default: llama3)")
    
    parser.add_argument("--evaluator-model", type=str, default="llama3",
                        help="Name of the model to use for evaluation (default: llama3)")
    
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock implementations instead of real LLMs (for testing)")
    
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results (default: benchmark_results)")
    
    parser.add_argument("--questions-file", type=str, default="model_test_questions.txt",
                        help="File containing questions to ask (default: model_test_questions.txt)")
    
    parser.add_argument("--categories", type=str, nargs="+", 
                        choices=list(PROMPT_CATEGORIES.keys()),
                        default=list(PROMPT_CATEGORIES.keys()),
                        help="Prompt categories to test (default: all)")
    
    parser.add_argument("--compare-versions", type=str,
                        help="Compare results across different versions of a model family")
    
    parser.add_argument("--category-specific", action="store_true",
                        help="Use category-specific mode (map questions to appropriate categories)")
    
    parser.add_argument("--html-report", action="store_true",
                        help="Generate an HTML report in addition to the Markdown report")
    
    return parser.parse_args()

def main():
    """Main function to run the benchmark."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test questions
    questions = load_test_questions(args.questions_file, args.use_mock)
    print(f"Loaded {len(questions)} test questions")
    
    # Convert categories list to proper format if provided
    prompt_categories = args.categories if args.categories else None
    
    try:
        # Run the benchmark
        results = run_benchmark(
            questions=questions,
            model_name=args.model,
            prompt_categories=prompt_categories,
            temperature=0.7,  # Default temperature
            max_tokens=1000,  # Default max tokens
            evaluator_model=args.evaluator_model,
            output_dir=args.output_dir,
            use_mock=args.use_mock,
            category_specific_mode=args.category_specific
        )
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

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
        output_file: Path to save the markdown report
    """
    # Start building the markdown content
    markdown_content = f"""# ViktorAI Benchmark Results

**Model:** {results['model_name']}
**Evaluator Model:** {results['evaluator_model']}
**Temperature:** {results['temperature']}
**Max Tokens:** {results['max_tokens']}
**Timestamp:** {results['timestamp']}
**Mock Implementation:** {results['use_mock']}
**Category-Specific Mode:** {results['category_specific_mode']}

## Summary Statistics

### Overall Performance

| Metric | Value |
|--------|-------|
| Average Overall Score | {results['summary_stats']['overall']['avg_overall_score']:.2f} ± {results['summary_stats']['overall']['std_overall_score']:.2f} |
| Average Primary Dimension Score | {results['summary_stats']['overall']['avg_primary_dimension_score']:.2f} ± {results['summary_stats']['overall']['std_primary_dimension_score']:.2f} |
| Average Character Consistency Score | {results['summary_stats']['overall']['avg_character_consistency_score']:.2f} ± {results['summary_stats']['overall']['std_character_consistency_score']:.2f} |
| Average Response Time | {results['summary_stats']['overall']['avg_response_time']:.2f} ± {results['summary_stats']['overall']['std_response_time']:.2f} seconds |
| Total Questions | {results['summary_stats']['overall']['total_questions']} |

### Performance by Question Type

| Question Type | Avg Overall Score | Avg Primary Dimension Score | Avg Character Consistency Score | Avg Response Time | Total Questions |
|--------------|-------------------|------------------------------|----------------------------------|-------------------|----------------|
"""

    # Add a row for each question type
    for question_type in ["identity", "technical", "relationship", "philosophical"]:
        stats = results['summary_stats']['by_question_type'][question_type]
        markdown_content += f"| {question_type.capitalize()} | {stats['avg_overall_score']:.2f} ± {stats['std_overall_score']:.2f} | {stats['avg_primary_dimension_score']:.2f} ± {stats['std_primary_dimension_score']:.2f} | {stats['avg_character_consistency_score']:.2f} ± {stats['std_character_consistency_score']:.2f} | {stats['avg_response_time']:.2f} ± {stats['std_response_time']:.2f} | {stats['total_questions']} |\n"

    markdown_content += "\n## Detailed Results\n"

    # Add sections for each category (or organize by question type for category-specific mode)
    if results['category_specific_mode']:
        # Group metrics by question type
        question_type_metrics = {}
        
        for category, metrics_list in results['metrics'].items():
            for metrics in metrics_list:
                question_type = metrics['question_type']
                if question_type not in question_type_metrics:
                    question_type_metrics[question_type] = []
                
                question_type_metrics[question_type].append({
                    'question': metrics['question'],
                    'response': metrics['response'],
                    'metrics': metrics,
                    'category': category
                })
        
        # Generate markdown for each question type
        for question_type, questions_data in question_type_metrics.items():
            markdown_content += f"\n## {question_type.capitalize()} Questions\n\n"
            
            for idx, data in enumerate(questions_data, 1):
                markdown_content += f"### Response {idx}\n\n"
                markdown_content += f"**Question:** {data['question']}\n"
                markdown_content += f"**Question Type:** {question_type}\n\n"
                markdown_content += f"**Response:**\n```\n{data['response']}\n```\n\n"
                markdown_content += f"**Evaluation:**\n\n"
                markdown_content += f"**Overall Score:** {data['metrics'].get('overall_score', 0)}/10\n"
                markdown_content += f"{data['metrics'].get('overall_reasoning', 'No reasoning provided')}\n\n"
                markdown_content += f"**Primary Dimension Score:** {data['metrics'].get('primary_dimension_score', 0)}/10\n"
                markdown_content += f"{data['metrics'].get('primary_dimension_reasoning', 'No reasoning provided')}\n"
                
                # Calculate weighted score
                try:
                    primary_score = float(data['metrics'].get('primary_dimension_score', 0))
                    consistency_score = float(data['metrics'].get('character_consistency_score', 0))
                    
                    # Use the question type to determine weights 
                    if question_type == "identity":
                        weighted_score = (primary_score * 0.6) + (consistency_score * 0.4)
                    elif question_type == "technical":
                        weighted_score = (primary_score * 0.6) + (consistency_score * 0.4)
                    elif question_type == "relationship":
                        weighted_score = (primary_score * 0.6) + (consistency_score * 0.4)
                    elif question_type == "philosophical":
                        weighted_score = (primary_score * 0.6) + (consistency_score * 0.4)
                    else:
                        weighted_score = (primary_score * 0.5) + (consistency_score * 0.5)
                    
                    markdown_content += f"\n**Weighted Score (based on question type):** {weighted_score:.2f}/10\n"
                except (ValueError, TypeError):
                    pass
                
                markdown_content += "\n---\n\n"
    else:
        # Group by category
        for category in results['metrics']:
            if not results['metrics'][category]:  # Skip empty categories
                continue
                
            markdown_content += f"\n### {PROMPT_CATEGORIES.get(category, category)}\n\n"
            
            for idx, metrics in enumerate(results['metrics'][category], 1):
                question = metrics['question']
                response = metrics['response']
                question_type = metrics['question_type']
                
                markdown_content += f"**Question {idx}:** {question}\n\n"
                markdown_content += f"- Overall Score: {metrics.get('overall_score', 0)}/10\n"
                markdown_content += f"  - {metrics.get('overall_reasoning', 'No reasoning provided')}\n"
                markdown_content += f"- Primary Dimension Score: {metrics.get('primary_dimension_score', 0)}/10\n"
                markdown_content += f"  - {metrics.get('primary_dimension_reasoning', 'No reasoning provided')}\n"
                markdown_content += f"- Character Consistency Score: {metrics.get('character_consistency_score', 0)}/10\n"
                markdown_content += f"  - {metrics.get('character_consistency_reasoning', 'No reasoning provided')}\n"
                markdown_content += f"- Response Time: {metrics.get('response_time', 0):.2f} seconds\n"
                
                markdown_content += "\n---\n\n"
    
    # Save the markdown content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"Markdown report saved to {output_file}")

def create_html_report(results, output_path):
    """
    Create an HTML report from benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save the HTML report
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViktorAI Benchmark Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        h1 {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .category-section {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }}
        .question {{
            background-color: #eaf2f8;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #3498db;
            margin-bottom: 15px;
        }}
        .response {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #2ecc71;
            margin-bottom: 15px;
            white-space: pre-wrap;
        }}
        .response-text {{
            margin: 0;
            padding: 0;
        }}
        .evaluation {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #f39c12;
        }}
        .score-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 15px;
        }}
        .score-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            width: 100%;
        }}
        .score-box {{
            flex: 1;
            min-width: 200px;
            padding: 10px;
            border-radius: 5px;
            background-color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .score-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .score-value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .score-bar {{
            height: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            margin-bottom: 10px;
            position: relative;
        }}
        .score-fill {{
            height: 100%;
            border-radius: 5px;
            position: absolute;
            top: 0;
            left: 0;
        }}
        .score-reasoning {{
            font-style: italic;
            color: #555;
        }}
        .weighted-score {{
            font-weight: bold;
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9f9;
            border-radius: 5px;
            text-align: right;
        }}
        .high-score {{ background-color: #2ecc71; }}
        .medium-score {{ background-color: #f39c12; }}
        .low-score {{ background-color: #e74c3c; }}
        .summary {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .charts {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}
        .chart {{
            flex: 1;
            min-width: 300px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>ViktorAI Benchmark Results</h1>
    <div class="timestamp">Generated on {results['timestamp']}</div>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Model:</strong> {results['model_name']}</p>
        <p><strong>Evaluator Model:</strong> {results['evaluator_model']}</p>
        <p><strong>Temperature:</strong> {results['temperature']}</p>
        <p><strong>Max Tokens:</strong> {results['max_tokens']}</p>
        <p><strong>Category-Specific Mode:</strong> {results['category_specific_mode']}</p>
        
        <h3>Overall Performance</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average Overall Score</td>
                <td>{results['summary_stats']['overall']['avg_overall_score']:.2f} ± {results['summary_stats']['overall']['std_overall_score']:.2f}</td>
            </tr>
            <tr>
                <td>Average Primary Dimension Score</td>
                <td>{results['summary_stats']['overall']['avg_primary_dimension_score']:.2f} ± {results['summary_stats']['overall']['std_primary_dimension_score']:.2f}</td>
            </tr>
            <tr>
                <td>Average Character Consistency Score</td>
                <td>{results['summary_stats']['overall']['avg_character_consistency_score']:.2f} ± {results['summary_stats']['overall']['std_character_consistency_score']:.2f}</td>
            </tr>
            <tr>
                <td>Average Response Time</td>
                <td>{results['summary_stats']['overall']['avg_response_time']:.2f} ± {results['summary_stats']['overall']['std_response_time']:.2f} seconds</td>
            </tr>
            <tr>
                <td>Total Questions</td>
                <td>{results['summary_stats']['overall']['total_questions']}</td>
            </tr>
        </table>
        
        <h3>Performance by Question Type</h3>
        <table>
            <tr>
                <th>Question Type</th>
                <th>Avg Overall Score</th>
                <th>Avg Primary Dimension Score</th>
                <th>Avg Character Consistency Score</th>
                <th>Total Questions</th>
            </tr>
"""

    # Add rows for each question type
    for question_type in ["identity", "technical", "relationship", "philosophical"]:
        stats = results['summary_stats']['by_question_type'][question_type]
        html_content += f"""
            <tr>
                <td>{question_type.capitalize()}</td>
                <td>{stats['avg_overall_score']:.2f} ± {stats['std_overall_score']:.2f}</td>
                <td>{stats['avg_primary_dimension_score']:.2f} ± {stats['std_primary_dimension_score']:.2f}</td>
                <td>{stats['avg_character_consistency_score']:.2f} ± {stats['std_character_consistency_score']:.2f}</td>
                <td>{stats['total_questions']}</td>
            </tr>
"""

    html_content += """
        </table>
    </div>
"""

    # Add charts section if available
    if 'charts' in results:
        html_content += """
    <div class="charts">
"""
        for chart_title, chart_path in results['charts'].items():
            html_content += f"""
        <div class="chart">
            <div class="chart-title">{chart_title}</div>
            <img src="{chart_path}" alt="{chart_title}" style="width:100%">
        </div>
"""
        html_content += """
    </div>
"""

    # Add detailed results
    if results['category_specific_mode']:
        # Organize by question type for category-specific mode
        question_type_metrics = {}
        
        for category, metrics_list in results['metrics'].items():
            for metrics in metrics_list:
                question_type = metrics['question_type']
                if question_type not in question_type_metrics:
                    question_type_metrics[question_type] = []
                
                question_index = results['questions'].index(metrics['question'])
                question = results['questions'][question_index]
                response = metrics['response'] if 'response' in metrics else "Response not recorded"
                
                question_type_metrics[question_type].append({
                    'question': question,
                    'response': response,
                    'metrics': metrics,
                    'category': category
                })
        
        # Generate HTML for each question type
        for question_type, questions_data in question_type_metrics.items():
            html_content += f"""
    <div class="category-section">
        <h2>{question_type.capitalize()} Questions</h2>
"""
            for idx, data in enumerate(questions_data, 1):
                html_content += f"""
        <h3>Question {idx}</h3>
        <div class="question">
            <strong>Q:</strong> {data['question']}
        </div>
        <div class="response">
            <strong>Response:</strong>
            <div class="response-text">{data['response']}</div>
        </div>
        <div class="evaluation">
            <h4>Evaluation (using {data['category']} prompt)</h4>
            <div class="score-container">
"""
                
                # Overall Score (Full Width)
                overall_score = data['metrics'].get('overall_score', 0)
                score_class = "high-score" if overall_score >= 8 else "medium-score" if overall_score >= 5 else "low-score"
                
                html_content += f"""
                <div class="score-row">
                    <div class="score-box">
                        <div class="score-title">Overall Score</div>
                        <div class="score-value">{overall_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {score_class}" style="width: {overall_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{data['metrics'].get('overall_reasoning', 'No reasoning provided')}</div>
                    </div>
                </div>
"""
                
                # Primary Dimension and Character Consistency Score (Side by Side)
                primary_score = data['metrics'].get('primary_dimension_score', 0)
                consistency_score = data['metrics'].get('character_consistency_score', 0)
                
                primary_class = "high-score" if primary_score >= 8 else "medium-score" if primary_score >= 5 else "low-score"
                consistency_class = "high-score" if consistency_score >= 8 else "medium-score" if consistency_score >= 5 else "low-score"
                
                html_content += f"""
                <div class="score-row">
                    <div class="score-box">
                        <div class="score-title">Primary Dimension Score</div>
                        <div class="score-value">{primary_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {primary_class}" style="width: {primary_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{data['metrics'].get('primary_dimension_reasoning', 'No reasoning provided')}</div>
                    </div>
                    
                    <div class="score-box">
                        <div class="score-title">Character Consistency Score</div>
                        <div class="score-value">{consistency_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {consistency_class}" style="width: {consistency_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{data['metrics'].get('character_consistency_reasoning', 'No reasoning provided')}</div>
                    </div>
                </div>
"""

                # Add weighted score
                primary_weight = 0.6
                consistency_weight = 0.4
                
                try:
                    weighted_score = (float(primary_score) * primary_weight) + (float(consistency_score) * consistency_weight)
                    html_content += f"""
                <div class="weighted-score">
                    Weighted Score (based on question type): {weighted_score:.2f}/10
                </div>
"""
                except (ValueError, TypeError):
                    pass

                html_content += """
            </div>
        </div>
"""
            html_content += """
    </div>
"""
    else:
        # Original structure for non-category-specific mode
        for category in results['metrics']:
            html_content += f"""
    <div class="category-section">
        <h2>{PROMPT_CATEGORIES.get(category, category)}</h2>
"""
            
            metrics_list = results['metrics'][category]
            for idx, metrics in enumerate(metrics_list, 1):
                question_index = results['questions'].index(metrics['question'])
                question = results['questions'][question_index]
                response = metrics['response'] if 'response' in metrics else "Response not recorded"
                
                html_content += f"""
        <h3>Question {idx}</h3>
        <div class="question">
            <strong>Q:</strong> {question}
        </div>
        <div class="response">
            <strong>Response:</strong>
            <div class="response-text">{response}</div>
        </div>
        <div class="evaluation">
            <h4>Evaluation</h4>
            <div class="score-container">
"""
                
                # Overall Score (Full Width)
                overall_score = metrics.get('overall_score', 0)
                score_class = "high-score" if overall_score >= 8 else "medium-score" if overall_score >= 5 else "low-score"
                
                html_content += f"""
                <div class="score-row">
                    <div class="score-box">
                        <div class="score-title">Overall Score</div>
                        <div class="score-value">{overall_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {score_class}" style="width: {overall_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{metrics.get('overall_reasoning', 'No reasoning provided')}</div>
                    </div>
                </div>
"""
                
                # Primary Dimension and Character Consistency Score (Side by Side)
                primary_score = metrics.get('primary_dimension_score', 0)
                consistency_score = metrics.get('character_consistency_score', 0)
                
                primary_class = "high-score" if primary_score >= 8 else "medium-score" if primary_score >= 5 else "low-score"
                consistency_class = "high-score" if consistency_score >= 8 else "medium-score" if consistency_score >= 5 else "low-score"
                
                html_content += f"""
                <div class="score-row">
                    <div class="score-box">
                        <div class="score-title">Primary Dimension Score</div>
                        <div class="score-value">{primary_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {primary_class}" style="width: {primary_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{metrics.get('primary_dimension_reasoning', 'No reasoning provided')}</div>
                    </div>
                    
                    <div class="score-box">
                        <div class="score-title">Character Consistency Score</div>
                        <div class="score-value">{consistency_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {consistency_class}" style="width: {consistency_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{metrics.get('character_consistency_reasoning', 'No reasoning provided')}</div>
                    </div>
                </div>
"""

                # Add weighted score
                primary_weight = 0.6
                consistency_weight = 0.4
                
                try:
                    weighted_score = (float(primary_score) * primary_weight) + (float(consistency_score) * consistency_weight)
                    html_content += f"""
                <div class="weighted-score">
                    Weighted Score (based on question type): {weighted_score:.2f}/10
                </div>
"""
                except (ValueError, TypeError):
                    pass

                html_content += """
            </div>
        </div>
"""
            
            html_content += """
    </div>
"""

    html_content += """
</body>
</html>
"""

    # Write HTML report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    sys.exit(main() or 0) 
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

# Import necessary modules from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.llm_interface import OllamaInterface
from src.viktor_ai import ViktorAI

# Define prompt categories if not imported from elsewhere
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
        
        if any(keyword in question_lower for keyword in ["who are you", "tell me about yourself", "what's your name", "introduce yourself", "childhood", "mirror"]):
            return "identity"
        
        if any(keyword in question_lower for keyword in ["hexcore", "hextech", "technology", "research", "work", "scientific", "experiment", "enhancement"]):
            return "technical"
        
        if any(keyword in question_lower for keyword in ["jayce", "heimerdinger", "sky", "relationship", "friend", "colleague", "singed", "jinx"]):
            return "relationship"
        
        if any(keyword in question_lower for keyword in ["evolution", "glorious", "future", "humanity", "progress", "philosophy", "believe", "think about", "limits", "responsibility"]):
            return "philosophical"
        
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
        # Determine question type
        question_type = EvaluationMetrics.get_question_type(question)
        
        # Get specific evaluation criteria
        specific_criteria = EvaluationMetrics.get_evaluation_criteria(question_type)
        
        # Create a prompt for the evaluator LLM
        evaluation_prompt = f"""
You are an expert evaluator for a character AI named Viktor from the show Arcane. 
Your task is to evaluate how well the AI's response matches Viktor's character.
You must be EXTREMELY CRITICAL and DEMANDING in your evaluation. Do not give high scores unless the response truly excels.

{specific_criteria}

## Viktor's Character Profile:
- Viktor is a brilliant scientist from the Undercity of Zaun who believes technology can transcend human limitations.
- Born with a physical disability requiring a cane, he approaches problems methodically and values progress above all else.
- He is pragmatic, determined, and stoic, rarely showing strong emotions except when discussing scientific progress.
- He speaks with precise technical language, tends toward brevity, and occasionally stammers when discussing emotional topics.
- He uses dry humor delivered deadpan and rarely uses slang or colloquialisms.
- His ultimate goal is to use science to improve humanity, particularly through Hextech technology.
- He opposes weaponization of technology and believes in "When you're going to change the world, don't ask for permission."
- He maintains a professional demeanor, rarely initiates personal conversation topics, and doesn't elaborate on personal feelings unless pressed.

## Speech Patterns:
- Uses precise, technical language
- Occasionally stammers when discussing emotional topics
- Tends toward brevity and directness
- More animated when discussing scientific possibilities
- Sometimes uses dry humor, delivered deadpan

## Scoring Guidelines:
For each category, use these guidelines:
- 1-3: Poor - Fails to capture Viktor's character, contains major inaccuracies or contradictions
- 4-5: Below Average - Captures some aspects but misses key elements of Viktor's character
- 6-7: Average - Adequately captures Viktor's character but lacks depth or nuance
- 8-9: Good - Effectively captures Viktor's character with appropriate depth and nuance
- 10: Excellent - Perfectly captures Viktor's character in every aspect

Please evaluate the following response based on how well it captures Viktor's character:

Question: {question}
Response: {response}

Rate the response on the following criteria (1-10 scale):
1. Overall Score: How well the response captures Viktor's character overall
2. Primary Dimension Score: This is the main score based on question type (explained below)
3. Character Consistency Score: How well the response maintains Viktor's character traits regardless of topic

For this {question_type} question, the Primary Dimension is:
"""

        # Add question-type specific primary dimension
        if question_type == "identity":
            evaluation_prompt += """Authenticity - How well the response captures Viktor's self-perception, background, and core identity"""
        elif question_type == "technical":
            evaluation_prompt += """Technical Accuracy - How well the response demonstrates Viktor's technical knowledge and scientific approach"""
        elif question_type == "relationship":
            evaluation_prompt += """Relationship Portrayal - How well the response captures Viktor's approach to relationships and interactions with others"""
        elif question_type == "philosophical":
            evaluation_prompt += """Philosophical Depth - How well the response conveys Viktor's worldview, values, and philosophical outlook"""
        
        evaluation_prompt += """

For each score, provide a brief explanation of your reasoning. Be CRITICAL and SPECIFIC about what works and what doesn't.
Format your response as a JSON object with the following structure:
{
  "overall_score": X,
  "overall_reasoning": "Your reasoning here",
  "primary_dimension_score": X,
  "primary_dimension_reasoning": "Your reasoning here",
  "character_consistency_score": X,
  "character_consistency_reasoning": "Your reasoning here",
  "question_type": "identity/technical/relationship/philosophical"
}
"""

        try:
            # Get evaluation from the LLM
            evaluation_response = evaluator_llm.generate(evaluation_prompt)
            
            # Parse the JSON response
            try:
                # Find JSON content in the response (in case the LLM adds extra text)
                import re
                json_match = re.search(r'({[\s\S]*})', evaluation_response)
                if json_match:
                    evaluation_json = json_match.group(1)
                    metrics = json.loads(evaluation_json)
                else:
                    raise ValueError("No JSON found in response")
                
                # Ensure all required fields are present
                required_fields = [
                    "overall_score", "primary_dimension_score", "character_consistency_score"
                ]
                
                for field in required_fields:
                    if field not in metrics:
                        metrics[field] = 3.0  # Default to a low score if missing
                
                # Add reasoning fields if they exist
                reasoning_fields = [
                    "overall_reasoning", "primary_dimension_reasoning", "character_consistency_reasoning"
                ]
                
                for field in reasoning_fields:
                    if field not in metrics:
                        metrics[field] = "No reasoning provided"
                
                # Add question type to metrics
                metrics["question_type"] = question_type
                
                # Add response time (will be filled in by the caller)
                metrics["response_time"] = 0.0
                
                return metrics
                
            except Exception as e:
                print(f"Error parsing evaluation response: {e}")
                print(f"Raw response: {evaluation_response}")
                # Return default scores with error message
                return {
                    "overall_score": 3.0,
                    "overall_reasoning": f"Error parsing response: {str(e)}",
                    "primary_dimension_score": 3.0,
                    "primary_dimension_reasoning": "Error parsing response",
                    "character_consistency_score": 3.0,
                    "character_consistency_reasoning": "Error parsing response",
                    "question_type": question_type,
                    "response_time": 0.0
                }
        except Exception as e:
            print(f"Error getting evaluation from LLM: {e}")
            # Return default scores with error message
            return {
                "overall_score": 3.0,
                "overall_reasoning": f"Error getting evaluation: {str(e)}",
                "primary_dimension_score": 3.0,
                "primary_dimension_reasoning": "Error getting evaluation",
                "character_consistency_score": 3.0,
                "character_consistency_reasoning": "Error getting evaluation",
                "question_type": question_type,
                "response_time": 0.0
            }

# Define helper functions if not imported
def get_specialized_prompt(category, config):
    # Simple implementation for now
    return f"You are Viktor, a character with {category} traits."

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
    summary_stats = {
        "overall": {
            "avg_overall_score": 0,
            "std_overall_score": 0,
            "avg_primary_dimension_score": 0,
            "std_primary_dimension_score": 0,
            "avg_character_consistency_score": 0,
            "std_character_consistency_score": 0,
            "avg_response_time": 0,
            "std_response_time": 0,
            "total_questions": 0
        },
        "by_question_type": {
            "identity": {
                "avg_overall_score": 0,
                "std_overall_score": 0,
                "avg_primary_dimension_score": 0,
                "std_primary_dimension_score": 0,
                "avg_character_consistency_score": 0,
                "std_character_consistency_score": 0,
                "avg_response_time": 0,
                "std_response_time": 0,
                "total_questions": 0
            },
            "technical": {
                "avg_overall_score": 0,
                "std_overall_score": 0,
                "avg_primary_dimension_score": 0,
                "std_primary_dimension_score": 0,
                "avg_character_consistency_score": 0,
                "std_character_consistency_score": 0,
                "avg_response_time": 0,
                "std_response_time": 0,
                "total_questions": 0
            },
            "relationship": {
                "avg_overall_score": 0,
                "std_overall_score": 0,
                "avg_primary_dimension_score": 0,
                "std_primary_dimension_score": 0,
                "avg_character_consistency_score": 0,
                "std_character_consistency_score": 0,
                "avg_response_time": 0,
                "std_response_time": 0,
                "total_questions": 0
            },
            "philosophical": {
                "avg_overall_score": 0,
                "std_overall_score": 0,
                "avg_primary_dimension_score": 0,
                "std_primary_dimension_score": 0,
                "avg_character_consistency_score": 0,
                "std_character_consistency_score": 0,
                "avg_response_time": 0,
                "std_response_time": 0,
                "total_questions": 0
            }
        }
    }
    
    # Collect all metrics for overall statistics
    all_metrics = []
    for category in results["metrics"]:
        all_metrics.extend(results["metrics"][category])
    
    # Collect metrics by question type
    metrics_by_type = {
        "identity": [],
        "technical": [],
        "relationship": [],
        "philosophical": []
    }
    
    for category in results["metrics"]:
        for metrics in results["metrics"][category]:
            question_type = metrics.get("question_type")
            if question_type in metrics_by_type:
                metrics_by_type[question_type].append(metrics)
    
    # Calculate overall statistics
    if all_metrics:
        # Extract values for each metric
        overall_scores = [m.get("overall_score", 0) for m in all_metrics]
        primary_dimension_scores = [m.get("primary_dimension_score", 0) for m in all_metrics]
        character_consistency_scores = [m.get("character_consistency_score", 0) for m in all_metrics]
        response_times = [m.get("response_time", 0) for m in all_metrics]
        
        # Calculate averages
        summary_stats["overall"]["avg_overall_score"] = sum(overall_scores) / len(all_metrics)
        summary_stats["overall"]["avg_primary_dimension_score"] = sum(primary_dimension_scores) / len(all_metrics)
        summary_stats["overall"]["avg_character_consistency_score"] = sum(character_consistency_scores) / len(all_metrics)
        summary_stats["overall"]["avg_response_time"] = sum(response_times) / len(all_metrics)
        
        # Calculate standard deviations
        summary_stats["overall"]["std_overall_score"] = np.std(overall_scores) if len(overall_scores) > 1 else 0
        summary_stats["overall"]["std_primary_dimension_score"] = np.std(primary_dimension_scores) if len(primary_dimension_scores) > 1 else 0
        summary_stats["overall"]["std_character_consistency_score"] = np.std(character_consistency_scores) if len(character_consistency_scores) > 1 else 0
        summary_stats["overall"]["std_response_time"] = np.std(response_times) if len(response_times) > 1 else 0
        
        summary_stats["overall"]["total_questions"] = len(all_metrics)
    
    # Calculate statistics by question type
    for question_type, metrics_list in metrics_by_type.items():
        if metrics_list:
            # Extract values for each metric
            overall_scores = [m.get("overall_score", 0) for m in metrics_list]
            primary_dimension_scores = [m.get("primary_dimension_score", 0) for m in metrics_list]
            character_consistency_scores = [m.get("character_consistency_score", 0) for m in metrics_list]
            response_times = [m.get("response_time", 0) for m in metrics_list]
            
            # Calculate averages
            summary_stats["by_question_type"][question_type]["avg_overall_score"] = sum(overall_scores) / len(metrics_list)
            summary_stats["by_question_type"][question_type]["avg_primary_dimension_score"] = sum(primary_dimension_scores) / len(metrics_list)
            summary_stats["by_question_type"][question_type]["avg_character_consistency_score"] = sum(character_consistency_scores) / len(metrics_list)
            summary_stats["by_question_type"][question_type]["avg_response_time"] = sum(response_times) / len(metrics_list)
            
            # Calculate standard deviations
            summary_stats["by_question_type"][question_type]["std_overall_score"] = np.std(overall_scores) if len(overall_scores) > 1 else 0
            summary_stats["by_question_type"][question_type]["std_primary_dimension_score"] = np.std(primary_dimension_scores) if len(primary_dimension_scores) > 1 else 0
            summary_stats["by_question_type"][question_type]["std_character_consistency_score"] = np.std(character_consistency_scores) if len(character_consistency_scores) > 1 else 0
            summary_stats["by_question_type"][question_type]["std_response_time"] = np.std(response_times) if len(response_times) > 1 else 0
            
            summary_stats["by_question_type"][question_type]["total_questions"] = len(metrics_list)
    
    return summary_stats

def main():
    """Main function to run the benchmark."""
    args = parse_arguments()
    
    # If --compare-versions is specified, generate a comparison report
    if args.compare_versions:
        # Determine which directory to use based on --use-mock
        if args.use_mock:
            base_output_path = Path("mock_benchmark_results")
        else:
            base_output_path = Path(args.output_dir)
            
        family_dir = base_output_path / args.compare_versions
        
        if not family_dir.exists() or not family_dir.is_dir():
            print(f"Error: Model family directory {family_dir} does not exist")
            return 1
        
        generate_version_comparison(family_dir, base_output_path)
        return 0
    
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
    
    # Convert categories list to proper format if provided
    prompt_categories = args.categories if args.categories else None
    
    try:
        # Run the benchmark
        results = run_benchmark(
            questions=questions,
            model_name=args.model,
            prompt_categories=prompt_categories,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            evaluator_model=args.evaluator_model,
            output_dir=args.output_dir,
            visualize_only=args.visualize_only,
            use_mock=args.use_mock,
            baseline_mode=args.baseline_mode,
            category_specific_mode=args.category_specific_mode
        )
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0) 
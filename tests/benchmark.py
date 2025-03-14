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
2. Authenticity Score: How authentic the response feels to Viktor's character
3. Technical Score: How well the response reflects Viktor's technical knowledge and approach
4. Emotional Score: How well the response captures Viktor's emotional state and expressions
5. Quality Score: The general quality and coherence of the response

For each score, provide a brief explanation of your reasoning. Be CRITICAL and SPECIFIC about what works and what doesn't.
Format your response as a JSON object with the following structure:
{{
  "overall_score": X,
  "overall_reasoning": "Your reasoning here",
  "authenticity_score": X,
  "authenticity_reasoning": "Your reasoning here",
  "technical_score": X,
  "technical_reasoning": "Your reasoning here",
  "emotional_score": X,
  "emotional_reasoning": "Your reasoning here",
  "quality_score": X,
  "quality_reasoning": "Your reasoning here"
}}
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
                    "overall_score", "authenticity_score", "technical_score", 
                    "emotional_score", "quality_score"
                ]
                
                for field in required_fields:
                    if field not in metrics:
                        metrics[field] = 3.0  # Default to a low score if missing
                
                # Add reasoning fields if they exist
                reasoning_fields = [
                    "overall_reasoning", "authenticity_reasoning", "technical_reasoning", 
                    "emotional_reasoning", "quality_reasoning"
                ]
                
                for field in reasoning_fields:
                    if field not in metrics:
                        metrics[field] = "No reasoning provided"
                
                # Add question type to metrics
                metrics["question_type"] = question_type
                
                # Calculate weighted score based on question type
                metrics["weighted_score"] = EvaluationMetrics.calculate_weighted_score(metrics)
                
                return metrics
                
            except Exception as e:
                print(f"Error parsing evaluation response: {e}")
                print(f"Raw response: {evaluation_response}")
                # Return default scores with error message
                return {
                    "overall_score": 3.0,
                    "overall_reasoning": f"Error parsing response: {str(e)}",
                    "authenticity_score": 3.0,
                    "authenticity_reasoning": "Error parsing response",
                    "technical_score": 3.0,
                    "technical_reasoning": "Error parsing response",
                    "emotional_score": 3.0,
                    "emotional_reasoning": "Error parsing response",
                    "quality_score": 3.0,
                    "quality_reasoning": "Error parsing response",
                    "question_type": question_type,
                    "weighted_score": 3.0
                }
        except Exception as e:
            print(f"Error getting evaluation from LLM: {e}")
            # Return default scores with error message
            return {
                "overall_score": 3.0,
                "overall_reasoning": f"Error getting evaluation: {str(e)}",
                "authenticity_score": 3.0,
                "authenticity_reasoning": "Error getting evaluation",
                "technical_score": 3.0,
                "technical_reasoning": "Error getting evaluation",
                "emotional_score": 3.0,
                "emotional_reasoning": "Error getting evaluation",
                "quality_score": 3.0,
                "quality_reasoning": "Error getting evaluation",
                "question_type": question_type,
                "weighted_score": 3.0
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

def calculate_summary_statistics(metrics_data):
    # Simple implementation for now
    return {"average_score": 7.0}

def create_html_report(results, output_path):
    """
    Create a detailed HTML report from benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save the HTML report
    """
    model_name = results["metadata"]["model"]
    timestamp = results["metadata"]["timestamp"]
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results: {model_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .response {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            margin-bottom: 20px;
        }}
        .evaluation {{
            background-color: #e9f7ef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .reasoning {{
            font-style: italic;
            color: #555;
            margin-left: 20px;
        }}
        .summary {{
            font-weight: bold;
        }}
        .score {{
            font-weight: bold;
        }}
        .divider {{
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <h1>Benchmark Results: {model_name}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Metadata</h2>
    <div class="metadata">
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Model Family:</strong> {results['metadata']['model_family']}</p>
        <p><strong>Temperature:</strong> {results['metadata']['temperature']}</p>
        <p><strong>Max Tokens:</strong> {results['metadata']['max_tokens']}</p>
        <p><strong>Evaluator Model:</strong> {results['metadata']['evaluator_model']}</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Mock Implementation:</strong> {'Yes' if results['metadata']['is_mock'] else 'No'}</p>
    </div>
    
    <h2>Summary Statistics</h2>
"""
    
    # Calculate average scores across all categories
    avg_scores = {
        "overall_score": 0,
        "weighted_score": 0,
        "authenticity_score": 0,
        "technical_score": 0,
        "emotional_score": 0,
        "quality_score": 0,
        "response_time": 0
    }
    
    total_metrics = 0
    
    for category in results["metrics"]:
        for metrics in results["metrics"][category]:
            for key in avg_scores:
                if key in metrics:
                    avg_scores[key] += metrics[key]
                elif key == "weighted_score" and "overall_score" in metrics:
                    # If weighted_score is not present, use overall_score as fallback
                    avg_scores[key] += metrics["overall_score"]
            total_metrics += 1
    
    # Calculate averages
    if total_metrics > 0:
        for key in avg_scores:
            avg_scores[key] /= total_metrics
    
    # Add summary table
    html += """
    <table>
        <tr>
            <th>Metric</th>
            <th>Average Score</th>
        </tr>
"""
    
    html += f"""
        <tr>
            <td>Overall Score</td>
            <td>{avg_scores['overall_score']:.2f}</td>
        </tr>
        <tr>
            <td>Weighted Score</td>
            <td>{avg_scores['weighted_score']:.2f}</td>
        </tr>
        <tr>
            <td>Authenticity</td>
            <td>{avg_scores['authenticity_score']:.2f}</td>
        </tr>
        <tr>
            <td>Technical</td>
            <td>{avg_scores['technical_score']:.2f}</td>
        </tr>
        <tr>
            <td>Emotional</td>
            <td>{avg_scores['emotional_score']:.2f}</td>
        </tr>
        <tr>
            <td>Quality</td>
            <td>{avg_scores['quality_score']:.2f}</td>
        </tr>
        <tr>
            <td>Response Time</td>
            <td>{avg_scores['response_time']:.2f}s</td>
        </tr>
    </table>
    
    <h2>Scores by Category</h2>
"""
    
    # Add category scores
    for category in results["metrics"]:
        html += f"""
    <h3>{category}</h3>
"""
        
        # Calculate average scores for this category
        cat_avg_scores = {
            "overall_score": 0,
            "weighted_score": 0,
            "authenticity_score": 0,
            "technical_score": 0,
            "emotional_score": 0,
            "quality_score": 0,
            "response_time": 0
        }
        
        cat_total = len(results["metrics"][category])
        
        for metrics in results["metrics"][category]:
            for key in cat_avg_scores:
                if key in metrics:
                    cat_avg_scores[key] += metrics[key]
                elif key == "weighted_score" and "overall_score" in metrics:
                    # If weighted_score is not present, use overall_score as fallback
                    cat_avg_scores[key] += metrics["overall_score"]
        
        # Calculate averages
        if cat_total > 0:
            for key in cat_avg_scores:
                cat_avg_scores[key] /= cat_total
        
        # Add category summary table
        html += """
    <table>
        <tr>
            <th>Metric</th>
            <th>Average Score</th>
        </tr>
"""
        
        html += f"""
        <tr>
            <td>Overall Score</td>
            <td>{cat_avg_scores['overall_score']:.2f}</td>
        </tr>
        <tr>
            <td>Weighted Score</td>
            <td>{cat_avg_scores.get('weighted_score', cat_avg_scores['overall_score']):.2f}</td>
        </tr>
        <tr>
            <td>Authenticity</td>
            <td>{cat_avg_scores['authenticity_score']:.2f}</td>
        </tr>
        <tr>
            <td>Technical</td>
            <td>{cat_avg_scores['technical_score']:.2f}</td>
        </tr>
        <tr>
            <td>Emotional</td>
            <td>{cat_avg_scores['emotional_score']:.2f}</td>
        </tr>
        <tr>
            <td>Quality</td>
            <td>{cat_avg_scores['quality_score']:.2f}</td>
        </tr>
        <tr>
            <td>Response Time</td>
            <td>{cat_avg_scores['response_time']:.2f}s</td>
        </tr>
    </table>
    
    <h4>Question Scores</h4>
    <table>
        <tr>
            <th>Question</th>
            <th>Type</th>
            <th>Overall</th>
            <th>Weighted</th>
            <th>Authenticity</th>
            <th>Technical</th>
            <th>Emotional</th>
            <th>Quality</th>
            <th>Time</th>
        </tr>
"""
        
        # Add individual question scores
        for i, (metrics, response_data) in enumerate(zip(results["metrics"][category], results["responses"][category])):
            question = response_data["question"]
            # Truncate long questions
            if len(question) > 50:
                question = question[:47] + "..."
            
            # Get question type
            question_type = metrics.get('question_type', 'unknown')
            
            # Get weighted score
            weighted_score = metrics.get('weighted_score', metrics.get('overall_score', 0))
            
            html += f"""
        <tr>
            <td>{question}</td>
            <td>{question_type}</td>
            <td>{metrics.get('overall_score', 0):.1f}</td>
            <td>{weighted_score:.1f}</td>
            <td>{metrics.get('authenticity_score', 0):.1f}</td>
            <td>{metrics.get('technical_score', 0):.1f}</td>
            <td>{metrics.get('emotional_score', 0):.1f}</td>
            <td>{metrics.get('quality_score', 0):.1f}</td>
            <td>{metrics.get('response_time', 0):.2f}s</td>
        </tr>
"""
        
        html += """
    </table>
"""
    
    # Add detailed responses and evaluations
    html += """
    <h2>Detailed Responses and Evaluations</h2>
"""
    
    for category in results["responses"]:
        html += f"""
    <h3>{category}</h3>
"""
        
        # Include all responses with evaluations
        for i, (response_data, metrics) in enumerate(zip(results["responses"][category], results["metrics"][category])):
            question = response_data["question"]
            response = response_data["response"].replace("<", "&lt;").replace(">", "&gt;")
            
            html += f"""
    <div class="question-container">
        <h4>Question {i+1}: {question}</h4>
        <p><strong>Question Type:</strong> {metrics.get('question_type', 'unknown')}</p>
        <p><strong>Response Time:</strong> {response_data['response_time']:.2f}s</p>
        
        <h5>Response:</h5>
        <div class="response">{response}</div>
        
        <h5>Evaluation:</h5>
        <div class="evaluation">
            <p class="score">Overall Score: {metrics.get('overall_score', 0):.1f}/10</p>
"""
            
            if 'overall_reasoning' in metrics:
                html += f"""
            <p class="reasoning">{metrics['overall_reasoning']}</p>
"""
            
            html += f"""
            <p class="score">Weighted Score: {metrics.get('weighted_score', metrics.get('overall_score', 0)):.1f}/10</p>
"""
            
            html += f"""
            <p class="score">Authenticity Score: {metrics.get('authenticity_score', 0):.1f}/10</p>
"""
            
            if 'authenticity_reasoning' in metrics:
                html += f"""
            <p class="reasoning">{metrics['authenticity_reasoning']}</p>
"""
            
            html += f"""
            <p class="score">Technical Score: {metrics.get('technical_score', 0):.1f}/10</p>
"""
            
            if 'technical_reasoning' in metrics:
                html += f"""
            <p class="reasoning">{metrics['technical_reasoning']}</p>
"""
            
            html += f"""
            <p class="score">Emotional Score: {metrics.get('emotional_score', 0):.1f}/10</p>
"""
            
            if 'emotional_reasoning' in metrics:
                html += f"""
            <p class="reasoning">{metrics['emotional_reasoning']}</p>
"""
            
            html += f"""
            <p class="score">Quality Score: {metrics.get('quality_score', 0):.1f}/10</p>
"""
            
            if 'quality_reasoning' in metrics:
                html += f"""
            <p class="reasoning">{metrics['quality_reasoning']}</p>
"""
            
            html += """
        </div>
    </div>
    <div class="divider"></div>
"""
    
    html += """
    <p><em>End of Report</em></p>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

def create_markdown_report(results, output_path):
    """
    Create a detailed markdown report from benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save the markdown report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        model_name = results["metadata"]["model"]
        timestamp = results["metadata"]["timestamp"]
        f.write(f"# Benchmark Results: {model_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write metadata
        f.write("## Metadata\n\n")
        f.write(f"- **Model:** {model_name}\n")
        f.write(f"- **Model Family:** {results['metadata']['model_family']}\n")
        f.write(f"- **Temperature:** {results['metadata']['temperature']}\n")
        f.write(f"- **Max Tokens:** {results['metadata']['max_tokens']}\n")
        f.write(f"- **Evaluator Model:** {results['metadata']['evaluator_model']}\n")
        f.write(f"- **Timestamp:** {timestamp}\n")
        f.write(f"- **Mock Implementation:** {'Yes' if results['metadata']['is_mock'] else 'No'}\n\n")
        
        # Write summary statistics
        f.write("## Summary Statistics\n\n")
        
        # Calculate average scores across all categories
        avg_scores = {
            "overall_score": 0,
            "weighted_score": 0,
            "authenticity_score": 0,
            "technical_score": 0,
            "emotional_score": 0,
            "quality_score": 0,
            "response_time": 0
        }
        
        total_metrics = 0
        
        for category in results["metrics"]:
            for metrics in results["metrics"][category]:
                for key in avg_scores:
                    if key in metrics:
                        avg_scores[key] += metrics[key]
                total_metrics += 1
        
        # Calculate averages
        if total_metrics > 0:
            for key in avg_scores:
                avg_scores[key] /= total_metrics
        
        # Write average scores
        f.write("| Metric | Average Score |\n")
        f.write("|--------|---------------|\n")
        f.write(f"| Overall Score | {avg_scores['overall_score']:.2f} |\n")
        f.write(f"| Weighted Score | {avg_scores['weighted_score']:.2f} |\n")
        f.write(f"| Authenticity | {avg_scores['authenticity_score']:.2f} |\n")
        f.write(f"| Technical | {avg_scores['technical_score']:.2f} |\n")
        f.write(f"| Emotional | {avg_scores['emotional_score']:.2f} |\n")
        f.write(f"| Quality | {avg_scores['quality_score']:.2f} |\n")
        f.write(f"| Response Time | {avg_scores['response_time']:.2f}s |\n\n")
        
        # Write scores by category
        f.write("## Scores by Category\n\n")
        
        for category in results["metrics"]:
            f.write(f"### {category}\n\n")
            
            # Calculate average scores for this category
            cat_avg_scores = {
                "overall_score": 0,
                "weighted_score": 0,
                "authenticity_score": 0,
                "technical_score": 0,
                "emotional_score": 0,
                "quality_score": 0,
                "response_time": 0
            }
            
            cat_total = len(results["metrics"][category])
            
            for metrics in results["metrics"][category]:
                for key in cat_avg_scores:
                    if key in metrics:
                        cat_avg_scores[key] += metrics[key]
                    elif key == "weighted_score" and "overall_score" in metrics:
                        # If weighted_score is not present, use overall_score as fallback
                        cat_avg_scores[key] += metrics["overall_score"]
            
            # Calculate averages
            if cat_total > 0:
                for key in cat_avg_scores:
                    cat_avg_scores[key] /= cat_total
            
            # Write average scores for this category
            f.write("| Metric | Average Score |\n")
            f.write("|--------|---------------|\n")
            f.write(f"| Overall Score | {cat_avg_scores['overall_score']:.2f} |\n")
            f.write(f"| Weighted Score | {cat_avg_scores.get('weighted_score', cat_avg_scores['overall_score']):.2f} |\n")
            f.write(f"| Authenticity | {cat_avg_scores['authenticity_score']:.2f} |\n")
            f.write(f"| Technical | {cat_avg_scores['technical_score']:.2f} |\n")
            f.write(f"| Emotional | {cat_avg_scores['emotional_score']:.2f} |\n")
            f.write(f"| Quality | {cat_avg_scores['quality_score']:.2f} |\n")
            f.write(f"| Response Time | {cat_avg_scores['response_time']:.2f}s |\n\n")
            
            # Write individual question scores
            f.write("#### Question Scores\n\n")
            f.write("| Question | Type | Overall | Weighted | Authenticity | Technical | Emotional | Quality | Time |\n")
            f.write("|----------|------|---------|----------|--------------|-----------|-----------|---------|------|\n")
            
            for i, (metrics, response_data) in enumerate(zip(results["metrics"][category], results["responses"][category])):
                question = response_data["question"]
                # Truncate long questions
                if len(question) > 50:
                    question = question[:47] + "..."
                
                # Get question type
                question_type = metrics.get('question_type', 'unknown')
                
                # Get weighted score
                weighted_score = metrics.get('weighted_score', metrics.get('overall_score', 0))
                
                f.write(f"| {question} | {question_type} | {metrics.get('overall_score', 0):.1f} | {weighted_score:.1f} | ")
                f.write(f"{metrics.get('authenticity_score', 0):.1f} | ")
                f.write(f"{metrics.get('technical_score', 0):.1f} | ")
                f.write(f"{metrics.get('emotional_score', 0):.1f} | ")
                f.write(f"{metrics.get('quality_score', 0):.1f} | ")
                f.write(f"{metrics.get('response_time', 0):.2f}s |\n")
            
            f.write("\n")
        
        # Write detailed responses and evaluations
        f.write("## Detailed Responses and Evaluations\n\n")
        
        for category in results["responses"]:
            f.write(f"### {category}\n\n")
            
            # Include all responses with evaluations
            for i, (response_data, metrics) in enumerate(zip(results["responses"][category], results["metrics"][category])):
                question = response_data["question"]
                response = response_data["response"]
                
                # Get question type and weighted score
                question_type = metrics.get('question_type', 'unknown')
                weighted_score = metrics.get('weighted_score', metrics.get('overall_score', 0))
                
                f.write(f"#### Question {i+1}: {question}\n\n")
                f.write(f"**Question Type:** {question_type}\n\n")
                f.write(f"**Response Time:** {response_data['response_time']:.2f}s\n\n")
                f.write("**Response:**\n\n")
                f.write(f"{response}\n\n")
                
                # Add evaluator's scores and reasoning
                f.write("**Evaluation:**\n\n")
                f.write(f"- **Overall Score:** {metrics.get('overall_score', 0):.1f}/10\n")
                if 'overall_reasoning' in metrics:
                    f.write(f"  - {metrics['overall_reasoning']}\n")
                
                f.write(f"- **Weighted Score:** {weighted_score:.1f}/10 (based on question type)\n")
                
                f.write(f"- **Authenticity Score:** {metrics.get('authenticity_score', 0):.1f}/10\n")
                if 'authenticity_reasoning' in metrics:
                    f.write(f"  - {metrics['authenticity_reasoning']}\n")
                
                f.write(f"- **Technical Score:** {metrics.get('technical_score', 0):.1f}/10\n")
                if 'technical_reasoning' in metrics:
                    f.write(f"  - {metrics['technical_reasoning']}\n")
                
                f.write(f"- **Emotional Score:** {metrics.get('emotional_score', 0):.1f}/10\n")
                if 'emotional_reasoning' in metrics:
                    f.write(f"  - {metrics['emotional_reasoning']}\n")
                
                f.write(f"- **Quality Score:** {metrics.get('quality_score', 0):.1f}/10\n")
                if 'quality_reasoning' in metrics:
                    f.write(f"  - {metrics['quality_reasoning']}\n")
                
                f.write("\n---\n\n")
        
        f.write("\n*End of Report*\n")

# Define a mock OllamaInterface for testing without a running Ollama server
class MockOllamaInterface:
    """Mock implementation of OllamaInterface for testing without a running Ollama server."""
    
    def __init__(self, config):
        """Initialize the MockOllamaInterface."""
        self.config = config
        self.history = []
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a mock response."""
        return f"This is a mock response to: {prompt}"
    
    def generate_with_chat_history(self, messages, system_prompt=None):
        """Generate a mock response with chat history."""
        return "This is a mock response with chat history."
    
    def get_history(self):
        """Get the chat history."""
        return self.history
    
    def clear_history(self):
        """Clear the chat history."""
        self.history = []

def run_benchmark(
    questions: List[str],
    model_name: str = "llama3",
    prompt_categories: List[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    evaluator_model: str = "llama3",
    output_dir: str = "benchmark_results",
    visualize_only: bool = False,
    use_mock: bool = False,  # Add a parameter to use mock implementation
    baseline_mode: bool = False  # Add parameter for baseline mode
) -> Dict[str, Any]:
    """
    Run a comprehensive benchmark of ViktorAI with different prompts.
    
    This is the core benchmarking function that:
    1. Tests each prompt category with all test questions
    2. Collects responses and metrics
    3. Saves results for analysis
    
    Args:
        questions: List of test questions to ask
        model_name: Name of the Ollama model to test
        prompt_categories: List of prompt categories to test (default: all)
        temperature: Temperature setting for response generation
        max_tokens: Maximum tokens for response generation
        evaluator_model: Model to use for evaluation
        output_dir: Directory to save results
        visualize_only: If True, only generate visualizations from existing results
        use_mock: If True, use mock implementations instead of real LLM
        baseline_mode: If True, use a minimal prompt regardless of category
        
    Returns:
        Dictionary containing all benchmark results
    """
    try:
        # Determine the base output directory based on whether we're using mock or real LLM
        if use_mock:
            base_output_path = Path("mock_benchmark_results")
        else:
            base_output_path = Path(output_dir)
            
        base_output_path.mkdir(exist_ok=True, parents=True)
        print(f"Created base output directory: {base_output_path}")
        
        # Extract model family and variant
        model_family = model_name.split(':')[0] if ':' in model_name else model_name
        
        # Create model family directory
        family_output_path = base_output_path / model_family
        family_output_path.mkdir(exist_ok=True, parents=True)
        print(f"Created model family directory: {family_output_path}")
        
        # Create model-specific directory
        model_output_path = family_output_path / model_name
        model_output_path.mkdir(exist_ok=True, parents=True)
        print(f"Created model-specific directory: {model_output_path}")
        
        # If visualize_only is True, load the most recent results file for the model
        if visualize_only:
            # Find all run directories for the model
            run_dirs = list(model_output_path.glob("run_*"))
            if not run_dirs:
                print(f"No results found for model {model_name}")
                return {}
            
            # Sort by modification time (newest first)
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Look for results files in all run directories
            results_file = None
            latest_run_dir = None
            
            for run_dir in run_dirs:
                # Check if this run directory has a results file
                result_files = list((run_dir / "raw_data").glob("benchmark_results_*.json"))
                if result_files:
                    results_file = result_files[0]
                    latest_run_dir = run_dir
                    print(f"Found results file in {run_dir}")
                    break
            
            if not results_file:
                print(f"No results file found in any run directory for model {model_name}")
                return {}
            
            print(f"Loading results from {results_file}")
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Generate visualizations
            run_timestamp = latest_run_dir.name.replace("run_", "")
            generate_visualizations(results, latest_run_dir, f"{model_name}_{run_timestamp}")
            
            return results
        
        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run-specific directory
        run_output_path = model_output_path / f"run_{timestamp}"
        run_output_path.mkdir(exist_ok=True, parents=True)
        print(f"Created run-specific directory: {run_output_path}")
        
        # Create subdirectories for different types of outputs
        raw_data_path = run_output_path / "raw_data"
        raw_data_path.mkdir(exist_ok=True)
        print(f"Created raw data directory: {raw_data_path}")
        
        visualizations_path = run_output_path / "visualizations"
        visualizations_path.mkdir(exist_ok=True)
        print(f"Created visualizations directory: {visualizations_path}")
        
        # If no categories specified, test all of them
        if prompt_categories is None:
            prompt_categories = list(PROMPT_CATEGORIES.keys())
        
        # Initialize results dictionary to store all data
        results = {
            "metadata": {
                "timestamp": timestamp,
                "model": model_name,
                "model_family": model_family,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "evaluator_model": evaluator_model,
                "questions": questions,
                "prompt_categories": prompt_categories,
                "is_mock": use_mock,
                "baseline_mode": baseline_mode  # Add baseline_mode to metadata
            },
            "responses": {},  # Will store all responses
            "metrics": {},    # Will store all metrics
            "summary": {}     # Will store summary statistics
        }
        
        # Initialize evaluator LLM for advanced metrics
        print(f"Initializing evaluator model: {evaluator_model}")
        evaluator_config = Config(
            model_name=evaluator_model,
            temperature=0.2,  # Lower temperature for more consistent evaluations
            max_tokens=1000
        )
        
        # Use mock implementation if specified
        if use_mock:
            print("Using mock implementation for OllamaInterface")
            evaluator_llm = MockOllamaInterface(evaluator_config)
        else:
            try:
                evaluator_llm = OllamaInterface(evaluator_config)
            except Exception as e:
                print(f"Error initializing OllamaInterface: {e}")
                print("Falling back to mock implementation")
                evaluator_llm = MockOllamaInterface(evaluator_config)
        
        # Run tests for each prompt category
        for category in prompt_categories:
            print(f"\n{'='*50}")
            print(f"Testing with {category} prompt")
            print(f"{'='*50}")
            
            # Get the specialized prompt for this category
            if baseline_mode:
                # Use a minimal prompt for baseline mode
                minimal_prompt = """
You are Viktor, a brilliant scientist from the show Arcane. You are pragmatic, determined, and stoic.
You speak with precise technical language and tend toward brevity.
"""
                specialized_prompt = minimal_prompt
                print("Using minimal baseline prompt")
            else:
                specialized_prompt = get_specialized_prompt(category, evaluator_config)
            
            # Initialize ViktorAI with this prompt
            config = Config(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Create a custom ViktorAI instance with the specialized prompt
            try:
                if use_mock:
                    # Create a mock implementation
                    viktor_ai = create_custom_viktor_ai(config, specialized_prompt)
                    # Replace the LLM with a mock implementation
                    viktor_ai.llm = MockOllamaInterface(config)
                else:
                    viktor_ai = create_custom_viktor_ai(config, specialized_prompt)
            except Exception as e:
                print(f"Error creating ViktorAI: {e}")
                print("Skipping category: {category}")
                continue
            
            # Initialize results for this category
            results["responses"][category] = []
            results["metrics"][category] = []
            
            # Test each question
            for i, question in enumerate(questions):
                print(f"\nQuestion {i+1}/{len(questions)}: {question}")
                
                try:
                    # Generate response
                    start_time = time.time()
                    response = viktor_ai.generate_response(question)
                    response_time = time.time() - start_time
                    
                    # Store results without evaluation for now
                    results["responses"][category].append({
                        "question": question,
                        "response": response,
                        "response_time": response_time
                    })
                    
                    # Print progress
                    print(f"A: {response[:100]}..." if len(response) > 100 else f"A: {response}")
                    print(f"Time: {response_time:.2f}s")
                except Exception as e:
                    print(f"Error processing question: {e}")
                    # Add a placeholder response
                    results["responses"][category].append({
                        "question": question,
                        "response": f"Error: {str(e)}",
                        "response_time": 0.0
                    })
            
            # Initialize metrics for this category
            results["metrics"][category] = []
            
            # Now evaluate all responses at once
            print(f"\nEvaluating all responses for {category} with {evaluator_model}...")
            for i, response_data in enumerate(results["responses"][category]):
                question = response_data["question"]
                response = response_data["response"]
                response_time = response_data["response_time"]
                
                try:
                    # Evaluate response
                    metrics = EvaluationMetrics.evaluate_response(
                        response, question, category, evaluator_llm
                    )
                    metrics["response_time"] = response_time
                    
                    # Store metrics
                    results["metrics"][category].append(metrics)
                    
                    # Print progress
                    print(f"Evaluated Q{i+1}: Score: {metrics.get('overall_score', 'N/A')}/10")
                except Exception as e:
                    print(f"Error evaluating response {i+1}: {e}")
                    # Add placeholder metrics
                    results["metrics"][category].append({
                        "overall_score": 0.0,
                        "overall_reasoning": f"Error evaluating response: {str(e)}",
                        "authenticity_score": 0.0,
                        "authenticity_reasoning": "Error evaluating response",
                        "technical_score": 0.0,
                        "technical_reasoning": "Error evaluating response",
                        "emotional_score": 0.0,
                        "emotional_reasoning": "Error evaluating response",
                        "quality_score": 0.0,
                        "quality_reasoning": "Error evaluating response",
                        "response_time": response_time
                    })
                
        # Calculate summary statistics
        results["summary"] = calculate_summary_statistics(results["metrics"])
        
        # Save results to raw_data directory
        results_file = raw_data_path / f"benchmark_results_{model_name}_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_file}")
        
        # Generate visualizations in the visualizations directory
        generate_visualizations(results, run_output_path, f"{model_name}_{timestamp}")
        
        return results
    except Exception as e:
        print(f"Error in run_benchmark: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_visualizations(results: Dict[str, Any], output_dir: Path, prefix: str) -> None:
    """
    Generate visualizations from benchmark results.
    
    This creates various charts and tables to help visualize the differences
    between prompt categories.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save visualizations (run-specific directory)
        prefix: Prefix for filenames
    """
    # Create visualizations directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data for plotting
    categories = list(results["metrics"].keys())
    
    # Prepare data for overall scores
    overall_scores = []
    for category in categories:
        for metrics in results["metrics"][category]:
            if "overall_score" in metrics:
                overall_scores.append({
                    "category": category,
                    "overall_score": metrics["overall_score"],
                    "weighted_score": metrics.get("weighted_score", metrics["overall_score"]),
                    "authenticity_score": metrics.get("authenticity_score", 0),
                    "technical_score": metrics.get("technical_score", 0),
                    "emotional_score": metrics.get("emotional_score", 0),
                    "quality_score": metrics.get("quality_score", 0),
                    "question_type": metrics.get("question_type", "unknown"),
                })
    
    if overall_scores:
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(overall_scores)
        
        # Plot overall scores by category
        plt.figure(figsize=(12, 8))
        numeric_columns = ["overall_score", "weighted_score", "authenticity_score", "technical_score", 
                          "emotional_score", "quality_score"]
        boxplot = df.boxplot(column=numeric_columns, by="category", 
                            rot=45, figsize=(12, 8))
        plt.title("Score Distribution by Prompt Category")
        plt.suptitle("")  # Remove pandas default suptitle
        plt.tight_layout()
        plt.savefig(vis_dir / f"{prefix}_scores_by_category.png")
        
        # Plot average scores by category
        avg_scores = df.groupby("category")[numeric_columns].mean().reset_index()
        
        # Bar chart for average overall scores
        plt.figure(figsize=(10, 6))
        plt.bar(avg_scores["category"], avg_scores["overall_score"], color='skyblue', label='Overall Score')
        plt.bar(avg_scores["category"], avg_scores["weighted_score"], color='lightgreen', alpha=0.7, label='Weighted Score')
        plt.title("Average Scores by Prompt Category")
        plt.xlabel("Prompt Category")
        plt.ylabel("Average Score (1-10)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(vis_dir / f"{prefix}_avg_overall_score.png")
        
        # Create a comparison table
        comparison_table = avg_scores.set_index("category")
        comparison_table.to_csv(vis_dir / f"{prefix}_score_comparison.csv")
        
        # Plot scores by question type
        if 'question_type' in df.columns:
            plt.figure(figsize=(12, 8))
            question_type_scores = df.groupby("question_type")[numeric_columns].mean().reset_index()
            
            # Bar chart for scores by question type
            plt.figure(figsize=(10, 6))
            x = range(len(question_type_scores))
            width = 0.15
            
            plt.bar([i - width*2 for i in x], question_type_scores["overall_score"], width=width, color='skyblue', label='Overall')
            plt.bar([i - width for i in x], question_type_scores["weighted_score"], width=width, color='lightgreen', label='Weighted')
            plt.bar([i for i in x], question_type_scores["authenticity_score"], width=width, color='coral', label='Authenticity')
            plt.bar([i + width for i in x], question_type_scores["technical_score"], width=width, color='gold', label='Technical')
            plt.bar([i + width*2 for i in x], question_type_scores["emotional_score"], width=width, color='orchid', label='Emotional')
            
            plt.title("Average Scores by Question Type")
            plt.xlabel("Question Type")
            plt.ylabel("Average Score (1-10)")
            plt.xticks([i for i in x], question_type_scores["question_type"], rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(vis_dir / f"{prefix}_scores_by_question_type.png")
        
        # Create a detailed HTML report
        create_html_report(results, vis_dir / f"{prefix}_report.html")
        
        # Create a detailed Markdown report
        create_markdown_report(results, vis_dir / f"{prefix}_report.md")

def get_available_models():
    """Get a list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            print(f"Warning: Failed to get models from Ollama API (status code: {response.status_code})")
            return ["llama3"]  # Default fallback
    except Exception as e:
        print(f"Warning: Failed to connect to Ollama API: {e}")
        print("Using default model list")
        return ["llama3"]  # Default fallback

def parse_arguments():
    """Parse command-line arguments for the benchmark script."""
    import argparse
    
    # Get available models
    available_models = get_available_models()
    
    parser = argparse.ArgumentParser(description="Run benchmarks for ViktorAI with different prompts")
    
    parser.add_argument("--model", type=str, default="llama3",
                        help=f"Name of the Ollama model to test (default: llama3, available: {', '.join(available_models)})")
    
    parser.add_argument("--evaluator-model", type=str, default="llama3",
                        help=f"Name of the model to use for evaluation (default: llama3, available: {', '.join(available_models)})")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature setting for response generation (default: 0.7)")
    
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum tokens for response generation (default: 500)")
    
    parser.add_argument("--questions-file", type=str, default="tests/model_test_questions.txt",
                        help="Path to file containing test questions (default: tests/model_test_questions.txt)")
    
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results (default: benchmark_results)")
    
    parser.add_argument("--categories", type=str, nargs="+",
                        help="Prompt categories to test (default: all)")
    
    parser.add_argument("--visualize-only", action="store_true",
                        help="Only generate visualizations from existing results")
    
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock implementations instead of real LLM (for testing)")
    
    parser.add_argument("--baseline-mode", action="store_true",
                        help="Use a minimal prompt regardless of category to establish a baseline")
    
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    
    parser.add_argument("--compare-versions", type=str,
                        help="Generate a comparison report for all versions of a model (e.g., 'gemma3')")
    
    args = parser.parse_args()
    
    # If --list-models is specified, print available models and exit
    if args.list_models:
        print("Available models:")
        for model in available_models:
            print(f"  - {model}")
        sys.exit(0)
    
    return args

def load_test_questions(file_path):
    """Load test questions from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        print(f"Warning: Questions file {file_path} not found. Using default questions.")
        return ["Who are you?", "What do you know about AI?", "Tell me about your family."]

def generate_version_comparison(family_dir: Path, output_dir: Path) -> None:
    """
    Generate a comparison report for all versions of a model.
    
    Args:
        family_dir: Directory containing model version subdirectories
        output_dir: Directory to save the comparison report
    """
    # Find all model version directories
    model_dirs = [d for d in family_dir.iterdir() if d.is_dir()]
    if len(model_dirs) <= 1:
        print(f"Not enough model versions found in {family_dir} to generate comparison")
        return
    
    print(f"Generating comparison for {len(model_dirs)} model versions in {family_dir}")
    
    # Collect the latest results for each model version
    model_results = {}
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        # Find the most recent run directory
        run_dirs = list(model_dir.glob("run_*"))
        if not run_dirs:
            print(f"No results found for model {model_name}")
            continue
        
        # Sort by modification time (newest first)
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_run_dir = run_dirs[0]
        
        # Find the results file in the raw_data directory
        results_files = list((latest_run_dir / "raw_data").glob("benchmark_results_*.json"))
        if not results_files:
            print(f"No results file found in {latest_run_dir / 'raw_data'}")
            continue
        
        latest_file = results_files[0]
        
        # Load the results
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                model_results[model_name] = results
        except Exception as e:
            print(f"Error loading results for {model_name}: {e}")
    
    if not model_results:
        print("No results found for any model version")
        return
    
    # Create the comparison directory
    comparison_dir = family_dir / "version_comparisons"
    comparison_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract the family name from the first model
    family_name = next(iter(model_results.keys())).split(':')[0]
    
    # Generate a timestamp for the comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the comparison report
    report_file = comparison_dir / f"{family_name}_comparison_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# {family_name.upper()} Model Versions Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Create a table of model versions and their average scores
        f.write("## Overall Scores\n\n")
        f.write("| Model | Overall Score | Authenticity | Technical | Emotional | Quality | Avg Response Time |\n")
        f.write("|-------|--------------|--------------|-----------|-----------|---------|-------------------|\n")
        
        for model_name, results in model_results.items():
            # Calculate average scores across all categories and questions
            avg_scores = {
                "overall_score": 0,
                "authenticity_score": 0,
                "technical_score": 0,
                "emotional_score": 0,
                "quality_score": 0,
                "response_time": 0
            }
            
            count = 0
            
            for category in results["metrics"]:
                for metrics in results["metrics"][category]:
                    for key in avg_scores:
                        if key in metrics:
                            avg_scores[key] += metrics[key]
                    count += 1
            
            # Calculate averages
            if count > 0:
                for key in avg_scores:
                    avg_scores[key] /= count
            
            # Format the response time
            response_time = f"{avg_scores['response_time']:.2f}s"
            
            # Add the row to the table
            f.write(f"| {model_name} | {avg_scores['overall_score']:.2f} | {avg_scores['authenticity_score']:.2f} | ")
            f.write(f"{avg_scores['technical_score']:.2f} | {avg_scores['emotional_score']:.2f} | ")
            f.write(f"{avg_scores['quality_score']:.2f} | {response_time} |\n")
        
        # Add a section for each category
        f.write("\n## Scores by Category\n\n")
        
        # Get all categories from the first model
        first_model = next(iter(model_results.values()))
        categories = first_model["metrics"].keys()
        
        for category in categories:
            f.write(f"\n### {category}\n\n")
            f.write("| Model | Overall Score | Authenticity | Technical | Emotional | Quality | Avg Response Time |\n")
            f.write("|-------|--------------|--------------|-----------|-----------|---------|-------------------|\n")
            
            for model_name, results in model_results.items():
                if category not in results["metrics"]:
                    continue
                
                # Calculate average scores for this category
                avg_scores = {
                    "overall_score": 0,
                    "authenticity_score": 0,
                    "technical_score": 0,
                    "emotional_score": 0,
                    "quality_score": 0,
                    "response_time": 0
                }
                
                count = len(results["metrics"][category])
                
                for metrics in results["metrics"][category]:
                    for key in avg_scores:
                        if key in metrics:
                            avg_scores[key] += metrics[key]
                
                # Calculate averages
                if count > 0:
                    for key in avg_scores:
                        avg_scores[key] /= count
                
                # Format the response time
                response_time = f"{avg_scores['response_time']:.2f}s"
                
                # Add the row to the table
                f.write(f"| {model_name} | {avg_scores['overall_score']:.2f} | {avg_scores['authenticity_score']:.2f} | ")
                f.write(f"{avg_scores['technical_score']:.2f} | {avg_scores['emotional_score']:.2f} | ")
                f.write(f"{avg_scores['quality_score']:.2f} | {response_time} |\n")
        
        # Add a section for sample responses and evaluations
        f.write("\n## Sample Responses and Evaluations\n\n")
        
        # Get a sample question from each category
        sample_questions = {}
        for category in categories:
            for model_name, results in model_results.items():
                if category in results["responses"] and results["responses"][category]:
                    # Get the first question in this category
                    sample_questions[category] = results["responses"][category][0]["question"]
                    break
        
        # For each sample question, show responses from all models
        for category, question in sample_questions.items():
            f.write(f"\n### Sample from {category}\n\n")
            f.write(f"**Question:** {question}\n\n")
            
            for model_name, results in model_results.items():
                if category not in results["responses"] or not results["responses"][category]:
                    continue
                
                # Find the response to this question
                response_data = None
                metrics_data = None
                
                for i, resp in enumerate(results["responses"][category]):
                    if resp["question"] == question:
                        response_data = resp
                        metrics_data = results["metrics"][category][i]
                        break
                
                if not response_data:
                    continue
                
                f.write(f"#### {model_name}\n\n")
                f.write(f"**Response Time:** {response_data['response_time']:.2f}s\n\n")
                f.write("**Response:**\n\n")
                f.write(f"{response_data['response']}\n\n")
                
                # Add evaluator's scores and reasoning
                if metrics_data:
                    f.write("**Evaluation:**\n\n")
                    f.write(f"- **Overall Score:** {metrics_data.get('overall_score', 0):.1f}/10\n")
                    if 'overall_reasoning' in metrics_data:
                        f.write(f"  - *{metrics_data['overall_reasoning']}*\n")
                    
                    f.write(f"- **Authenticity Score:** {metrics_data.get('authenticity_score', 0):.1f}/10\n")
                    if 'authenticity_reasoning' in metrics_data:
                        f.write(f"  - *{metrics_data['authenticity_reasoning']}*\n")
                    
                    f.write(f"- **Technical Score:** {metrics_data.get('technical_score', 0):.1f}/10\n")
                    if 'technical_reasoning' in metrics_data:
                        f.write(f"  - *{metrics_data['technical_reasoning']}*\n")
                    
                    f.write(f"- **Emotional Score:** {metrics_data.get('emotional_score', 0):.1f}/10\n")
                    if 'emotional_reasoning' in metrics_data:
                        f.write(f"  - *{metrics_data['emotional_reasoning']}*\n")
                    
                    f.write(f"- **Quality Score:** {metrics_data.get('quality_score', 0):.1f}/10\n")
                    if 'quality_reasoning' in metrics_data:
                        f.write(f"  - *{metrics_data['quality_reasoning']}*\n")
                
                f.write("\n---\n\n")
    
    print(f"Version comparison report saved to {report_file}")
    
    # Also generate a CSV file for easier data analysis
    csv_file = comparison_dir / f"{family_name}_comparison_{timestamp}.csv"
    
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Model,Category,Overall Score,Authenticity,Technical,Emotional,Quality,Response Time\n")
        
        for model_name, results in model_results.items():
            for category in results["metrics"]:
                # Calculate average scores for this category
                avg_scores = {
                    "overall_score": 0,
                    "authenticity_score": 0,
                    "technical_score": 0,
                    "emotional_score": 0,
                    "quality_score": 0,
                    "response_time": 0
                }
                
                count = len(results["metrics"][category])
                
                for metrics in results["metrics"][category]:
                    for key in avg_scores:
                        if key in metrics:
                            avg_scores[key] += metrics[key]
                
                # Calculate averages
                if count > 0:
                    for key in avg_scores:
                        avg_scores[key] /= count
                
                # Add the row to the CSV
                f.write(f"{model_name},{category},{avg_scores['overall_score']:.2f},{avg_scores['authenticity_score']:.2f},")
                f.write(f"{avg_scores['technical_score']:.2f},{avg_scores['emotional_score']:.2f},")
                f.write(f"{avg_scores['quality_score']:.2f},{avg_scores['response_time']:.2f}\n")
    
    print(f"Version comparison CSV saved to {csv_file}")
    
    return report_file

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
            baseline_mode=args.baseline_mode
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
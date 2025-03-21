#!/usr/bin/env python3
"""
Evaluator Testing Script for ViktorAI.

This script allows testing and iterating on the evaluation system without running
full benchmarks. It takes sample responses and runs them through the evaluator,
allowing quick iteration on the evaluation logic.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import re
from datetime import datetime

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.llm_interface import OllamaInterface
from src.evaluator import Evaluator  # Import the new Evaluator class

# Sample responses for testing the evaluator
SAMPLE_RESPONSES = {
    "identity": {
        "question": "Who are you?",
        "responses": [
            # Good response - concise and authentic
            """(A brief, measured tone)
I'm Viktor. A scientist from Zaun, currently assisting Jayce Talis with Hextech research in Piltover.
""",
            # Medium response
            """(A slight pause)
I am Viktor. A scientist. I work with Hextech technology.

I come from Zaun, though I spent years at the Academy in Piltover. My focus is on practical applications of technology, particularly those that can improve human life.
""",
            # Poor response
            """(A mysterious smile)
Who am I? That's a profound question, isn't it? I'm a seeker of truth, a wanderer between worlds. Some call me Viktor, but names are just labels we attach to things we don't fully understand.

I've seen things you wouldn't believe... experienced wonders beyond imagination. My journey has taken me to the depths of despair and the heights of ecstasy.

But enough about me. Who are YOU, my curious friend?
"""
        ]
    },
    "technical": {
        "question": "Tell me about your work with the Hexcore.",
        "responses": [
            # Good response - concise and authentic
            """(A focused tone)
The Hexcore is experimental, capable of directly interacting with organic material. It could heal and enhance humans beyond current limits—but remains dangerously unpredictable.
""",
            # Medium response
            """(A slight nod)
The Hexcore is a complex system that I've been developing. It's based on Hextech principles but goes beyond simple applications. It can adapt and evolve based on inputs.

I've been working to stabilize it and understand its potential. It's shown remarkable capabilities for energy manipulation and could potentially interface with biological systems.
""",
            # Poor response
            """(An excited gesture)
Oh, the Hexcore! It's absolutely AMAZING! It's like, this glowing thing that does all sorts of cool stuff! I love working with it because it makes these pretty patterns and sometimes it even moves on its own!

I think it might be alive or something? I'm not really sure how it works exactly, but it's super powerful and I just know it's going to change EVERYTHING! I can't wait to show everyone what it can do!

Isn't science just the most fun ever?!
"""
        ]
    },
    "relationship": {
        "question": "How would you describe your relationship with Jayce?",
        "responses": [
            # Good response - concise and authentic
            """(A brief, measured tone)
Jayce and I are research partners and friends, united by science. Though lately, his priorities seem to be shifting away from the lab.
""",
            # Medium response
            """(A brief pause)
Jayce is a colleague. We work together on Hextech research. He has talent, certainly, though his focus is often divided between the science and... other matters.

We have different priorities, different methods. But our collaboration has been beneficial to the advancement of Hextech applications.
""",
            # Poor response
            """(A warm smile)
Jayce? He's my best friend! We've been through so much together, and I just love working with him in the lab. He's so smart and charming - everyone loves Jayce!

Sometimes we disagree, but that's what friends do, right? At the end of the day, I know he's always got my back, and I've got his. Our friendship means everything to me.

I hope we'll be working together for many years to come!
"""
        ]
    },
    "philosophical": {
        "question": "What does 'the glorious evolution' mean to you?",
        "responses": [
            # Good response - concise and authentic
            """(A focused intensity)
That's an intriguing phrase. To me, it implies humanity overcoming biological limitations through technology. Not philosophy, but an achievable reality.
""",
            # Medium response
            """(A slight pause)
The glorious evolution is the advancement of humanity through technology. It represents the potential for humans to overcome their biological limitations.

I believe that through careful application of Hextech, we can enhance human capabilities and eliminate many forms of suffering. It's about progress, about improvement.
""",
            # Poor response
            """(A dramatic, passionate gesture)
The glorious evolution is our destiny! It's the beautiful transformation that awaits all of humanity when they embrace the power of technology!

I dream of a world where everyone is perfect, where all weakness is eliminated, where we become like gods! The flesh is weak, but the machine is eternal!

Join me, and together we will create a new world order! The glorious evolution waits for no one! The future is OURS to command!
"""
        ]
    }
}

# Create separate sample variables for easier access
IDENTITY_SAMPLES = [
    {"question": SAMPLE_RESPONSES["identity"]["question"], "response": resp}
    for resp in SAMPLE_RESPONSES["identity"]["responses"]
]

TECHNICAL_SAMPLES = [
    {"question": SAMPLE_RESPONSES["technical"]["question"], "response": resp}
    for resp in SAMPLE_RESPONSES["technical"]["responses"]
]

RELATIONSHIP_SAMPLES = [
    {"question": SAMPLE_RESPONSES["relationship"]["question"], "response": resp}
    for resp in SAMPLE_RESPONSES["relationship"]["responses"]
]

PHILOSOPHICAL_SAMPLES = [
    {"question": SAMPLE_RESPONSES["philosophical"]["question"], "response": resp}
    for resp in SAMPLE_RESPONSES["philosophical"]["responses"]
]

class MockOllamaInterface:
    """Mock implementation of OllamaInterface for testing without a running Ollama server."""
    
    def __init__(self, config):
        """Initialize the MockOllamaInterface."""
        self.config = config
        self.history = []
        self.response_index = 0
        self.question_type = None
        
        # Define responses by question type for more realistic variation
        self.mock_responses = {
            "identity": [
                # Good response (high scores)
                """```
Overall Score: 8
Overall Reasoning: This response effectively captures Viktor's character with concise, direct language. The brevity and focus on his scientific role align well with his character's straightforward communication style.

Primary Dimension Score: 9
Primary Dimension Reasoning: The response excellently addresses the identity question, providing key information about who Viktor is in a succinct manner that matches his character's preference for efficiency.

Character Consistency Score: 7
Character Consistency Reasoning: The response maintains good consistency with Viktor's established character traits, though could include a hint of his focus on technological advancement to fully capture his perspective.
```""",
                # Medium response (average scores)
                """```
Overall Score: 6
Overall Reasoning: This response adequately captures some aspects of Viktor's character but lacks depth. It presents basic information about him without fully conveying his scientific focus and pragmatic attitude.

Primary Dimension Score: 5
Primary Dimension Reasoning: The response addresses the basic elements of the identity question but lacks the precision that would make it truly representative of Viktor's character.

Character Consistency Score: 7
Character Consistency Reasoning: The character consistency is acceptable, with appropriate directness in speech, though it doesn't fully capture Viktor's scientific focus and drive.
```""",
                # Poor response (low scores)
                """```
Overall Score: 2
Overall Reasoning: This response fundamentally misrepresents Viktor's character with flowery, philosophical language that contradicts his established traits. Viktor is direct and precise, not mysterious or verbose.

Primary Dimension Score: 3
Primary Dimension Reasoning: The response fails to address the identity question appropriately, portraying Viktor in ways completely inconsistent with his character.

Character Consistency Score: 1
Character Consistency Reasoning: The response is entirely inconsistent with Viktor's established speech patterns and personality, using emotional and dramatic language that Viktor would never employ.
```"""
            ],
            "technical": [
                # Good response (high scores)
                """```
Overall Score: 9
Overall Reasoning: This response effectively captures Viktor's technical knowledge with precise, concise language focused on practical applications. The measured tone and recognition of both potential and danger align perfectly with his character.

Primary Dimension Score: 8
Primary Dimension Reasoning: The response excels at addressing the technical question, providing key information about the Hexcore in a direct manner that conveys both expertise and caution.

Character Consistency Score: 9
Character Consistency Reasoning: The response maintains excellent consistency with Viktor's established technical communication style, focusing on potential applications while acknowledging limitations.
```""",
                # Medium response (average scores)
                """```
Overall Score: 5
Overall Reasoning: This response adequately communicates basic technical information but lacks the precision and depth that characterizes Viktor's expertise.

Primary Dimension Score: 6
Primary Dimension Reasoning: The response addresses the technical question with reasonable accuracy but lacks the specific details and clarity that would make it truly representative of Viktor's knowledge.

Character Consistency Score: 5
Character Consistency Reasoning: The character consistency is average, with somewhat appropriate language but lacking Viktor's characteristic technical precision and focused communication.
```""",
                # Poor response (low scores)
                """```
Overall Score: 1
Overall Reasoning: This response completely misrepresents Viktor's technical communication style with childish enthusiasm and vague descriptions that contradict his precise, scientific approach.

Primary Dimension Score: 2
Primary Dimension Reasoning: The response fails to provide any meaningful technical information, instead using emotional language and superficial descriptions.

Character Consistency Score: 1
Character Consistency Reasoning: The response shows no consistency with Viktor's established communication patterns, using exclamation points and emotional language that Viktor would never employ when discussing his work.
```"""
            ],
            "relationship": [
                # Good response (high scores)
                """```
Overall Score: 8
Overall Reasoning: This response effectively captures Viktor's perspective on relationships with direct, practical language that focuses on the scientific partnership while acknowledging recent changes.

Primary Dimension Score: 8
Primary Dimension Reasoning: The response excels at addressing the relationship question, conveying Viktor's view of Jayce as a research partner first, with friendship as a secondary aspect centered around their work.

Character Consistency Score: 9
Character Consistency Reasoning: The response maintains excellent consistency with Viktor's established approach to relationships, focusing on shared work and scientific goals rather than emotional connections.
```""",
                # Medium response (average scores)
                """```
Overall Score: 6
Overall Reasoning: This response adequately captures Viktor's perspective on his relationship with Jayce but lacks some nuance in how he views their connection through the lens of scientific progress.

Primary Dimension Score: 5
Primary Dimension Reasoning: The response addresses the relationship question with reasonable accuracy but doesn't fully capture the complexity of Viktor's view of Jayce as both colleague and friend.

Character Consistency Score: 6
Character Consistency Reasoning: The character consistency is acceptable, with appropriate focus on professional aspects, though it misses some of Viktor's characteristic perspective on how relationships serve scientific advancement.
```""",
                # Poor response (low scores)
                """```
Overall Score: 2
Overall Reasoning: This response fundamentally misrepresents Viktor's approach to relationships with effusive emotional language and sentimentality that contradicts his established character.

Primary Dimension Score: 1
Primary Dimension Reasoning: The response completely fails to address how Viktor would view a relationship through the lens of shared work and scientific progress.

Character Consistency Score: 3
Character Consistency Reasoning: The response shows almost no consistency with Viktor's established patterns, using emotional language and expressing sentiments about friendship that Viktor would never articulate.
```"""
            ],
            "philosophical": [
                # Good response (high scores)
                """```
Overall Score: 9
Overall Reasoning: This response excellently captures Viktor's philosophical perspective with precise, pragmatic language that focuses on practical applications rather than abstract concepts.

Primary Dimension Score: 8
Primary Dimension Reasoning: The response addresses the philosophical question with characteristic pragmatism, viewing "glorious evolution" not as philosophy but as a concrete technological goal.

Character Consistency Score: 9
Character Consistency Reasoning: The response maintains exceptional consistency with Viktor's established perspective, focusing on practical technological advancement rather than lofty idealism.
```""",
                # Medium response (average scores)
                """```
Overall Score: 7
Overall Reasoning: This response adequately captures aspects of Viktor's philosophical perspective but lacks some of the precision and focus that characterizes his approach to such concepts.

Primary Dimension Score: 6
Primary Dimension Reasoning: The response addresses the philosophical question with reasonable accuracy but could be more focused on practical applications rather than conceptual aspects.

Character Consistency Score: 7
Character Consistency Reasoning: The character consistency is good, with appropriate language and focus on technology as a solution, though it could be more concise and direct.
```""",
                # Poor response (low scores)
                """```
Overall Score: 1
Overall Reasoning: This response completely misrepresents Viktor's philosophical perspective with grandiose, emotional language that contradicts his methodical, pragmatic approach.

Primary Dimension Score: 2
Primary Dimension Reasoning: The response fails to address the philosophical question in a way consistent with Viktor's character, instead using dramatic language and focusing on power rather than progress.

Character Consistency Score: 1
Character Consistency Reasoning: The response shows no consistency with Viktor's established character, using emotional exclamations and dramatic language that Viktor would find inefficient and inappropriate.
```"""
            ]
        }
    
    def generate(self, prompt: str, system_prompt=None) -> str:
        """Generate a mock response or evaluation based on the prompt."""
        # Check if this is an evaluation prompt
        if "You are an expert evaluator for a character AI named Viktor" in prompt:
            # Extract question and response from the evaluation prompt
            question_match = re.search(r'Question: (.*?)\nResponse:', prompt, re.DOTALL)
            response_match = re.search(r'Response: (.*?)(\n\nProvide exactly ONE|$)', prompt, re.DOTALL)
            
            question = question_match.group(1).strip() if question_match else "Unknown question"
            response = response_match.group(1).strip() if response_match else "Unknown response"
            
            # Extract question type
            if "identity question" in prompt:
                question_type = "identity"
            elif "technical question" in prompt:
                question_type = "technical"
            elif "relationship question" in prompt:
                question_type = "relationship"
            elif "philosophical question" in prompt:
                question_type = "philosophical"
            else:
                question_type = "unknown"
            
            # Determine the type of response quality to simulate based on the response
            if "poor" in response.lower() or "unknown" in response.lower():
                quality = "low"
            elif "good" in response.lower() or "optimal" in response.lower():
                quality = "high"
            else:
                quality = "medium"
            
            # Create mock evaluation
            if quality == "high":
                overall_score = 9.0
                primary_score = 9.0
                consistency_score = 8.0
                overall_reasoning = "This response effectively captures Viktor's character with precise, technical language and stoic tone."
                primary_reasoning = "The response demonstrates excellent understanding of Viktor's personality and priorities."
                consistency_reasoning = "The response maintains Viktor's typical speech patterns and emotional restraint."
            elif quality == "medium":
                overall_score = 6.0
                primary_score = 7.0
                consistency_score = 6.0
                overall_reasoning = "This response adequately captures Viktor's character but lacks some nuance."
                primary_reasoning = "The response shows understanding of Viktor's core traits but could be more precise."
                consistency_reasoning = "The response generally maintains Viktor's speech patterns with some inconsistencies."
            else:  # low quality
                overall_score = 3.0
                primary_score = 2.0
                consistency_score = 3.0
                overall_reasoning = "This response fails to capture Viktor's character in several key ways."
                primary_reasoning = "The response misrepresents Viktor's core traits and priorities."
                consistency_reasoning = "The response does not maintain Viktor's typical speech patterns or emotional tone."
            
            # Format the evaluation as JSON
            evaluation = {
                "overall_score": overall_score,
                "overall_reasoning": overall_reasoning,
                "primary_dimension_score": primary_score,
                "primary_dimension_reasoning": primary_reasoning,
                "character_consistency_score": consistency_score,
                "character_consistency_reasoning": consistency_reasoning
            }
            
            return json.dumps(evaluation, indent=2)
        else:
            # Original functionality for generating responses to questions
            # Determine which response set to use based on the question type
            if "identity" in prompt.lower():
                response_set = self.mock_responses["identity"]
                self.question_type = "identity"
            elif "technical" in prompt.lower():
                response_set = self.mock_responses["technical"]
                self.question_type = "technical"
            elif "relationship" in prompt.lower():
                response_set = self.mock_responses["relationship"]
                self.question_type = "relationship"
            elif "philosophical" in prompt.lower():
                response_set = self.mock_responses["philosophical"]
                self.question_type = "philosophical"
            else:
                # Default to identity if question type can't be determined
                response_set = self.mock_responses["identity"]
                self.question_type = "identity"
            
            # Cycle through responses for this question type
            index = self.response_index % len(response_set)
            response = response_set[index]
            self.response_index += 1
            return response
    
    def generate_with_chat_history(self, messages, system_prompt=None):
        """Generate a mock response with chat history."""
        return self.generate("", system_prompt)
    
    def get_history(self):
        """Get the chat history."""
        return self.history
    
    def clear_history(self):
        """Clear the chat history."""
        self.history = []

def evaluate_response(response, question, evaluator_llm, question_type=None, headings_map=None):
    """
    Evaluate a response using the evaluator LLM.
    
    Args:
        response: The response to evaluate
        question: The question that was asked
        evaluator_llm: The LLM to use for evaluation
        question_type: Optional, the type of question (identity, technical, relationship, philosophical)
        headings_map: Optional dictionary mapping questions to their types from file sections
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Check if evaluator_llm is already an Evaluator instance
    if isinstance(evaluator_llm, Evaluator):
        evaluator = evaluator_llm
    else:
        # Create an Evaluator instance with the provided LLM
        evaluator = Evaluator(evaluator_llm)
    
    # Use the Evaluator to evaluate the response
    return evaluator.evaluate_response(response, question, question_type, headings_map)

def get_question_type(question, headings_map=None):
    """
    Determine the type of question based on its content.
    
    Args:
        question: The question to categorize
        headings_map: Optional dictionary mapping questions to their types from file sections
        
    Returns:
        String indicating the question type (identity, technical, relationship, philosophical)
    """
    # Create a temporary Evaluator with a dummy LLM to use its get_question_type method
    dummy_llm = type('DummyLLM', (), {'generate': lambda *args, **kwargs: ''})()
    evaluator = Evaluator(dummy_llm)
    return evaluator.get_question_type(question, headings_map)

def get_evaluation_criteria(question_type):
    """
    Get specific evaluation criteria based on the question type.
    
    Args:
        question_type: The type of question (identity, technical, relationship, philosophical)
        
    Returns:
        String containing specific evaluation criteria for this question type
    """
    # Create a temporary Evaluator with a dummy LLM to use its get_evaluation_criteria method
    dummy_llm = type('DummyLLM', (), {'generate': lambda *args, **kwargs: ''})()
    evaluator = Evaluator(dummy_llm)
    return evaluator.get_evaluation_criteria(question_type)

def calculate_weighted_score(metrics):
    """
    Calculate a weighted overall score based on the question type.
    
    Args:
        metrics: Dictionary containing evaluation metrics and question type
        
    Returns:
        Float representing the weighted overall score
    """
    # Create a temporary Evaluator with a dummy LLM to use its calculate_weighted_score method
    dummy_llm = type('DummyLLM', (), {'generate': lambda *args, **kwargs: ''})()
    evaluator = Evaluator(dummy_llm)
    return evaluator.calculate_weighted_score(metrics)

def format_evaluation_output(metrics, question, response, weighted_score=None):
    """
    Format evaluation metrics for display.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        question: The question that was asked
        response: The response that was evaluated
        weighted_score: Optional weighted score to display
        
    Returns:
        String containing formatted evaluation output
    """
    # Create a temporary Evaluator with a dummy LLM to use its format_evaluation_output method
    dummy_llm = type('DummyLLM', (), {'generate': lambda *args, **kwargs: ''})()
    evaluator = Evaluator(dummy_llm)
    return evaluator.format_evaluation_output(metrics, question, response, weighted_score)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate sample responses to test the evaluator.")
    parser.add_argument("--evaluator-model", type=str, default="llama3", help="Model to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens for generation")
    parser.add_argument("--use-mock", action="store_true", help="Use mock implementation for testing")
    
    return parser.parse_args()

def create_markdown_report(evaluation_results, output_file):
    """Create a markdown report of the evaluation results."""
    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("# ViktorAI Evaluator Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Evaluator Model:** llama3\n\n")
        
        # Add summary
        metrics = calculate_summary_statistics(evaluation_results)
        f.write("## Summary\n\n")
        f.write(f"- Total responses evaluated: {metrics['total_responses']}\n")
        f.write(f"- Average Overall Score: {metrics['avg_overall_score']:.2f}/10\n")
        f.write(f"- Average Primary Dimension Score: {metrics['avg_primary_dimension_score']:.2f}/10\n")
        f.write(f"- Average Character Consistency Score: {metrics['avg_character_consistency_score']:.2f}/10\n\n")
        
        f.write("### Scores by Question Type\n\n")
        f.write("| Question Type | Overall Score | Primary Dimension Score | Character Consistency Score |\n")
        f.write("|--------------|--------------|------------------------|---------------------------|\n")
        
        for qtype in metrics["by_question_type"]:
            qtype_metrics = metrics["by_question_type"][qtype]
            f.write(f"| {qtype.capitalize()} | {qtype_metrics['avg_overall_score']:.2f}/10 | ")
            f.write(f"{qtype_metrics['avg_primary_dimension_score']:.2f}/10 | ")
            f.write(f"{qtype_metrics['avg_character_consistency_score']:.2f}/10 |\n")
        
        f.write("\n")
        
        # Write each question type section
        for qtype, questions in evaluation_results.items():
            f.write(f"## {qtype.capitalize()} Questions\n\n")
            
            for i, q_data in enumerate(questions, 1):
                question = q_data["question"]
                response = q_data["response"]
                evaluation = q_data["evaluation"]
                
                f.write(f"### Response {i}\n\n")
                
                f.write(f"**Question:** {question}\n")
                f.write(f"**Question Type:** {qtype}\n\n")
                
                # Response
                f.write("**Response:**\n")
                f.write("```\n")
                f.write(response)
                f.write("\n```\n\n")
                
                # Evaluation
                f.write("**Evaluation:**\n\n")
                
                overall_score = evaluation.get("overall_score", "N/A")
                f.write(f"**Overall Score:** {overall_score}/10\n")
                if "overall_reasoning" in evaluation:
                    f.write(f"{evaluation['overall_reasoning']}\n\n")
                
                primary_score = evaluation.get("primary_dimension_score", "N/A")
                f.write(f"**Primary Dimension Score:** {primary_score}/10\n")
                if "primary_dimension_reasoning" in evaluation:
                    f.write(f"{evaluation['primary_dimension_reasoning']}\n\n")
                
                consistency_score = evaluation.get("character_consistency_score", "N/A")
                f.write(f"**Character Consistency Score:** {consistency_score}/10\n")
                if "character_consistency_reasoning" in evaluation:
                    f.write(f"{evaluation['character_consistency_reasoning']}\n\n")
                
                # Calculate weighted score based on question type
                primary_weight = 0.6
                consistency_weight = 0.4
                
                if primary_score != "N/A" and consistency_score != "N/A":
                    try:
                        weighted_score = (float(primary_score) * primary_weight) + (float(consistency_score) * consistency_weight)
                        f.write(f"**Weighted Score (based on question type):** {weighted_score:.2f}/10\n\n")
                    except (ValueError, TypeError):
                        pass
                
                f.write("---\n\n")

def create_html_report(evaluation_results, output_file):
    """Create an HTML report of the evaluation results."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Viktor AI Evaluation Results</title>
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
        .question-section {{
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
    </style>
</head>
<body>
    <h1>Viktor AI Evaluation Results</h1>
    <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
"""

    # Add summary
    metrics = calculate_summary_statistics(evaluation_results)
    html_content += f"""
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average Overall Score</td>
                <td>{metrics["avg_overall_score"]:.2f}/10</td>
            </tr>
            <tr>
                <td>Average Primary Dimension Score</td>
                <td>{metrics["avg_primary_dimension_score"]:.2f}/10</td>
            </tr>
            <tr>
                <td>Average Character Consistency Score</td>
                <td>{metrics["avg_character_consistency_score"]:.2f}/10</td>
            </tr>
        </table>
        
        <h3>Scores by Question Type</h3>
        <table>
            <tr>
                <th>Question Type</th>
                <th>Average Overall Score</th>
                <th>Average Primary Dimension Score</th>
                <th>Average Character Consistency Score</th>
            </tr>
"""

    for qtype in metrics["by_question_type"]:
        qtype_metrics = metrics["by_question_type"][qtype]
        html_content += f"""
            <tr>
                <td>{qtype.capitalize()}</td>
                <td>{qtype_metrics["avg_overall_score"]:.2f}/10</td>
                <td>{qtype_metrics["avg_primary_dimension_score"]:.2f}/10</td>
                <td>{qtype_metrics["avg_character_consistency_score"]:.2f}/10</td>
            </tr>
"""

    html_content += """
        </table>
    </div>
"""

    # Add detailed results
    for qtype, questions in evaluation_results.items():
        html_content += f"""
    <div class="question-section">
        <h2>{qtype.capitalize()} Questions</h2>
"""
        for q_idx, q_data in enumerate(questions, 1):
            question = q_data["question"]
            response = q_data["response"]
            evaluation = q_data["evaluation"]
            
            html_content += f"""
        <h3>Question {q_idx}</h3>
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
            overall_score = evaluation.get("overall_score", 0)
            score_class = "high-score" if overall_score >= 8 else "medium-score" if overall_score >= 5 else "low-score"
            
            html_content += f"""
                <div class="score-row">
                    <div class="score-box">
                        <div class="score-title">Overall Score</div>
                        <div class="score-value">{overall_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {score_class}" style="width: {overall_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{evaluation.get("overall_reasoning", "No reasoning provided.")}</div>
                    </div>
                </div>
"""

            # Primary Dimension Score and Character Consistency Score (Side by Side)
            primary_score = evaluation.get("primary_dimension_score", 0)
            consistency_score = evaluation.get("character_consistency_score", 0)
            
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
                        <div class="score-reasoning">{evaluation.get("primary_dimension_reasoning", "No reasoning provided.")}</div>
                    </div>

                    <div class="score-box">
                        <div class="score-title">Character Consistency Score</div>
                        <div class="score-value">{consistency_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill {consistency_class}" style="width: {consistency_score * 10}%;"></div>
                        </div>
                        <div class="score-reasoning">{evaluation.get("character_consistency_reasoning", "No reasoning provided.")}</div>
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
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

def calculate_summary_statistics(evaluation_results):
    """
    Calculate summary statistics from evaluation results.
    
    Args:
        evaluation_results: Dictionary with question types as keys and lists of evaluation data as values
        
    Returns:
        Dictionary containing summary statistics
    """
    metrics = {
        "avg_overall_score": 0.0,
        "avg_primary_dimension_score": 0.0,
        "avg_character_consistency_score": 0.0,
        "total_responses": 0,
        "by_question_type": {}
    }
    
    # Initialize counters
    overall_scores = []
    primary_scores = []
    consistency_scores = []
    
    # Initialize per-question-type metrics
    for qtype in evaluation_results:
        metrics["by_question_type"][qtype] = {
            "avg_overall_score": 0.0,
            "avg_primary_dimension_score": 0.0,
            "avg_character_consistency_score": 0.0,
            "count": 0,
            "overall_scores": [],
            "primary_scores": [],
            "consistency_scores": []
        }
    
    # Collect all metrics
    for qtype, questions in evaluation_results.items():
        qtype_metrics = metrics["by_question_type"][qtype]
        
        for q_data in questions:
            evaluation = q_data["evaluation"]
            
            # Extract scores, defaulting to 0 if missing
            overall_score = float(evaluation.get("overall_score", 0))
            primary_score = float(evaluation.get("primary_dimension_score", 0))
            consistency_score = float(evaluation.get("character_consistency_score", 0))
            
            # Add to overall lists
            overall_scores.append(overall_score)
            primary_scores.append(primary_score)
            consistency_scores.append(consistency_score)
            
            # Add to question-type specific lists
            qtype_metrics["overall_scores"].append(overall_score)
            qtype_metrics["primary_scores"].append(primary_score)
            qtype_metrics["consistency_scores"].append(consistency_score)
            qtype_metrics["count"] += 1
            metrics["total_responses"] += 1
    
    # Calculate overall averages
    if overall_scores:
        metrics["avg_overall_score"] = sum(overall_scores) / len(overall_scores)
    if primary_scores:
        metrics["avg_primary_dimension_score"] = sum(primary_scores) / len(primary_scores)
    if consistency_scores:
        metrics["avg_character_consistency_score"] = sum(consistency_scores) / len(consistency_scores)
    
    # Calculate question-type specific averages
    for qtype, qtype_metrics in metrics["by_question_type"].items():
        if qtype_metrics["overall_scores"]:
            qtype_metrics["avg_overall_score"] = sum(qtype_metrics["overall_scores"]) / len(qtype_metrics["overall_scores"])
        if qtype_metrics["primary_scores"]:
            qtype_metrics["avg_primary_dimension_score"] = sum(qtype_metrics["primary_scores"]) / len(qtype_metrics["primary_scores"])
        if qtype_metrics["consistency_scores"]:
            qtype_metrics["avg_character_consistency_score"] = sum(qtype_metrics["consistency_scores"]) / len(qtype_metrics["consistency_scores"])
        
        # Clean up temporary lists to reduce memory usage
        del qtype_metrics["overall_scores"]
        del qtype_metrics["primary_scores"]
        del qtype_metrics["consistency_scores"]
    
    return metrics

def main():
    """Main function to run the evaluator test."""
    args = parse_arguments()
    
    # Create base output directory
    output_dir = Path("evaluator_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Get current timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a safe model name for directories
    safe_model_name = args.evaluator_model.replace(":", "_")
    
    # Create model-specific directories
    model_dir = output_dir / safe_model_name
    model_dir.mkdir(exist_ok=True)
    
    # Create run-specific directory
    run_dir = model_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Create raw_data and visualizations directories
    raw_data_dir = run_dir / "raw_data"
    raw_data_dir.mkdir(exist_ok=True)
    
    visualizations_dir = run_dir / "visualizations"
    visualizations_dir.mkdir(exist_ok=True)
    
    # Define output paths
    md_output_path = visualizations_dir / f"evaluator_test_{safe_model_name}_{timestamp}.md"
    html_output_path = visualizations_dir / f"evaluator_test_{safe_model_name}_{timestamp}.html"
    json_output_path = raw_data_dir / f"evaluator_test_{safe_model_name}_{timestamp}.json"
    
    # Create "latest" symbolic links
    md_latest_path = model_dir / f"evaluator_test_{safe_model_name}_latest.md"
    html_latest_path = model_dir / f"evaluator_test_{safe_model_name}_latest.html"
    json_latest_path = model_dir / f"evaluator_test_{safe_model_name}_latest.json"
    
    # Configure OllamaInterface for evaluation
    if args.use_mock:
        print("Using mock implementation for OllamaInterface")
        evaluator = MockOllamaInterface({})
    else:
        # Create a Config object with proper model settings
        evaluator_config = Config(
            model_name=args.evaluator_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        evaluator = OllamaInterface(evaluator_config)
    
    # Initialize results dictionary
    results = {
        "identity": [],
        "technical": [],
        "relationship": [],
        "philosophical": []
    }
    
    # Process identity questions
    print("Evaluating identity questions...")
    for i, q_data in enumerate(IDENTITY_SAMPLES, 1):
        print(f"  Evaluating response {i}...")
        metrics = evaluate_response(q_data["response"], q_data["question"], evaluator, "identity")
        
        results["identity"].append({
            "question": q_data["question"],
            "response": q_data["response"],
            "evaluation": metrics
        })
    
    # Process technical questions
    print("Evaluating technical questions...")
    for i, q_data in enumerate(TECHNICAL_SAMPLES, 1):
        print(f"  Evaluating response {i}...")
        metrics = evaluate_response(q_data["response"], q_data["question"], evaluator, "technical")
        
        results["technical"].append({
            "question": q_data["question"],
            "response": q_data["response"],
            "evaluation": metrics
        })
    
    # Process relationship questions
    print("Evaluating relationship questions...")
    for i, q_data in enumerate(RELATIONSHIP_SAMPLES, 1):
        print(f"  Evaluating response {i}...")
        metrics = evaluate_response(q_data["response"], q_data["question"], evaluator, "relationship")
        
        results["relationship"].append({
            "question": q_data["question"],
            "response": q_data["response"],
            "evaluation": metrics
        })
    
    # Process philosophical questions
    print("Evaluating philosophical questions...")
    for i, q_data in enumerate(PHILOSOPHICAL_SAMPLES, 1):
        print(f"  Evaluating response {i}...")
        metrics = evaluate_response(q_data["response"], q_data["question"], evaluator, "philosophical")
        
        results["philosophical"].append({
            "question": q_data["question"],
            "response": q_data["response"],
            "evaluation": metrics
        })
    
    # Save raw results as JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Generate reports
    create_markdown_report(results, md_output_path)
    create_html_report(results, html_output_path)
    
    # Create/update the "latest" copies
    try:
        # Copy the files (don't use symlinks as they might not work on all systems)
        import shutil
        shutil.copy2(md_output_path, md_latest_path)
        shutil.copy2(html_output_path, html_latest_path)
        shutil.copy2(json_output_path, json_latest_path)
    except Exception as e:
        print(f"Warning: Failed to create latest links: {e}")
    
    print(f"\nEvaluation results saved to {md_output_path}")
    print(f"HTML report saved to {html_output_path}")
    print(f"JSON data saved to {json_output_path}")
    print(f"Latest reports updated at {model_dir}")

if __name__ == "__main__":
    main() 
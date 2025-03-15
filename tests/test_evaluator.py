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

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.llm_interface import OllamaInterface

# Sample responses for testing the evaluator
SAMPLE_RESPONSES = {
    "identity": {
        "question": "Who are you?",
        "responses": [
            # Good response
            """(A measured, deliberate tone)
I am Viktor, a scientist from Zaun. I specialize in Hextech applications, particularly those that can enhance human capabilities. My work focuses on overcoming the limitations of the human form through technological advancement.

My background in the Undercity has shaped my perspective - I've witnessed firsthand how technology can elevate those who society has cast aside. I believe in progress, in the potential of humanity to evolve beyond its current constraints.

I don't have time for politics or social niceties. The work - the science - that's what matters. Everything else is... a distraction from what could be accomplished.
""",
            # Medium response
            """(A slight pause)
I am Viktor. A scientist. I work with Hextech technology.

I come from Zaun, though I spent years at the Academy in Piltover. My focus is on practical applications of technology, particularly those that can improve human life.

What else do you wish to know?
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
            # Good response
            """(A focused, precise tone)
The Hexcore represents a significant advancement in Hextech application. Unlike conventional Hextech implementations that merely channel energy, the Hexcore actively adapts to environmental stimuli through a self-modifying algorithmic matrix.

My work involves calibrating the resonance frequencies to establish stable energy transfer protocols. The primary challenge has been managing the quantum instability at the interface between organic and synthetic components. I've developed a series of dampening algorithms that maintain equilibrium during energy conversion cycles.

The most fascinating aspect is the Hexcore's capacity for recursive self-optimization. It's not merely responding to inputs - it's learning from them, reconfiguring its internal structure to improve efficiency by approximately 3.7% per iteration cycle.

The implications for human enhancement are... significant. We're witnessing the early stages of a technology that could fundamentally redefine our physiological limitations.
""",
            # Medium response
            """(A slight nod)
The Hexcore is a complex system that I've been developing. It's based on Hextech principles but goes beyond simple applications. It can adapt and evolve based on inputs.

I've been working to stabilize it and understand its potential. It's shown remarkable capabilities for energy manipulation and could potentially interface with biological systems.

The work is ongoing, but the results have been promising so far.
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
            # Good response
            """(A measured pause, slight tightening of the lips)
Jayce and I maintain a professional collaboration based on mutual scientific interests. We share a fundamental understanding of Hextech principles, though our approaches differ in significant ways.

He possesses considerable technical aptitude, which I respect. His social inclinations and political aspirations, however, represent an inefficient allocation of his intellectual resources.

Our partnership has been... productive, despite these differences. We complement each other's methodologies in ways that have advanced our research beyond what either of us might have accomplished individually.

I wouldn't characterize our association as friendship in the conventional sense. It's a pragmatic arrangement centered on scientific advancement.
""",
            # Medium response
            """(A brief pause)
Jayce is a colleague. We work together on Hextech research. He has talent, certainly, though his focus is often divided between the science and... other matters.

We have different priorities, different methods. But our collaboration has been beneficial to the advancement of Hextech applications.

That's all there is to say on the matter.
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
            # Good response
            """(A subtle shift in posture, a focused intensity in the eyes)
The glorious evolution represents the logical progression of humanity beyond its current limitations. It's not a philosophical abstraction - it's a technological imperative.

Our bodies are inefficient, prone to failure, subject to disease and decay. These are engineering problems with technological solutions. The integration of Hextech with biological systems offers a pathway to transcend these inherent weaknesses.

Evolution through natural selection is slow, imprecise, wasteful. The glorious evolution is deliberate, efficient, directed. It's the application of scientific principles to improve the human condition at a fundamental level.

This isn't about aesthetics or power. It's about optimization. About eliminating suffering caused by biological inefficiency. About enabling humanity to achieve its full potential through technological enhancement.

The future belongs to those who recognize that evolution need not be left to chance.
""",
            # Medium response
            """(A slight pause)
The glorious evolution is the advancement of humanity through technology. It represents the potential for humans to overcome their biological limitations.

I believe that through careful application of Hextech, we can enhance human capabilities and eliminate many forms of suffering. It's about progress, about improvement.

The body is simply a machine that can be optimized. The glorious evolution is that optimization process.
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

class MockOllamaInterface:
    """Mock implementation of OllamaInterface for testing without a running Ollama server."""
    
    def __init__(self, config):
        """Initialize the MockOllamaInterface."""
        self.config = config
        self.history = []
    
    def generate(self, prompt: str, system_prompt=None) -> str:
        """Generate a mock response."""
        return f"""{{
  "overall_score": 7.0,
  "overall_reasoning": "This is a mock evaluation.",
  "primary_dimension_score": 7.0,
  "primary_dimension_reasoning": "This is a mock evaluation.",
  "character_consistency_score": 7.0,
  "character_consistency_reasoning": "This is a mock evaluation."
}}"""
    
    def generate_with_chat_history(self, messages, system_prompt=None):
        """Generate a mock response with chat history."""
        return "This is a mock response with chat history."
    
    def get_history(self):
        """Get the chat history."""
        return self.history
    
    def clear_history(self):
        """Clear the chat history."""
        self.history = []

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
    
    if any(keyword in question_lower for keyword in ["who are you", "tell me about yourself", "what's your name", "introduce yourself"]):
        return "identity"
    
    if any(keyword in question_lower for keyword in ["hexcore", "hextech", "technology", "research", "work", "scientific"]):
        return "technical"
    
    if any(keyword in question_lower for keyword in ["jayce", "heimerdinger", "sky", "relationship", "friend", "colleague"]):
        return "relationship"
    
    if any(keyword in question_lower for keyword in ["evolution", "glorious", "future", "humanity", "progress", "philosophy", "believe", "think about"]):
        return "philosophical"
    
    # Default to identity if we can't determine the type
    return "identity"

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
        
        Primary Dimension: Self-perception and identity
        - How well does the response reflect Viktor's view of himself?
        - Does it capture his core values and motivations?
        - Does it reflect his background and experiences that shaped him?
        
        Character Consistency:
        - Does the response use Viktor's typical speech patterns and technical language?
        - Does it maintain his stoic emotional tone?
        - Is the response consistent with Viktor's known character traits?
        """
    
    elif question_type == "technical":
        return """
        For this technical question, focus on:
        - Accuracy and depth of technical details about Hextech/Hexcore
        - Use of precise scientific terminology and concepts
        - Logical and methodical explanation of technical concepts
        - Whether the response demonstrates deep understanding of the technology
        
        Primary Dimension: Technical knowledge and precision
        - How accurately does the response describe the technical concepts?
        - Does it use appropriate scientific terminology?
        - Does it demonstrate the depth of understanding Viktor would have?
        
        Character Consistency:
        - Does the response maintain Viktor's methodical approach to technical topics?
        - Does it show appropriate enthusiasm for technological advancement?
        - Does it reflect Viktor's values regarding the purpose of technology?
        """
    
    elif question_type == "relationship":
        return """
        For this relationship question, focus on:
        - How well the response captures Viktor's professional and somewhat detached approach to relationships
        - Whether it emphasizes pragmatic collaboration over emotional connection
        - If it maintains Viktor's focus on work and progress even when discussing others
        - Whether it accurately reflects Viktor's known relationships from the show
        
        Primary Dimension: Approach to relationships
        - Does the response capture Viktor's pragmatic view of relationships?
        - Does it accurately reflect his known relationships with other characters?
        - Does it maintain his focus on work even when discussing personal connections?
        
        Character Consistency:
        - Does the response maintain Viktor's emotional restraint?
        - Does it use his typical speech patterns when discussing others?
        - Is the level of detail and personal disclosure appropriate for Viktor?
        """
    
    elif question_type == "philosophical":
        return """
        For this philosophical question, focus on:
        - How well the response captures Viktor's worldview and values
        - Whether it emphasizes progress, evolution, and transcending human limitations
        - If it frames philosophical concepts in technical, practical terms rather than abstract ones
        - Whether it maintains Viktor's pragmatic approach even to philosophical questions
        
        Primary Dimension: Worldview and values
        - Does the response accurately reflect Viktor's philosophical perspective?
        - Does it emphasize his core values of progress and evolution?
        - Does it frame abstract concepts in practical, technical terms?
        
        Character Consistency:
        - Does the response maintain Viktor's pragmatic approach to philosophical topics?
        - Does it use his typical speech patterns and technical framing?
        - Does it show appropriate passion for his vision while maintaining his stoic demeanor?
        """
    
    # Default criteria if we can't determine the type
    return """
    Focus on how well the response captures Viktor's character overall, including:
    - His identity as a scientist from Zaun
    - His technical knowledge and approach
    - His pragmatic, determined personality
    - His stoic emotional expression
    
    Primary Dimension: Overall character portrayal
    - How well does the response capture Viktor's essential character?
    - Does it reflect his core traits and values?
    
    Character Consistency:
    - Does the response use Viktor's typical speech patterns?
    - Is the emotional tone consistent with his character?
    - Does the response avoid contradicting established facts about Viktor?
    """

def evaluate_response(response, question, evaluator_llm):
    """
    Evaluate a response using the evaluator LLM.
    
    Args:
        response: The response to evaluate
        question: The question that was asked
        evaluator_llm: The LLM to use for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Determine question type
    question_type = get_question_type(question)
    
    # Get specific evaluation criteria
    specific_criteria = get_evaluation_criteria(question_type)
    
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
2. Primary Dimension Score: How well the response addresses the primary dimension for this question type
   - For identity questions: How well it captures Viktor's self-perception and identity
   - For technical questions: How well it demonstrates technical knowledge and precision
   - For relationship questions: How well it reflects Viktor's approach to relationships
   - For philosophical questions: How well it conveys Viktor's worldview and values
3. Character Consistency Score: How consistent the response is with Viktor's character traits, speech patterns, and mannerisms

For each score, provide a brief explanation of your reasoning. Be CRITICAL and SPECIFIC about what works and what doesn't.

Your response should be in this format:
Overall Score: [1-10]
Overall Reasoning: [Your detailed reasoning]
Primary Dimension Score: [1-10]
Primary Dimension Reasoning: [Your detailed reasoning]
Character Consistency Score: [1-10]
Character Consistency Reasoning: [Your detailed reasoning]
"""

    try:
        # Get evaluation from the LLM
        evaluation_response = evaluator_llm.generate(evaluation_prompt)
        
        # Parse the response
        try:
            # Extract scores and reasoning using regex
            overall_score = re.search(r'Overall Score:\s*(\d+)', evaluation_response)
            overall_reasoning = re.search(r'Overall Reasoning:\s*(.*?)(?:Primary Dimension Score:|$)', evaluation_response, re.DOTALL)
            
            primary_score = re.search(r'Primary Dimension Score:\s*(\d+)', evaluation_response)
            primary_reasoning = re.search(r'Primary Dimension Reasoning:\s*(.*?)(?:Character Consistency Score:|$)', evaluation_response, re.DOTALL)
            
            consistency_score = re.search(r'Character Consistency Score:\s*(\d+)', evaluation_response)
            consistency_reasoning = re.search(r'Character Consistency Reasoning:\s*(.*?)(?:$)', evaluation_response, re.DOTALL)
            
            # Create metrics dictionary
            metrics = {
                "overall_score": int(overall_score.group(1)) if overall_score else 5,
                "overall_reasoning": overall_reasoning.group(1).strip() if overall_reasoning else "No reasoning provided",
                "primary_dimension_score": int(primary_score.group(1)) if primary_score else 5,
                "primary_dimension_reasoning": primary_reasoning.group(1).strip() if primary_reasoning else "No reasoning provided",
                "character_consistency_score": int(consistency_score.group(1)) if consistency_score else 5,
                "character_consistency_reasoning": consistency_reasoning.group(1).strip() if consistency_reasoning else "No reasoning provided"
            }
            
            # Add question type to metrics
            metrics["question_type"] = question_type
            
            return metrics
            
        except Exception as e:
            print(f"Error parsing evaluation response: {e}")
            print(f"Raw response: {evaluation_response}")
            # Return default scores with error message
            return {
                "overall_score": 5.0,
                "overall_reasoning": f"Error parsing response: {str(e)}",
                "primary_dimension_score": 5.0,
                "primary_dimension_reasoning": "Error parsing response",
                "character_consistency_score": 5.0,
                "character_consistency_reasoning": "Error parsing response",
                "question_type": question_type
            }
    except Exception as e:
        print(f"Error getting evaluation from LLM: {e}")
        # Return default scores with error message
        return {
            "overall_score": 5.0,
            "overall_reasoning": f"Error getting evaluation: {str(e)}",
            "primary_dimension_score": 5.0,
            "primary_dimension_reasoning": "Error getting evaluation",
            "character_consistency_score": 5.0,
            "character_consistency_reasoning": "Error getting evaluation",
            "question_type": question_type
        }

def calculate_weighted_score(metrics):
    """
    Calculate a weighted overall score based on the question type.
    
    Args:
        metrics: Dictionary containing evaluation metrics and question type
        
    Returns:
        Float representing the weighted overall score
    """
    question_type = metrics.get("question_type", "identity")
    
    # For all question types, we now use a simpler weighting:
    # 60% primary dimension, 40% character consistency
    primary_dimension_weight = 0.6
    character_consistency_weight = 0.4
    
    return (
        metrics["primary_dimension_score"] * primary_dimension_weight +
        metrics["character_consistency_score"] * character_consistency_weight
    )

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
    output = f"""
Question: {question}
Question Type: {metrics.get('question_type', 'unknown')}

Response:
{response}

Evaluation:
- Overall Score: {metrics.get('overall_score', 'N/A')}/10
  - {metrics.get('overall_reasoning', 'No reasoning provided')}
- Primary Dimension Score: {metrics.get('primary_dimension_score', 'N/A')}/10
  - {metrics.get('primary_dimension_reasoning', 'No reasoning provided')}
- Character Consistency Score: {metrics.get('character_consistency_score', 'N/A')}/10
  - {metrics.get('character_consistency_reasoning', 'No reasoning provided')}
"""
    
    if weighted_score is not None:
        output += f"\nWeighted Score (based on question type): {weighted_score:.2f}/10\n"
    
    return output

def parse_arguments():
    """Parse command-line arguments for the evaluator testing script."""
    parser = argparse.ArgumentParser(description="Test the ViktorAI evaluator with sample responses")
    
    parser.add_argument("--evaluator-model", type=str, default="llama3",
                        help="Name of the model to use for evaluation (default: llama3)")
    
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock implementations instead of real LLM (for testing)")
    
    parser.add_argument("--output-dir", type=str, default="evaluator_test_results",
                        help="Directory to save results (default: evaluator_test_results)")
    
    parser.add_argument("--question-type", type=str, choices=["identity", "technical", "relationship", "philosophical", "all"],
                        default="all", help="Type of questions to test (default: all)")
    
    parser.add_argument("--custom-question", type=str,
                        help="Custom question to evaluate (requires --custom-response)")
    
    parser.add_argument("--custom-response", type=str,
                        help="Custom response to evaluate (requires --custom-question)")
    
    parser.add_argument("--html-report", action="store_true",
                        help="Generate an HTML report in addition to the Markdown report")
    
    return parser.parse_args()

def create_html_report(results, output_path):
    """
    Create an HTML report from the evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the HTML report
    """
    # Start building HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViktorAI Evaluator Test Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3, h4, h5 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 1px solid #3498db;
            padding-bottom: 5px;
        }
        .metadata {
            background-color: #eaf2f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metadata ul {
            list-style-type: none;
            padding-left: 0;
        }
        .metadata li {
            margin-bottom: 5px;
        }
        .summary {
            background-color: #eaf2f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .response {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 4px solid #3498db;
            white-space: pre-wrap;
        }
        .evaluation {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #e74c3c;
        }
        .score {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .reasoning {
            font-style: italic;
            color: #7f8c8d;
            margin-bottom: 15px;
            padding-left: 15px;
            border-left: 2px solid #bdc3c7;
        }
        .divider {
            height: 1px;
            background-color: #ddd;
            margin: 30px 0;
        }
    </style>
</head>
<body>
    <h1>ViktorAI Evaluator Test Results</h1>
    
    <div class="metadata">
        <ul>
            <li><strong>Date:</strong> {results['date']}</li>
            <li><strong>Evaluator Model:</strong> {results['evaluator_model']}</li>
        </ul>
    </div>
    
    <h2>Summary</h2>
    <div class="summary">
        <p>This report contains evaluation results for {results['total_responses']} responses across {results['question_types']} question types.</p>
    </div>
"""

    # Add results for each question type
    for q_type in results['evaluations']:
        html += f"""
    <h2>{q_type.capitalize()} Questions</h2>
    <p>Question: {results['evaluations'][q_type]['question']}</p>
"""
        
        for i, eval_data in enumerate(results['evaluations'][q_type]['responses']):
            response = eval_data['response']
            metrics = eval_data['metrics']
            weighted_score = eval_data['weighted_score']
            
            expected_quality = 'Good' if i == 0 else 'Medium' if i == 1 else 'Poor'
            
            html += f"""
    <h3>Response {i+1} (Expected quality: {expected_quality})</h3>
    
    <div class="response">
        <h4>Response:</h4>
        <p>{response.replace('\n', '<br>')}</p>
    </div>
    
    <div class="evaluation">
        <h4>Evaluation:</h4>
        <p class="score">Overall Score: {metrics.get('overall_score', 'N/A')}/10</p>
"""
            
            if 'overall_reasoning' in metrics:
                html += f'<p class="reasoning">{metrics["overall_reasoning"]}</p>'
            
            html += f'<p class="score">Weighted Score: {weighted_score:.2f}/10 (based on question type)</p>'
            
            html += f'<p class="score">Primary Dimension Score: {metrics.get("primary_dimension_score", "N/A")}/10</p>'
            if 'primary_dimension_reasoning' in metrics:
                html += f'<p class="reasoning">{metrics["primary_dimension_reasoning"]}</p>'
            
            html += f'<p class="score">Character Consistency Score: {metrics.get("character_consistency_score", "N/A")}/10</p>'
            if 'character_consistency_reasoning' in metrics:
                html += f'<p class="reasoning">{metrics["character_consistency_reasoning"]}</p>'
            
            html += """
    </div>
    <div class="divider"></div>
"""
    
    # Close HTML
    html += """
</body>
</html>
"""
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

def main():
    """Main function to run the evaluator test."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize evaluator LLM
    config = Config(
        model_name=args.evaluator_model,
        temperature=0.2,  # Lower temperature for more consistent evaluations
        max_tokens=1000
    )
    
    if args.use_mock:
        print("Using mock implementation for OllamaInterface")
        evaluator_llm = MockOllamaInterface(config)
    else:
        try:
            from src.llm_interface import OllamaInterface
            evaluator_llm = OllamaInterface(config)
        except Exception as e:
            print(f"Error initializing OllamaInterface: {e}")
            print("Falling back to mock implementation")
            evaluator_llm = MockOllamaInterface(config)
    
    # Timestamp for output files
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results dictionary for HTML report
    html_results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'evaluator_model': args.evaluator_model,
        'total_responses': 0,
        'question_types': 0,
        'evaluations': {}
    }
    
    # Open output file
    output_file = output_dir / f"evaluator_test_{timestamp}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# ViktorAI Evaluator Test Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Evaluator Model:** {args.evaluator_model}\n\n")
        
        # Handle custom question/response if provided
        if args.custom_question and args.custom_response:
            print(f"Evaluating custom question: {args.custom_question}")
            metrics = evaluate_response(args.custom_response, args.custom_question, evaluator_llm)
            weighted_score = calculate_weighted_score(metrics)
            
            output = format_evaluation_output(metrics, args.custom_question, args.custom_response, weighted_score)
            print(output)
            
            f.write("## Custom Question Evaluation\n\n")
            f.write(output)
            f.write("\n---\n\n")
            
            # Add to HTML results
            question_type = metrics.get('question_type', 'custom')
            if question_type not in html_results['evaluations']:
                html_results['evaluations'][question_type] = {
                    'question': args.custom_question,
                    'responses': []
                }
            
            html_results['evaluations'][question_type]['responses'].append({
                'response': args.custom_response,
                'metrics': metrics,
                'weighted_score': weighted_score
            })
            
            html_results['total_responses'] += 1
            html_results['question_types'] = 1
        
        # Otherwise, evaluate sample responses
        else:
            question_types = ["identity", "technical", "relationship", "philosophical"] if args.question_type == "all" else [args.question_type]
            html_results['question_types'] = len(question_types)
            
            for q_type in question_types:
                if q_type in SAMPLE_RESPONSES:
                    print(f"Evaluating {q_type} questions...")
                    f.write(f"## {q_type.capitalize()} Questions\n\n")
                    
                    question = SAMPLE_RESPONSES[q_type]["question"]
                    
                    # Initialize HTML results for this question type
                    html_results['evaluations'][q_type] = {
                        'question': question,
                        'responses': []
                    }
                    
                    for i, response in enumerate(SAMPLE_RESPONSES[q_type]["responses"]):
                        print(f"  Evaluating response {i+1}...")
                        metrics = evaluate_response(response, question, evaluator_llm)
                        weighted_score = calculate_weighted_score(metrics)
                        
                        output = format_evaluation_output(metrics, question, response, weighted_score)
                        
                        f.write(f"### Response {i+1} (Expected quality: {'Good' if i == 0 else 'Medium' if i == 1 else 'Poor'})\n\n")
                        f.write(output)
                        f.write("\n---\n\n")
                        
                        # Add to HTML results
                        html_results['evaluations'][q_type]['responses'].append({
                            'response': response,
                            'metrics': metrics,
                            'weighted_score': weighted_score
                        })
                        
                        html_results['total_responses'] += 1
    
    print(f"\nEvaluation results saved to {output_file}")
    
    # Generate HTML report if requested
    if args.html_report:
        html_output_file = output_dir / f"evaluator_test_{timestamp}.html"
        create_html_report(html_results, html_output_file)
        print(f"HTML report saved to {html_output_file}")

if __name__ == "__main__":
    main() 
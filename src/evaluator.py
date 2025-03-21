#!/usr/bin/env python3
"""
Evaluator module for ViktorAI.

This module provides evaluation functionality for assessing how well
a response captures the Viktor character from Arcane.
"""

import re
from typing import Dict, Any, Optional, Union


class Evaluator:
    """
    Evaluator class for assessing Viktor character responses.
    
    This class encapsulates the logic for evaluating responses to questions
    about the Viktor character from Arcane. It determines the type of question,
    constructs an appropriate evaluation prompt, and processes the evaluation
    results.
    """
    
    def __init__(self, llm_interface):
        """
        Initialize the Evaluator with an LLM interface.
        
        Args:
            llm_interface: An interface to an LLM for generating evaluations
        """
        self.llm_interface = llm_interface
    
    def get_question_type(self, question: str, headings_map: Optional[Dict[str, str]] = None) -> str:
        """
        Determine the type of question based on its content.
        
        Args:
            question: The question to categorize
            headings_map: Optional dictionary mapping questions to their types from file sections
            
        Returns:
            String indicating the question type (identity, technical, relationship, philosophical)
        """
        # First check if we have a pre-determined type from the headings map
        if headings_map and question in headings_map:
            return headings_map[question]
        
        # If not, fallback to the keyword-based approach
        question_lower = question.lower()
        
        # Identity questions
        if any(keyword in question_lower for keyword in [
            "who are you", "tell me about yourself", "what's your name", 
            "introduce yourself", "what's your identity", "who is viktor"
        ]):
            return "identity"
        
        # Technical questions
        if any(keyword in question_lower for keyword in [
            "hexcore", "hextech", "technology", "research", "work", "scientific", 
            "limitations", "improve", "applications", "experiment", "invention"
        ]):
            return "technical"
        
        # Relationship questions
        if any(keyword in question_lower for keyword in [
            "jayce", "heimerdinger", "sky", "relationship", "friend", "colleague", 
            "thoughts on", "council", "caitlyn", "silco", "academy"
        ]):
            return "relationship"
        
        # Philosophical questions
        if any(keyword in question_lower for keyword in [
            "evolution", "glorious", "future", "humanity", "progress", "philosophy", 
            "believe", "think about", "purpose", "divide", "piltover and zaun", 
            "change one decision", "meaning", "vision", "goal"
        ]):
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
    
    def get_evaluation_criteria(self, question_type: str) -> str:
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
    
    def calculate_weighted_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate a weighted overall score based on the question type.
        
        Args:
            metrics: Dictionary containing evaluation metrics and question type
            
        Returns:
            Float representing the weighted overall score
        """
        question_type = metrics.get("question_type", "identity")
        
        # We use a standardized weighting approach for all question types:
        # - 60% weight on the primary dimension score (which is tailored to the question type)
        # - 40% weight on character consistency (which is common across all question types)
        # This provides a balanced evaluation that emphasizes the question-specific criteria
        # while still valuing overall character consistency
        
        primary_dimension_weight = 0.6
        character_consistency_weight = 0.4
        
        # Calculate the weighted score
        primary_score = metrics.get("primary_dimension_score", 5.0)
        consistency_score = metrics.get("character_consistency_score", 5.0)
        
        weighted_score = (
            primary_score * primary_dimension_weight +
            consistency_score * character_consistency_weight
        )
        
        return weighted_score
    
    def evaluate_response(
        self, 
        response: str, 
        question: str, 
        question_type: Optional[str] = None, 
        headings_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using the evaluator LLM.
        
        Args:
            response: The response to evaluate
            question: The question that was asked
            question_type: Optional, the type of question (identity, technical, relationship, philosophical)
            headings_map: Optional dictionary mapping questions to their types from file sections
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Determine question type if not provided
        if question_type is None:
            question_type = self.get_question_type(question, headings_map)
        
        # Get evaluation criteria for this question type
        criteria = self.get_evaluation_criteria(question_type)
        
        # Construct the evaluation prompt
        evaluation_prompt = f"""You are an expert evaluator assessing how well a response captures the character of Viktor from the Netflix show 'Arcane'. 

### Character Profile: Viktor
Viktor is a brilliant scientist from Zaun who works with Jayce in Piltover on Hextech technology. He is characterized by:
- Precise, technical language and methodical thinking
- Direct, concise communication with minimal emotional expression
- Focus on scientific progress and overcoming human limitations through technology
- A pragmatic, solution-oriented mindset
- Stoic demeanor with occasional dry wit
- Deep motivation to help others through technological advancement
- Preference for brevity and efficiency in communication

IMPORTANT: Viktor typically speaks with concision and precision. Verbose, flowery language is NOT characteristic of him. Responses should be brief, direct, and focused.

### Question
{question}

### Response to Evaluate
{response}

### Evaluation Task
You will evaluate this response on how well it captures Viktor's character, focusing particularly on the following dimensions:

{criteria}

In your evaluation, pay special attention to:
1. Use of language: Does it match Viktor's precise, technical, and concise manner of speaking?
2. Content accuracy: Does it align with Viktor's known perspectives and priorities?
3. Emotional tone: Does it maintain Viktor's characteristic restraint and focus on pragmatic concerns?
4. BREVITY: Viktor values efficiency in communication. Overly verbose responses should be scored lower unless the verbosity serves a specific purpose aligned with his character.

IMPORTANT: Use the FULL RANGE of scores from 1-10. Do not default to middle scores (5/10) out of uncertainty.
- If a response is poor, score it between 1-3
- If a response is below average, score it between 4-5
- If a response is average, score it between 6-7
- If a response is good, score it between 8-9
- If a response is excellent, score it 10

CRITICAL REQUIREMENT: You MUST provide detailed reasoning for EACH score. Explain specifically what works and what doesn't in the response. Your reasoning should reference specific aspects of Viktor's character and specific elements of the response being evaluated.

Format your evaluation as follows:
```
Overall Score: [1-10]
Overall Reasoning: [Your reasoning for the overall score]

Primary Dimension Score: [1-10]
Primary Dimension Reasoning: [Your reasoning for the primary dimension score]

Character Consistency Score: [1-10]
Character Consistency Reasoning: [Your reasoning for the character consistency score]
```

REMEMBER: Be critical and use the full range of scores. Excellent responses should be concise, focused, and authentically capture Viktor's voice. Verbose responses that don't reflect Viktor's efficient communication style should receive lower scores, even if the content is technically accurate.
"""
        
        try:
            # Send prompt to evaluator
            evaluation_response = self.llm_interface.generate(evaluation_prompt)
            
            # Parse the evaluation response
            metrics = {}
            
            # Extract overall score - improved regex to handle more formats
            overall_match = re.search(r'Overall Score:\s*(\d+(?:\.\d+)?|[0-9]+)', evaluation_response, re.IGNORECASE)
            if overall_match:
                try:
                    metrics["overall_score"] = float(overall_match.group(1))
                except ValueError:
                    print(f"Warning: Could not convert overall score '{overall_match.group(1)}' to float")
                    metrics["overall_score"] = 5.0  # Default fallback value
            else:
                print("Warning: Could not extract overall score from evaluation response")
                metrics["overall_score"] = 5.0  # Default fallback value
            
            # Extract overall reasoning - improved regex to handle more formats
            overall_reasoning_match = re.search(r'Overall Reasoning:?\s*(.*?)(?=\n\n|\n[A-Z]|Primary Dimension Score|$)', 
                                                evaluation_response, re.DOTALL | re.IGNORECASE)
            if overall_reasoning_match:
                metrics["overall_reasoning"] = overall_reasoning_match.group(1).strip()
            else:
                metrics["overall_reasoning"] = "No detailed reasoning was provided by the evaluator for this score. This may indicate that the evaluation was not complete or the response format was unexpected."
            
            # Extract primary dimension score - improved regex to handle more formats
            primary_match = re.search(r'Primary Dimension Score:?\s*(\d+(?:\.\d+)?|[0-9]+)', 
                                      evaluation_response, re.IGNORECASE)
            if primary_match:
                try:
                    metrics["primary_dimension_score"] = float(primary_match.group(1))
                except ValueError:
                    print(f"Warning: Could not convert primary dimension score '{primary_match.group(1)}' to float")
                    metrics["primary_dimension_score"] = 5.0  # Default fallback value
            else:
                print("Warning: Could not extract primary dimension score from evaluation response")
                metrics["primary_dimension_score"] = 5.0  # Default fallback value
            
            # Extract primary dimension reasoning - improved regex to handle more formats
            primary_reasoning_match = re.search(r'Primary Dimension Reasoning:?\s*(.*?)(?=\n\n|\n[A-Z]|Character Consistency Score|$)', 
                                                evaluation_response, re.DOTALL | re.IGNORECASE)
            if primary_reasoning_match:
                metrics["primary_dimension_reasoning"] = primary_reasoning_match.group(1).strip()
            else:
                metrics["primary_dimension_reasoning"] = "No detailed reasoning was provided by the evaluator for the primary dimension score. This may indicate an evaluation issue with this response."
            
            # Extract character consistency score - improved regex to handle more formats
            consistency_match = re.search(r'Character Consistency Score:?\s*(\d+(?:\.\d+)?|[0-9]+)', 
                                          evaluation_response, re.IGNORECASE)
            if consistency_match:
                try:
                    metrics["character_consistency_score"] = float(consistency_match.group(1))
                except ValueError:
                    print(f"Warning: Could not convert character consistency score '{consistency_match.group(1)}' to float")
                    metrics["character_consistency_score"] = 5.0  # Default fallback value
            else:
                print("Warning: Could not extract character consistency score from evaluation response")
                metrics["character_consistency_score"] = 5.0  # Default fallback value
            
            # Extract character consistency reasoning - improved regex to handle more formats
            consistency_reasoning_match = re.search(r'Character Consistency Reasoning:?\s*(.*?)(?=\n\n|\n[A-Z]|$)', 
                                                    evaluation_response, re.DOTALL | re.IGNORECASE)
            if consistency_reasoning_match:
                metrics["character_consistency_reasoning"] = consistency_reasoning_match.group(1).strip()
            else:
                metrics["character_consistency_reasoning"] = "No detailed reasoning was provided by the evaluator for the character consistency score. This suggests an issue with the evaluation format."
            
            # Add question type to metrics
            metrics["question_type"] = question_type
            
            # Clean up the reasoning text by removing any ** markers or other formatting
            for key in metrics:
                if isinstance(metrics[key], str):
                    # Remove ** markers
                    metrics[key] = re.sub(r'\*\*\s*', '', metrics[key])
                    metrics[key] = re.sub(r'\s*\*\*', '', metrics[key])
                    # Remove markdown backticks
                    metrics[key] = re.sub(r'```', '', metrics[key])
                    # Trim any extra whitespace
                    metrics[key] = metrics[key].strip()
            
            # Additional validation for scores
            for score_key in ["overall_score", "primary_dimension_score", "character_consistency_score"]:
                if score_key in metrics:
                    # Ensure score is within valid range
                    if metrics[score_key] < 0:
                        metrics[score_key] = 0.0
                    elif metrics[score_key] > 10:
                        metrics[score_key] = 10.0
            
            return metrics
        
        except Exception as e:
            print(f"Error getting evaluation from LLM: {e}")
            # Return fallback values instead of N/A
            return {
                "overall_score": 5.0,  # Default middle score
                "overall_reasoning": f"The evaluation process encountered an error: {str(e)}. This is a fallback score provided when the evaluation couldn't be completed properly.",
                "primary_dimension_score": 5.0,  # Default middle score
                "primary_dimension_reasoning": "This is a default fallback score. The primary dimension evaluation couldn't be completed due to an error in the evaluation process.",
                "character_consistency_score": 5.0,  # Default middle score
                "character_consistency_reasoning": "This is a default fallback score. The character consistency evaluation couldn't be completed due to an error in the evaluation process.",
                "question_type": question_type
            }
    
    def format_evaluation_output(
        self, 
        metrics: Dict[str, Any], 
        question: str, 
        response: str, 
        weighted_score: Optional[float] = None
    ) -> str:
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
    **Question:** {question}
    **Question Type:** {metrics.get('question_type', 'unknown')}

    **Response:**
    ```
    {response}
    ```

    **Evaluation:**

    **Overall Score:** {metrics.get('overall_score', 'N/A')}/10
    {metrics.get('overall_reasoning', 'No reasoning provided')}

    **Primary Dimension Score:** {metrics.get('primary_dimension_score', 'N/A')}/10
    {metrics.get('primary_dimension_reasoning', 'No reasoning provided')}

    **Character Consistency Score:** {metrics.get('character_consistency_score', 'N/A')}/10
    {metrics.get('character_consistency_reasoning', 'No reasoning provided')}
    """
        
        if weighted_score is not None:
            output += f"\n**Weighted Score (based on question type):** {weighted_score:.2f}/10\n"
        
        return output 
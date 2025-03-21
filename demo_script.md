# ViktorAI Demo Script

## 1. Introduction (1-2 minutes)
- Brief overview of ViktorAI project and goals
- Explain Viktor's character from Arcane (brilliant scientist, focused on technological progress, stoic personality)
- Introduce RAG system and how it helps maintain character consistency

## 2. Demonstration of ViktorAI (3-4 minutes)
- Run ViktorAI with gemma3:1b model:
  ```
  python main.py --model gemma3:1b --temperature 0.7 --max_tokens 500
  ```
- Show a conversation with Viktor covering different question types:
  - Identity question: "Who are you?"
  - Technical question: "Tell me about your work with the Hexcore."
  - Relationship question: "How would you describe your relationship with Jayce?"
  - Philosophical question: "What does 'the glorious evolution' mean to you?"

## 3. Benchmark System Explanation (2 minutes)
- Explain the need for objective evaluation of character accuracy
- Walk through the benchmark architecture:
  - Question categorization system
  - Category-specific prompting
  - Evaluation using LLM (llama3)
  - Weighted scoring system
  - Report generation

## 4. Benchmark Results (2-3 minutes)
- Show benchmark results for gemma3:1b:
  ```
  cd benchmark_results/gemma3/gemma3:1b
  ```
- Highlight key performance metrics:
  - Overall scores by question type
  - Primary dimension and character consistency scores
  - Sample evaluations for different question types
- Compare with previous benchmark results (if available)

## 5. Conclusion and Next Steps (1 minute)
- Summarize key achievements:
  - Implemented sophisticated RAG system for character consistency
  - Created robust benchmark system for objective evaluation
  - Identified optimal models for character implementation
- Future improvements:
  - Fine-tuning models for better performance
  - Improving the benchmark with more detailed questions
  - Adding voice interaction capabilities
  - Developing a web interface

## Demo Tips
- Have backup plans for any technical issues
- Prepare example questions in a text file for easy copy-paste
- Have benchmark results already generated in case of time constraints
- Keep focused on the most important features given the short time frame 
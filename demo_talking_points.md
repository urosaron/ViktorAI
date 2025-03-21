# Benchmark System Talking Points

## Key Features

### Question Categorization
- System automatically categorizes questions into four types:
  - **Identity**: Questions about Viktor's self-perception, motivations, and background
  - **Technical**: Questions about technical concepts, Hextech, and scientific details
  - **Relationship**: Questions about Viktor's relationships with others
  - **Philosophical**: Questions about values, worldview, and abstract concepts
- Implementation uses keyword matching and specific question handling
- Questions are mapped to appropriate prompt categories for optimal responses

### Category-Specific Prompting
- Each question type gets a specialized prompt:
  - **Identity** → personality_focused prompt
  - **Technical** → technical_focused prompt
  - **Relationship** → relationship_focused prompt
  - **Philosophical** → full prompt
- This ensures responses are tailored to the specific context of the question
- More efficient than testing every question with every prompt category

### Evaluation System
- Uses a separate LLM (llama3) to evaluate responses
- Evaluation criteria tailored to question type:
  - **Identity**: Focus on self-perception and identity accuracy
  - **Technical**: Focus on technical knowledge and precision
  - **Relationship**: Focus on approach to relationships
  - **Philosophical**: Focus on worldview and values
- Provides scores and detailed reasoning for each evaluation

### Scoring and Metrics
- **Overall Score**: General quality of the response (1-10)
- **Primary Dimension Score**: Performance on the main aspect of the question type (1-10)
- **Character Consistency**: How well the response maintains Viktor's character (1-10)
- **Weighted Score**: Combines scores based on question type importance
- Aggregated metrics by category and question type

### Directory Structure
- Organized hierarchical storage of benchmark results:
  - Model family folder (e.g., gemma3)
  - Model version folder (e.g., gemma3:1b)
  - Timestamped run folders with raw data and visualizations
- Latest reference files for quick access to most recent results

## Key Results

- gemma3:1b performs well across all question types
- Philosophical questions score highest (8.17/10)
- Technical questions also score high (8.25/10)
- Relationship questions slightly lower (7.50/10)
- Identity questions solid (7.67/10)
- Average overall score: 7.86/10

## Value Delivered

- Objective, quantifiable measurement of character accuracy
- Clear identification of strengths and weaknesses
- Ability to compare different models systematically
- Framework for continuous improvement of responses
- Organized system for maintaining benchmark history 
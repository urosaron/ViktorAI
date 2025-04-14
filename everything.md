# ViktorAI Project Documentation

This document provides a comprehensive explanation of the ViktorAI project, including its structure, components, and how they work together.

## Project Overview

ViktorAI is a character AI chatbot that embodies Viktor from Arcane Season 1. The project uses Retrieval-Augmented Generation (RAG) to provide accurate, in-character responses that are grounded in the show's content. The chatbot interfaces with Ollama LLMs to generate responses in Viktor's voice while maintaining character consistency.

## Core Components

### 1. Character Data Files (`character_data/`)

These files contain Viktor's character information, organized into different aspects:

- **viktor_core_profile.md**: Core personality traits, background, and motivations
- **viktor_technical_knowledge.md**: Viktor's scientific and technical knowledge
- **viktor_relationships.md**: Viktor's relationships with other characters
- **viktor_world_context.md**: Information about the world of Arcane
- **viktor_response_guidelines.md**: Guidelines for generating in-character responses
- **viktor_test_scenarios.md**: Test cases for evaluating the chatbot
- **viktor_main_prompt.md**: The main system prompt for the LLM
- **viktor_scenes_and_events.md**: Detailed scene-by-scene analysis of Viktor's actions and events from Arcane Season 1

### 2. Source Code (`src/`)

#### Main Classes

- **`config.py`**: Configuration settings for the chatbot
  - Stores model settings (model name, temperature, max tokens)
  - Manages file paths for character data
  - Provides utility methods for accessing character files

- **`character_data_loader.py`**: Loads and processes character data
  - Reads markdown files from the character_data directory
  - Combines character data into a structured format
  - Provides search functionality for character analysis

- **`llm_interface.py`**: Interfaces with the Ollama LLM
  - Sends prompts to the Ollama API
  - Handles conversation history
  - Manages response generation

- **`viktor_ai.py`**: Main chatbot implementation
  - Initializes components (data loader, LLM interface, vector store)
  - Prepares system prompts
  - Generates responses using RAG
  - Maintains backward compatibility with previous methods

- **`evaluator.py`**: LLM-based evaluation system
  - Categorizes questions by type (identity, technical, relationship, philosophical)
  - Constructs specialized evaluation prompts based on question type
  - Evaluates responses for character consistency and other metrics
  - Used primarily for benchmarking and testing

- **`response_classifier.py`**: PyTorch-based response quality classifier
  - Real-time evaluation of response quality during conversations
  - Neural network model for character accuracy and response quality assessment
  - Feature extraction from prompts and responses
  - Response regeneration for low-quality outputs

#### RAG Components

- **`vector_store.py`**: Vector database implementation
  - Provides embedding and storage for character data
  - Implements similarity search for retrieval
  - Includes both FAISS and simple vector store implementations
  - Handles saving and loading the vector store

- **`indexer.py`**: Processes and indexes character data
  - Splits text into chunks for better retrieval
  - Extracts sections from markdown files
  - Adds metadata to each document
  - Creates and saves the vector store

### 3. Entry Points

- **`main.py`**: Main entry point for the chatbot
  - Parses command-line arguments
  - Initializes the chatbot
  - Runs the conversation loop

- **`build_vector_store.py`**: Script to build the vector store
  - Processes character data
  - Creates embeddings
  - Saves the vector store for later use

- **`run_benchmark.py`**: Script to run benchmarks
  - Evaluates chatbot performance across different question types
  - Generates reports with metrics and scores
  - Organizes results by model and timestamp

### 4. Testing and Documentation

- **`tests/test_viktor_ai.py`**: Tests for the chatbot
  - Tests character data loading
  - Tests combined character data generation
  - Tests character analysis search

- **`tests/test_evaluator.py`**: Tests for the evaluator
  - Tests question categorization
  - Tests evaluation criteria selection
  - Tests response evaluation and scoring

- **`tests/test_response_classifier.py`**: Tests for the classifier
  - Tests feature extraction
  - Tests model initialization and prediction
  - Tests response quality assessment

- **`README.md`**: Project documentation
  - Setup instructions
  - Usage examples
  - Project structure
  - RAG system explanation

- **`logbook.md`**: Development history and changes
  - Tracks major changes to the project
  - Documents motivations for changes
  - Outlines future plans

- **`everything.md`**: Comprehensive project documentation (this file)
  - Detailed explanation of all components
  - How the system works
  - Data flow and architecture

## How the RAG System Works

The RAG system is the core innovation in this project and works as follows:

1. **Indexing Phase** (done once during setup):
   - Character data is processed into chunks with metadata
   - Each chunk is embedded using sentence-transformers
   - Embeddings are stored in a vector database (FAISS or simple vector store)

2. **Retrieval Phase** (for each user query):
   - The user's query is embedded using the same model
   - Similar documents are retrieved from the vector store
   - Retrieved documents are formatted with their metadata

3. **Generation Phase**:
   - A prompt is created that includes the retrieved context
   - The prompt is sent to the LLM with the system prompt
   - The LLM generates a response based on both the retrieved context and character knowledge

## Data Flow

1. User enters a query in `main.py`
2. Query is passed to `viktor_ai.py`'s `generate_response` method
3. `_retrieve_context` method embeds the query and searches the vector store
4. Retrieved context is formatted and added to a prompt template
5. The prompt is sent to the LLM via `llm_interface.py`
6. The LLM generates a response that is returned to the user

## Response Quality Classifier

One of the key enhancements to ViktorAI is the addition of a real-time response quality classifier built with PyTorch. Unlike the LLM-based evaluation system used for benchmarking, this classifier works during live conversations to ensure high-quality, in-character responses.

### Architecture and Components

- **ResponseClassifier**: Main class that interfaces with the PyTorch model
  - Initializes and loads the pre-trained model
  - Manages feature extraction and prediction
  - Evaluates response quality in real-time
  - Integrated with the main conversation flow

- **ResponseQualityModel**: PyTorch neural network with dual output heads
  - Character accuracy branch: Evaluates character consistency
  - Response quality branch: Evaluates overall response coherence and relevance
  - Uses simple feed-forward architecture with ReLU activations
  - Sigmoid outputs for normalized scoring (0-1 range)

### Feature Extraction

The classifier extracts features from the prompt and response, including:
- Presence of character-specific keywords and terms (e.g., "hextech", "hexcore", "progress")
- Response and prompt length metrics
- Language patterns typical of Viktor's speech
- Term frequency for key character elements

The feature extraction method (`_prepare_features`) transforms text into numerical features that the model can process. In a production environment, this could be enhanced with more sophisticated embedding techniques.

### Integration with Response Generation

When enabled (via the `--use-classifier` flag), the response generation process follows this flow:

1. User input is received and processed through the RAG system
2. An initial response is generated by the LLM
3. The ResponseClassifier evaluates the response, producing:
   - Character accuracy score (0-1)
   - Response quality score (0-1)
   - Overall score (average of both scores)
4. If the overall score is below the minimum threshold (`min_response_score` in Config):
   - The system regenerates a new response (up to `max_retries` times)
   - Each new response is evaluated until a satisfactory one is found
   - If no response meets the threshold after maximum attempts, the best one is used
5. The final response is returned to the user

This process is transparent to the user but significantly improves response quality.

### Training Process

The classifier model can be trained or fine-tuned using:
- `scripts/generate_classifier_data.py`: Creates labeled training examples
- `scripts/train_classifier.py`: Trains the model on these examples
- Training data is stored in `models/classifier_training_data.json`

The training process uses mean squared error loss and Adam optimizer to learn weights that accurately predict both character accuracy and response quality.

### Command-Line Arguments

Several command-line arguments control the classifier's behavior:
- `--use-classifier`: Enable the response quality classifier
- `--no-classifier`: Disable the classifier even if available
- `--min-score`: Set minimum acceptable score (default: 0.6)
- `--debug`: Show detailed score information

Example with all options:
```bash
python main.py --model gemma3:1b --use-classifier --min-score 0.7 --debug
```

### Implementation Details

The pre-trained model file (`models/response_classifier.pt`) contains the trained neural network weights. The model uses PyTorch's standard modules:
- Linear layers for feature transformation
- ReLU activations for non-linearity
- Sigmoid output for score normalization

The model automatically selects the appropriate device (CUDA if available, otherwise CPU) for inference.

### Dual Evaluation Approach

ViktorAI now benefits from two complementary evaluation systems:
1. **LLM-based Evaluator** (src/evaluator.py): Used for systematic benchmarking and detailed assessment
2. **Neural Network Classifier** (src/response_classifier.py): Used for real-time filtering during conversations

This dual approach leverages the strengths of both methods: LLMs provide detailed, human-like assessment for benchmarking, while the neural network offers fast, consistent filtering during interactive use.

## Key Features

1. **Character Consistency**: By retrieving relevant character information for each query, the system maintains Viktor's consistent personality and knowledge.

2. **Factual Accuracy**: The RAG system grounds responses in actual show content, reducing hallucinations and made-up events.

3. **Graceful Degradation**: If the vector store is unavailable, the system falls back to simpler methods.

4. **Extensibility**: The modular design makes it easy to add new features or modify existing ones.

5. **Response Quality Filtering**: The response classifier ensures high-quality, in-character responses by filtering low-quality outputs.

## Setup and Usage

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally

### Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Build the vector store: `python build_vector_store.py`
5. Run the chatbot: `python main.py`

### Command-line Arguments

- `--model`: The Ollama model to use (default: llama3)
- `--temperature`: Temperature for response generation (default: 0.7)
- `--max_tokens`: Maximum tokens for response generation (default: 500)
- `--use-classifier`: Enable the response quality classifier
- `--no-classifier`: Disable the classifier even if available
- `--min-score`: Set minimum acceptable score (default: 0.6)
- `--debug`: Show detailed score information

### User Interface

When you start ViktorAI, the system displays:
- The model being used (e.g., llama3)
- The temperature setting
- The maximum tokens setting
- Whether the response classifier is enabled

This information helps users understand the current configuration and ensures they're using the intended model.

Example startup display:
```
==================================================
Welcome to ViktorAI - Viktor from Arcane Season 1
Using model: llama3
Temperature: 0.7
Max tokens: 500
Response quality classifier: Enabled (min score: 0.6)
Type 'exit', 'quit', or Ctrl+C to end the conversation.
==================================================
```

## Code Quality and Maintenance

### Code Maintenance Notes
A comprehensive code review was conducted to identify redundancies and areas for improvement in the ViktorAI codebase. The following findings were noted:

1. **Legacy Methods**: Some methods in `viktor_ai.py` are marked as legacy and used only as fallbacks when the vector store is not available. These could be refactored into a separate utility module.

2. **Vector Store Implementation**: The current implementation has two approaches (FAISS-based and simple numpy-based) with conditional logic throughout the code. This creates some code duplication that could be refactored using a cleaner inheritance pattern.

3. **Error Handling**: Error handling could be improved with more specific error types and better error messages throughout the codebase.

4. **Test Coverage**: The test suite only covers the `CharacterDataLoader` class but not other critical components. Additional tests would improve code reliability.

5. **Unused Parameters**: Some parameters are extracted but not used in certain parts of the code.

## Model Testing and Evaluation

ViktorAI includes a comprehensive model testing system that allows for systematic evaluation of different LLM models. This system helps in selecting the most appropriate model for character AI applications.

### Testing System Components

1. **`test_models.py`**: The main script for automated model testing
   - Tests one model at a time to avoid resource conflicts
   - Measures response times and performance metrics
   - Generates detailed reports with timestamped files
   - Maintains historical performance data for each model

2. **Test Questions**: Predefined or custom questions organized into categories
   - General character questions
   - Specific scene questions
   - Technical questions
   - Philosophical questions

3. **Results Directory**: Structured output in the `model_test_results` directory
   - Model-specific folders for each tested model
   - Timestamped test result files for historical tracking
   - "Latest" files showing the most recent test results
   - History files tracking performance metrics over time

### How to Use the Testing System

```bash
# Test the default model (llama3)
python -m tests.test_models

# Test a specific model
python -m tests.test_models --model phi4

# List all available models
python -m tests.test_models --list-models

# Use custom test questions
python -m tests.test_models --model gemma2 --questions-file tests/test_questions.txt

# Customize generation parameters
python -m tests.test_models --model phi3 --temperature 0.8 --max-tokens 800
```

### Results Structure

The test results are organized in the following structure:

```
model_test_results/
├── llama3/
│   ├── llama3_test_20240226_123456.md  # Timestamped test results
│   ├── llama3_latest.md                # Most recent test results
│   └── llama3_history.md               # Performance history table
├── phi4/
│   ├── phi4_test_20240226_123456.md
│   ├── phi4_latest.md
│   └── phi4_history.md
└── ... (other models)
```

### Evaluation Criteria

When evaluating models for ViktorAI, consider the following criteria:

1. **Character Consistency**: How well the model maintains Viktor's voice and personality
2. **Factual Accuracy**: Whether responses correctly reference events from Arcane Season 1
3. **Response Quality**: The depth, coherence, and relevance of responses
4. **Performance**: Response time and generation efficiency
5. **Hallucination Resistance**: Ability to avoid making up events that didn't happen in the show

The testing system provides a structured way to compare these aspects across different models, helping to make an informed decision about which model to use for the best character AI experience.

## Benchmarking System

The benchmarking system (`run_benchmark.py`) provides a more comprehensive evaluation than the testing system, with category-specific prompting and detailed metrics:

### Benchmark Features

1. **Question Categorization**: Automatically categorizes questions into four types:
   - Identity questions: About Viktor's self-perception and background
   - Technical questions: About Hextech, research, and scientific concepts
   - Relationship questions: About Viktor's connections with other characters
   - Philosophical questions: About Viktor's worldview and values

2. **Category-Specific Prompting**: Uses different prompt configurations based on question type:
   - Identity → personality-focused prompt
   - Technical → technical-focused prompt
   - Relationship → relationship-focused prompt
   - Philosophical → full prompt

3. **Weighted Scoring**: Calculates scores based on importance for each question type:
   - Primary dimension score (60%): Question-type specific criteria
   - Character consistency score (40%): General character accuracy

4. **Detailed Reporting**: Generates comprehensive reports with:
   - Overall scores by category and question type
   - Individual question scores and evaluations
   - Weighted metrics that emphasize important criteria
   - HTML visualization for easy interpretation

### Running Benchmarks

```bash
# Run a complete benchmark with default settings
python run_benchmark.py

# Specify a model to benchmark
python run_benchmark.py --model gemma3:1b

# Run only specific question categories
python run_benchmark.py --categories identity,technical

# Use mock mode for testing without actual LLM calls
python run_benchmark.py --use-mock
```

The benchmark results provide valuable insights for comparing model performance and identifying areas for improvement in the character implementation.

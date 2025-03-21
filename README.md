# ViktorAI

A Python-based chatbot that embodies Viktor from Arcane Season 1. This project enables users to have text conversations with Viktor, with responses that accurately reflect his authentic character.

## Project Overview

ViktorAI is a character AI project that:

- Loads detailed character data from markdown files
- Uses Retrieval-Augmented Generation (RAG) for accurate, in-character responses
- Interfaces with Ollama LLMs to generate responses in Viktor's voice
- Maintains character consistency throughout conversations
- Can reference detailed scene analysis for specific queries

## Project Structure

The project is organized into several key components:

- **character_data/**: Character analysis files for Viktor
- **src/**: Core source code including RAG implementation
- **tests/**: Test scripts and test data
- **model_test_results/**: Performance results for different LLMs
- **vector_store/**: Vector database files for semantic search
- **main.py**: Main entry point for running the chatbot
- **build_vector_store.py**: Script to build the vector database
- **everything.md**: Comprehensive technical documentation

```
ViktorAI/
├── character_data/           # Character analysis files
├── src/                      # Source code
├── tests/                    # Test code
├── model_test_results/       # Model test results
├── vector_store/             # Vector database files
├── main.py                   # Main entry point
├── build_vector_store.py     # Script to build the vector store
├── requirements.txt          # Dependencies
└── everything.md             # Comprehensive documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/urosaron/ViktorAI.git
   cd ViktorAI
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Build the vector store for RAG capabilities:

   ```bash
   python build_vector_store.py
   ```

5. Ensure Ollama is installed and running:

   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/version
   ```

6. Pull the desired LLM model (if not already available):
   ```bash
   ollama pull llama3
   ```

## Usage

### Running the Chatbot

To start a conversation with Viktor:

```bash
python main.py
```

You can customize the LLM model and parameters:

```bash
python main.py --model llama3 --temperature 0.7 --max_tokens 500
```

### Available Command-Line Arguments

- `--model`: The Ollama model to use (default: llama3)
- `--temperature`: Temperature for response generation (default: 0.7)
- `--max_tokens`: Maximum tokens for response generation (default: 500)

### Example Conversation

```
Welcome to ViktorAI - Viktor from Arcane Season 1
Type 'exit', 'quit', or Ctrl+C to end the conversation.
==================================================

You: What do you think about Jayce's recent focus on politics?

Viktor: Jayce's increasing focus on politics is a natural progression given his position, but it does create a growing disconnect between us. I respect his brilliance and dedication, yet I find myself becoming increasingly frustrated with his caution. Our shared vision for Hextech's potential to improve lives has not wavered, but the path to achieving that goal seems to be diverging.
```

## Running Tests

To run the automated test suite:

```bash
python -m unittest discover tests
```

This command runs all test files in the `tests` directory. The tests verify that:

- Character data loads correctly
- The RAG system retrieves relevant information
- The chatbot generates appropriate responses

## Knowledge Boundaries

Viktor's knowledge is limited to events of Arcane Season 1. He:

- Has knowledge of the entire Season 1
- Cannot know future events beyond Season 1
- Maintains character consistency with Viktor's portrayal in the show

## License

[MIT License](LICENSE)

## Acknowledgments

- Arcane and its characters are the property of Riot Games and Fortiche Productions
- This project is for educational and entertainment purposes only

## Documentation

For detailed documentation about how the system works, including:

- Comprehensive explanation of the RAG system
- Data flow and architecture
- Code structure and components
- Advanced usage and customization
- Model testing and evaluation

Please refer to the [everything.md](everything.md) file.

## Response Quality Classifier

ViktorAI now includes a PyTorch-based response quality classifier to evaluate and ensure high-quality, in-character responses:

### Features

- Uses a neural network to evaluate both character accuracy and response quality
- Automatically rejects and regenerates low-quality responses
- Improves character consistency across different LLM models
- Provides scoring metrics for responses

### Setup and Training

1. Generate training data:

   ```bash
   python -m src.generate_classifier_data
   ```

2. Train the classifier:

   ```bash
   python train_classifier.py
   ```

3. Use the classifier with ViktorAI:
   ```bash
   python main.py --use-classifier
   ```

### Command-Line Arguments

- `--use-classifier`: Enable the PyTorch response classifier
- `--no-classifier`: Disable the classifier even if available
- `--min-score`: Set minimum acceptable score (0.0-1.0, default: 0.6)
- `--debug`: Show detailed scoring information

### How It Works

The classifier uses a simple feed-forward neural network with two output heads:

1. **Character Accuracy**: Evaluates how well the response matches Viktor's character
2. **Response Quality**: Evaluates the overall coherence and relevance of the response

When enabled, ViktorAI will regenerate responses that score below the minimum threshold, ensuring better quality interactions.

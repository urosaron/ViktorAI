# ViktorAI

A Python-based chatbot that embodies Viktor from Arcane Season 1. This project enables users to have text conversations with Viktor, with responses that accurately reflect his authentic character.

## Project Overview

ViktorAI is a character AI project that:

- Loads detailed character data from markdown files
- Uses Retrieval-Augmented Generation (RAG) for accurate, in-character responses
- Interfaces with Ollama LLMs to generate responses in Viktor's voice
- Maintains character consistency throughout conversations
- Can reference detailed scene analysis for specific queries
- Integrates with ViktorBrain for neurally-influenced response generation
- Provides a REST API for chatbot interaction

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
│   ├── api.py                # REST API implementation
│   ├── brain_client.py       # Client for ViktorBrain integration
│   ├── viktor_ai.py          # Core AI implementation
│   └── ...                   # Other source files
├── tests/                    # Test code
├── model_test_results/       # Model test results
├── vector_store/             # Vector database files
├── main.py                   # Main entry point
├── build_vector_store.py     # Script to build the vector store
├── Dockerfile.ai             # Docker container definition
├── requirements.txt          # Dependencies
└── everything.md             # Comprehensive documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- (Optional) ViktorBrain for neural integration

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

To start a conversation with Viktor through the command line:

```bash
python main.py
```

You can customize the LLM model and parameters:

```bash
python main.py --model llama3 --temperature 0.7 --max_tokens 500
```

### Starting the API Server

To run the REST API service:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Alternatively, you can use the system management script from the ViktorBrain repository:

```bash
# From ViktorBrain directory
python scripts/start_system.py start --ai-only
```

### API Endpoints

The REST API provides these endpoints:

- `GET /`: API status information
- `POST /chat`: Send a message to Viktor
- `GET /brain_status`: Check ViktorBrain connection status
- `POST /reset_brain`: Reset the ViktorBrain connection
- `GET /model_info`: Get information about the current AI configuration

Example API usage:

```python
import requests

# Check API status
response = requests.get("http://localhost:8080/")
print(response.json())

# Send a chat message
chat_response = requests.post(
    "http://localhost:8080/chat",
    json={"message": "Hello Viktor, what are you working on today?"}
)
print(chat_response.json()["response"])
```

### Available Command-Line Arguments

- `--model`: The Ollama model to use (default: llama3)
- `--temperature`: Temperature for response generation (default: 0.7)
- `--max_tokens`: Maximum tokens for response generation (default: 500)
- `--use-brain`: Enable ViktorBrain integration (default: True if available)
- `--brain-api`: ViktorBrain API URL (default: http://localhost:8000)

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

## Docker Deployment

You can build and run ViktorAI using Docker:

```bash
# Build the Docker image
docker build -f Dockerfile.ai -t viktorai .

# Run the container
docker run -p 8080:8080 viktorai
```

## Integration with ViktorBrain

ViktorAI can integrate with ViktorBrain to create a neurally-influenced character AI system. This integration:

- Processes user inputs through a simulated neural network
- Adjusts response parameters based on brain state
- Provides feedback to the brain based on conversations
- Creates a more dynamic and contextually aware character

To use this integration:

1. Ensure ViktorBrain is running:
   ```bash
   # From ViktorBrain directory
   python scripts/start_system.py start --brain-only
   ```

2. Start ViktorAI with brain integration:
   ```bash
   python main.py --use-brain
   ```

Or use the combined system:
```bash
# From ViktorBrain directory
python scripts/start_system.py start
```

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
   python -m scripts.generate_classifier_data
   ```

2. Train the classifier:

   ```bash
   python scripts/train_classifier.py
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

### Testing

To run the classifier tests:

```bash
# Test the core classifier functionality
python -m tests.test_response_classifier

# Test the training pipeline
python -m tests.test_classifier_training

# Run all tests
python -m unittest discover tests
```

The tests validate:

- Feature extraction from text
- Model initialization and architecture
- Training process and effectiveness
- Model saving and loading
- Response evaluation accuracy

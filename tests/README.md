# ViktorAI Testing System

This directory contains testing tools for the ViktorAI project.

## Model Testing System

The `test_models.py` script allows you to test a single model at a time with a set of predefined questions and save the results in a structured format.

### Key Features

- Tests one model at a time to avoid resource conflicts
- Organizes results in model-specific folders
- Saves timestamped test results for historical tracking
- Maintains a "latest" file for each model with the most recent results
- Keeps a history file with performance metrics across all test runs
- Supports custom test questions

### Usage

```bash
# Test the default model (llama3)
python -m tests.test_models

# Test a specific model
python -m tests.test_models --model phi4

# List all available models
python -m tests.test_models --list-models

# Use custom test questions
python -m tests.test_models --model mixtral:8x7b --questions-file tests/model_test_questions.txt

# Customize generation parameters
python -m tests.test_models --model hermes3 --temperature 0.8 --max-tokens 800
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

### Test Questions

The default test questions are defined in the script, but you can also use a custom file with your own questions. The file format is simple:

- One question per line
- Lines starting with `#` are treated as comments and ignored

See `model_test_questions.txt` for an example.

## Unit Tests

The `test_character_loader.py` file contains unit tests for the ViktorAI character data loading components. To run these tests:

```bash
python -m unittest tests.test_character_loader
``` 
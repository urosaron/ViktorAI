# ViktorAI Benchmarking System

This benchmarking system implements a structured approach to evaluate and compare different prompt configurations for ViktorAI. It follows the recommendations from Leon to establish a baseline, develop custom metrics, and provide comparative assessment.

## Overview

The benchmarking system:

1. **Establishes a baseline** using a simple, generic prompt
2. **Tests specialized prompts** that focus on different aspects of Viktor's character
3. **Evaluates outputs** using both qualitative and quantitative metrics
4. **Provides comparative assessment** between different prompt configurations
5. **Generates visualizations and reports** to help analyze the results

## Prompt Categories

The system tests the following prompt categories:

- **baseline**: A minimal prompt with basic information about Viktor
- **personality_focused**: Emphasizes Viktor's personality traits and mannerisms
- **technical_focused**: Emphasizes Viktor's technical knowledge and scientific approach
- **relationship_focused**: Emphasizes Viktor's relationships with other characters
- **full**: The complete specialized prompt used in the main application

## Evaluation Metrics

The system uses multiple metrics to evaluate responses:

### Basic Metrics
- Response length
- Word count
- Character-specific keyword presence
- Response time

### Advanced Metrics (via LLM evaluation)
- Character Authenticity: How well the response captures Viktor's personality
- Technical Accuracy: How accurate the response is to Viktor's knowledge
- Emotional Resonance: How well the response conveys Viktor's emotional state
- Response Quality: How well-structured and appropriate the response is
- Overall Score: A combined score representing the overall quality

## Usage

### Basic Usage

```bash
python -m tests.benchmark
```

This will run the benchmark with default settings:
- Model: llama3
- All prompt categories
- Questions from model_test_questions.txt

### Specify a Different Model

```bash
python -m tests.benchmark --model phi4
```

### Test Specific Prompt Categories

```bash
python -m tests.benchmark --categories baseline personality_focused
```

### Use a Different Evaluator Model

```bash
python -m tests.benchmark --evaluator-model mixtral:8x7b
```

### Custom Questions File

```bash
python -m tests.benchmark --questions-file my_questions.txt
```

### Baseline Mode Testing

```bash
python -m tests.benchmark --baseline-mode
```

This will run the benchmark using a minimal prompt regardless of the category, allowing you to compare how the model performs with and without detailed character data.

### Full Options

```bash
python -m tests.benchmark --model llama3 --evaluator-model llama3 --temperature 0.7 --max-tokens 500 --questions-file tests/model_test_questions.txt --output-dir benchmark_results --categories baseline full
```

## Output

The benchmark generates several outputs:

1. **JSON Results**: Complete benchmark data including all responses and metrics
2. **Visualizations**: Charts comparing different prompt categories
3. **HTML Report**: A detailed report with all responses and evaluations
4. **CSV Tables**: Summary statistics for easy import into spreadsheets

All outputs are saved to the specified output directory (default: `benchmark_results`).

## Interpreting Results

The benchmark results help answer several key questions:

1. **How much does the specialized prompt improve over the baseline?**
   - Compare the overall scores between baseline and specialized prompts

2. **Which aspects of Viktor's character are captured well?**
   - Look at the individual metrics (authenticity, technical accuracy, etc.)

3. **Which prompt configuration works best?**
   - Compare the different specialized prompts to see which performs best

4. **Are there specific questions where the model struggles?**
   - Review the detailed results to identify areas for improvement

## Example Workflow

1. Run a baseline benchmark:
   ```bash
   python -m tests.benchmark --baseline-mode
   ```

2. Run a full data benchmark:
   ```bash
   python -m tests.benchmark
   ```

3. Compare the results to see how much your specialized prompt improves over the baseline

4. Test different specialized prompts:
   ```bash
   python -m tests.benchmark --categories personality_focused technical_focused relationship_focused
   ```

5. Identify which aspects of Viktor's character are captured well and which need improvement

6. Refine your prompts based on the results and repeat the process 
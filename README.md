# Prompt Enhancement Benchmark Framework

A code example for evaluating and enhancing LLM prompting strategies on the GPQA (Graduate-Level Google-Proof Q&A) benchmark.

## Overview

This repository contains code for:
1. **Benchmark Testing**: Evaluate various prompting strategies (zero-shot, chain-of-thought, self-consistency, self-refine) on GPQA
2. **Failure Analysis**: Automatically analyze failed questions to identify error patterns
3. **Prompt Enhancement**: Generate targeted prompt improvements based on failure analysis

## Features

- **Multiple Prompting Strategies**:
  - Zero-shot direct answering
  - Zero-shot Chain-of-Thought (CoT)
  - Self-consistency with majority voting
  - Self-refine with iterative improvement

- **Three Enhancement Types**:
  - **Concise**: Quick warnings and key points
  - **Specific**: Detailed guidance with verification steps
  - **Reasoning**: Minimal hints for self-discovery

- **Robust Infrastructure**:
  - Checkpointing for resumable runs
  - Answer shuffling for multiple choice questions
  - Category-based analysis (Physics, Chemistry, Biology)
  - Automated evaluation using OpenAI models

## Repository Structure

```
.
├── strategies/
│   ├── zero-shot.py                 # Zero-shot baseline benchmark
│   ├── zero-shot-cot.py            # Chain-of-thought benchmark
│   ├── self-consistency.py         # Self-consistency with majority voting
│   ├── zero-shot-enhancement.py    # Enhancement generator for zero-shot
│   ├── zero-shot-cot-enhancement.py # Enhancement generator for CoT
│   └── self-refine-enhancement.py  # Enhancement generator for self-refine
├── data/                           # GPQA dataset files (not included)
├── results/                        # Benchmark results output
├── checkpoints/                    # Checkpoint files for resumable runs
├── enhanced_prompts/               # Generated prompt enhancements
├── prompts.json                    # Prompt templates configuration
├── requirements.txt                # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/prompt-enhancement-benchmark.git
cd prompt-enhancement-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### API Keys

Set your API keys as environment variables:

```bash
# For benchmark testing (OpenAI)
export OPENAI_API_KEY="your-openai-api-key"

# For enhancement generation (Together AI)
export TOGETHER_API_KEY="your-together-api-key"
```

### Dataset

Download the GPQA dataset and place it in the `data/` directory:
- `gpqa_diamond_train.csv`
- `gpqa_diamond_test.csv`

## Usage

### Running Benchmarks

#### Zero-Shot Baseline
```bash
python strategies/zero-shot.py \
    --data-file ./data/gpqa_diamond_test.csv \
    --prompt-type zero_shot \
    --model gpt-4o
```

#### Chain-of-Thought
```bash
python strategies/zero-shot-cot.py \
    --data-file ./data/gpqa_diamond_test.csv \
    --prompt-type zero_shot_cot \
    --model gpt-4o
```

#### Self-Consistency (Multiple Samples)
```bash
python strategies/self-consistency.py \
    --data-file ./data/gpqa_diamond_test.csv \
    --prompt-type zero_shot_self_consistency \
    --model gpt-4o
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-file` | Path to GPQA CSV file | `./data/gpqa_diamond_train.csv` |
| `--data-dir` | Directory containing data files | `./data` |
| `--model` | Model to use for evaluation | `gpt-4o` |
| `--prompt-type` | Prompt strategy key from prompts.json | `zero_shot_cot` |
| `--start-index` | Start question index | `0` |
| `--end-index` | End question index | `None` (all) |
| `--categories` | Filter by category | `None` (all) |
| `--checkpoint-interval` | Save checkpoint every N questions | `5` |
| `--resume-from` | Resume from checkpoint file | `None` |
| `--no-shuffle` | Disable answer choice shuffling | `False` |

### Generating Prompt Enhancements

After running benchmarks, generate enhanced prompts from failure analysis:

```bash
python strategies/zero-shot-enhancement.py \
    --input ./results/gpqa_results_YYYYMMDD_HHMMSS \
    --output-dir ./enhanced_prompts \
    --model Qwen/Qwen2.5-72B-Instruct-Turbo \
    --enhancement-types concise specific reasoning
```

### Enhancement Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to benchmark results directory | Required |
| `--output-dir` | Directory for enhanced prompts | `./enhanced_prompts` |
| `--model` | Model for enhancement generation | `Qwen/Qwen2.5-72B-Instruct-Turbo` |
| `--enhancement-types` | Types to generate | `concise specific reasoning` |
| `--max-questions` | Max questions per category | `None` (all) |
| `--categories` | Filter by category | `None` (all) |
| `--stream` | Use streaming API calls | `False` |

## Prompt Configuration

Edit `prompts.json` to customize prompt templates:

```json
{
  "prompts": {
    "zero_shot_cot": {
      "template": "Solve this problem step by step.\nQuestion: {question}\n\n<reasoning>\n[Your reasoning here]\n</reasoning>\n\n<answer>\n[Final answer]\n</answer>",
      "sampling_params": {
        "temperature": 0,
        "num_samples": 1
      }
    }
  }
}
```

## Output Format

### Benchmark Results
Results are saved in JSON format with per-question details:
- Correct/predicted answers
- Reasoning traces
- Category and subdomain information
- Confidence scores (for self-consistency)

### Enhanced Prompts
Three enhancement types are generated:
1. **Concise**: Brief, targeted guidance
2. **Specific**: Detailed step-by-step instructions
3. **Reasoning**: Socratic-style hints

## Supported Models

### Benchmark Testing (OpenAI)
- `gpt-4o`
- `gpt-4o-mini`
- `o3-mini` (for evaluation)

### Enhancement Generation (Together AI)
- `Qwen/Qwen2.5-72B-Instruct-Turbo`
- `meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo`
- `deepseek-ai/DeepSeek-V3`
- `moonshotai/Kimi-K2-Thinking`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{prompt-enhancement-benchmark,
  title={Prompt Enhancement Benchmark Framework},
  author={Anonymous},
  year={2025},
  howpublished={\url{https://github.com/anonymous/prompt-enhancement-benchmark}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- GPQA dataset creators
- OpenAI for GPT models
- Together AI for hosted model inference

# TabBridge: Bridging Structure and Context for Accurate Table Reasoning

## Abstract
> TabBridge

## Method Overview

Our research focuses on improving LLM reasoning and QA performance over table data through:
![Method Overview](method_overview.png)


## Installation & Setup

### 1. Environment Setup

```bash
# Create virtual environment (Python 3.9 recommended)
python3.9 -m venv env

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Execution

```bash
python run_tabbridge.py
```

## Project Structure

```
TabBridge/
├── 📁 datasets/                    # Dataset files
│   ├── wtq.jsonl                  # WikiTableQuestions dataset (main)
│   └── wtq.json                   # Alternative format
├── 📁 prompt/                     # All prompt templates
│   ├── SS_Generation.txt         # Structured Specification generation
│   ├── SQL_Reasoning.txt         # SQL query generation
│   ├── SS_row_analysis_evaluation.txt  # Direct SS evaluation
│   ├── SQL_Evaluation.txt        # SQL quality assessment
│   ├── 📁 SS_feedback/           # SS refinement prompts
│   │   ├── row_analysis_feedback.txt
│   │   ├── header_feedback.txt
│   │   └── structure_feedback.txt
│   ├── 📁 SQL_feedback/          # SQL refinement prompts
│   │   ├── appropriate_role.txt
│   │   ├── well_used_row_analysis.txt
│   │   ├── appropriate_data_type.txt
│   │   └── faithfulness.txt
│   └── 📁 Table_Generation/      # Table reconstruction prompts
│       ├── Schema_Extraction.txt
│       ├── Instruction_Extraction.txt
│       └── Table_Generation.txt
├── 📁 utils/                      # Core utility modules
│   ├── preprocess.py             # Data preprocessing utilities
│   ├── prompt_wtq.py             # Prompt loading and management
│   ├── ss_row_analysis_evaluator.py      # Direct SS evaluation
│   ├── ss_reconstruction_based_evaluator.py  # Reconstruction-based SS evaluation
│   └── sql_evaluator.py          # SQL quality evaluation
├── 📁 outputs/                    # Generated results (auto-created)
│   ├── 📁 subtables/             # Cached sub-table extractions
│   ├── 📁 ss_logs/               # SS generation & evaluation logs
│   ├── 📁 sql_logs/              # SQL generation & execution logs
│   ├── wtq_results.jsonl         # Final results (JSON format)
│   └── wtq_results.csv           # Final results (CSV format)
├── run_tabbridge.py              # Main execution script
├── subtable_extractor_euclid.py  # Sub-table extraction logic
└── requirements.txt              # Python dependencies
```
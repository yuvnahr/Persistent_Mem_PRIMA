# LoCoMo Benchmark

This folder contains the benchmark evaluation code for comparing baseline vs PRIMA agents on the LoCoMo dataset.

## Setup

1. Download the LoCoMo dataset:
   ```bash
   cd benchmarks/data
   # Download locomo10.json from the A-mem repository or source
   # The dataset should be available at: https://github.com/AGI-Edgerunners/A-mem/tree/main/data
   ```

2. Install additional dependencies:
   ```bash
   pip install rouge-score bert-score nltk sentence-transformers
   ```

## Files

- `load_dataset.py`: Dataset loading utilities
- `utils.py`: Evaluation metrics calculation
- `baseline_agent.py`: Baseline agent (no memory)
- `prima_agent.py`: PRIMA agent (with memory system)
- `evaluate_benchmark.py`: Main evaluation script

## Running the Benchmark

```bash
cd benchmarks
python evaluate_benchmark.py
```

This will:
1. Load the LoCoMo dataset
2. Evaluate both baseline and PRIMA agents
3. Calculate metrics (F1, BLEU, ROUGE, BERTScore, etc.)
4. Save results to `experiments_outputs/`
5. Print aggregated results

## Expected Output

Results are saved as JSON files and include metrics for each question category (1-5) and overall performance.
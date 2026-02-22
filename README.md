# CS288 Assignment 2: Transformer Language Model

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for Part 4, not needed for Part 1-3)

### Setup

1. **Installation:**
```bash
conda create -n cs288a2 python=3.10 -y
conda activate cs288a2
pip install -r requirements.txt
```

## Running Tests

Run tests from within each part's directory:

```bash
# Part 1: Tokenization
cd part1
python -m pytest tests/ -v

# Part 2: Transformer Model
cd part2
python -m pytest tests/ -v

# Part 3: NN Utilities
cd part3
python -m pytest tests/ -v
```

Or run all tests from the source directory:
```bash
cd source
python -m pytest part1/tests/ part2/tests/ part3/tests/ -v
```

## Part 4: Training and Evaluation (Bonus)

After completing Parts 1-3, you can train and evaluate models for bonus points.

### Run Training Pipeline

```bash
cd part4
python train_baseline.py
```

This will:
1. Train a BPE tokenizer on TinyStories
2. Pretrain a transformer language model
3. Fine-tune on multiple-choice QA
4. Evaluate using zero-shot prompting
5. Save predictions to `part4/outputs/`

### Configuration Options

```bash
# Quick test run (smaller model, fewer steps)
python train_baseline.py --quick

# Medium configuration
python train_baseline.py --medium

# Full training (default)
python train_baseline.py
```

### Output Files

After training, prediction files are saved to `part4/outputs/`:
- `finetuned_predictions.json` - Fine-tuned model predictions
- `prompting_predictions.json` - Zero-shot prompting predictions

These files are required for Part 4 bonus points.

## Submission

Create your submission zip file:

```bash
bash create_submission.sh
```

### Part 4 Bonus Scoring
- **Fine-tuned model (12 pts)**: 30% accuracy = 0 pts, 50% = full pts (linear scale)
- **Prompting model (8 pts)**: 0% boost = 0 pts, 2%+ boost over fine-tuned = full pts

## Implementation Requirements

### Part 1: Tokenization
- `train_bpe()`: Train BPE vocabulary from text corpus
- `Tokenizer._bpe()`: Apply BPE merges to a token
- `Tokenizer._encode_chunk()`: Encode text to token IDs
- `Tokenizer.decode()`: Decode token IDs to text

### Part 2: Model Components
- `Linear`: Linear transformation layer
- `Embedding`: Token embedding layer
- `RMSNorm`: Root mean square layer normalization
- `softmax()`: Numerically stable softmax
- `silu()`: SiLU activation function
- `SwiGLU`: Gated feed-forward network
- `RotaryPositionEmbedding`: RoPE positional encoding
- `scaled_dot_product_attention()`: Attention mechanism
- `MultiHeadSelfAttention`: Multi-head attention
- `MultiHeadSelfAttentionWithRoPE`: Attention with RoPE
- `TransformerBlock`: Single transformer layer
- `TransformerLM`: Complete language model
- `count_flops_per_token()`: FLOPs estimation
- `estimate_memory_bytes()`: Memory estimation

### Part 3: Training Utilities
- `softmax()`: Numerically stable softmax (for training)
- `cross_entropy()`: Cross-entropy loss
- `gradient_clipping()`: Gradient norm clipping
- `token_accuracy()`: Token-level accuracy
- `perplexity()`: Language model perplexity

## Important Notes

- Do NOT modify function signatures or class interfaces
- Do NOT add dependencies beyond `requirements.txt`
- Ensure your code passes local tests before submitting
- The autograder runs additional hidden tests
- Use the provided fixtures for testing
# CS288-Advanced-Natural-Language-Processing-Assignment2
# CS282-Advanced-NLP-HW2

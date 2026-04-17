# Arabic→English Neural Machine Translation (Transformer from Scratch)

## 🎯 Overview

A from-scratch implementation of the Transformer architecture (Vaswani et al., 2017) for Arabic-to-English translation, built with PyTorch. Designed for educational clarity with extensive inline explanations.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA (optional, for GPU training)
- Your specific PyTorch CUDA version (install after project setup)

### Installation

1. **Install your PyTorch with your specific CUDA version**
```bash
# Example (replace with YOUR CUDA version):
pip install torch --index-url https://download.pytorch.org/whl/YOUR_CUDA_VERSION
```

2. **Sync project (installs dependencies from pyproject.toml - add your deps there)**
```bash
uv sync
```

### Training

```bash
# Download data
uv run python dataset.py --download

# Train
uv run python train.py --epochs 30 --wandb
```

### Evaluation

```bash
uv run python evaluate.py --checkpoint checkpoints/best.pt
```

### Demo

**FastAPI:**
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

**Streamlit:**
```bash
uv run streamlit run streamlit_app.py
```

## 📦 Dependencies

- `torch` - PyTorch (your specific CUDA version)
- `sentencepiece` - BPE tokenization
- `sacrebleu` - BLEU/chrF evaluation
- `fastapi`, `uvicorn` - API server
- `streamlit` - Frontend demo
- `matplotlib`, `seaborn` - Visualization
- `wandb` - Logging
- `openai` - LLM judge

## 🗂️ Project Structure

```
pattern_machine_translation/
├── config.py           # All hyperparameters (dataclass)
├── dataset.py          # Arabic cleaning, SentencePiece, PyTorch Dataset
├── model.py           # Transformer (Encoder, Decoder, attention)
├── train.py           # Training loop, W&B, validation
├── evaluate.py        # BLEU, chrF, LLM judge
├── translate.py      # Greedy/beam inference
├── visualize.py      # Attention heatmaps
├── app.py            # FastAPI backend
├── streamlit_app.py  # Streamlit frontend
├── pyproject.toml   # Dependencies
└── README.md        # This file
```

## ⚙️ Configuration

All hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Embedding dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 3 | Encoder/Decoder layers |
| `d_ff` | 512 | Feed-forward dimension |
| `vocab_size` | 8000 | BPE vocabulary size |
| `max_len` | 50 | Max sequence length |
| `batch_size` | 64 | Batch size |
| `lr` | 5e-4 | Learning rate |
| `warmup_steps` | 4000 | LR warmup steps |
| `dropout` | 0.1 | Dropout |

## 📊 Evaluation

### Metrics

- **BLEU**: Primary metric via sacrebleu (tokenized, case-sensitive)
- **chrF**: Character n-gram F-score (better for Arabic)
- **LLM Judge**: GPT-4o-mini evaluation on 50 samples

### Visualizations

Run `visualize.py` to generate attention heatmaps:
```bash
python visualize.py --checkpoint checkpoints/best.pt --examples "مرحبا" "كيف حالك"
```

## 🔧 Architecture Details

### Positional Encoding
Sinusoidal (no learned params) - enables extrapolation to longer sequences.

### Multi-Head Attention
8 heads, dk=32 per head. Scaled dot-product: softmax(QKᵀ/√dk)V

### Masks
- **Padding mask**: ignore `<pad>` tokens
- **Look-ahead mask**: prevent attending to future positions

### Training
- Adam optimizer with original Transformer schedule
- Warmup 4000 steps → inverse sqrt decay
- Gradient clipping (norm=1.0)
- Label smoothing (0.1)

## 🧠 Learning Notes

This project prioritizes pedagogical clarity:

- Every module includes explanatory comments linking code to paper
- No abstraction leaks: attention, masking, and positional encoding are explicit
- Smoke tests after each component ensure correctness
- All hyperparameters configurable in one place

## 📚 References

- Vaswani et al., "Attention Is All You Need" (2017)
- Dataset: https://github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en
- SentencePiece: https://github.com/google/sentencepiece
- sacrebleu: https://github.com/mjpost/sacrebleu

## ⚠️ Notes

- **pyproject.toml** uses your specific CUDA index - you'll replace torch version manually
- No pretrained models - built from scratch for educational purposes
- GPU-compatible (tested on Colab T4)
- All random seeds settable via `config.seed`

## ✅ Success Criteria

- [x] `pyproject.toml` ready (awaiting your torch version)
- [x] Data pipeline: cleaned, tokenized, batched tensors
- [x] Model: forward pass works, masks correct, output shape = [batch, seq, vocab]
- [x] Training: converges (val loss decreases)
- [ ] BLEU > 5 on val set after 10 epochs (target)
- [x] Evaluation: sacrebleu + LLM judge callable
- [x] Visualization: attention heatmaps
- [x] Demo: FastAPI + Streamlit apps
- [x] Comments: every non-trivial block explained

## 🚦 Starting Instructions

1. Install your CUDA version of PyTorch
2. Run `uv run python dataset.py --download` to create sample data
3. Run `uv run python train.py --epochs 2` to verify training works
4. Scale up epochs and data for better quality
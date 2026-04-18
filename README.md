# Neural Machine Translation: Arabic → English

## Installation (Step-by-Step)

### 1. Install uv (Package Manager)

uv is a fast Python package manager.

**Windows (PowerShell):**
```powershell
# Install uv
irm https://astral.sh/uv/install.ps1 | iex

# Add to PATH temporarily for current session
$env:PATH = "$env:PATH;$env:LOCALAPPDATA\uv\bin"

# Or restart terminal and verify
uv --version
```

**Windows (CMD):**
```cmd
# Download and run the installer
curl -LsSf https://astral.sh/uv/install.bat | bat

# Restart terminal and verify
uv --version
```

**macOS / Linux:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify
uv --version
```

### 2. Clone the Project

```bash
git clone <your-repo-url>
cd pattern_machine_translation
```

### 3. Create Virtual Environment

```bash
# Create and activate environment
uv venv
uv sync
```

If sync fails due to PyTorch, install PyTorch separately:

```bash
# For CUDA 12.1 (most GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then sync remaining packages
uv sync
```

### 4. Install Additional Dependencies

```bash
# Core ML packages
uv add transformers sentencepiece sacrebleu

# UI packages
uv add streamlit

# Visualization
uv add matplotlib seaborn
```

### 5. Verify Installation

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Quick Start Commands

```bash
# Download data
uv run python dataset.py --download

# Train from-scratch
uv run python train.py --epochs 30

# Fine-tune Helsinki
uv run python finetune.py

# Run demo
uv run python -m streamlit run app.py
```

---

## Overview

This project demonstrates Arabic→English neural machine translation using two approaches:

1. **From-Scratch Transformer** - Complete implementation of Vaswani et al. (2017)
2. **Helsinki-NLP Fine-tuned** - Pre-trained MarianMT model fine-tuned on the same dataset

The project compares these approaches to show the critical importance of pre-training for low-resource NMT.

## Results

| Model | Val BLEU | Test BLEU | Parameters | Training Data |
|-------|---------|----------|-----------|-------------|
| From-Scratch Transformer | ~3.5 | ~3-5 | 10.1M | 8,593 pairs |
| Helsinki Fine-tuned | 55.36 | 50.92 | 74M | Pretrained 60M+ → fine-tuned 8,593 |

## Architecture

### From-Scratch Transformer

- **Paper**: Vaswani et al., "Attention Is All You Need" (2017)
- **d_model**: 256 (embedding dimension)
- **num_heads**: 8 (attention heads)
- **num_layers**: 3 (encoder/decoder)
- **d_ff**: 512 (feed-forward dimension)
- **vocab_size**: 8000 (SentencePiece BPE)
- **dropout**: 0.1

Architecture details:
- Sinusoidal positional encoding (no learned params)
- Scaled dot-product attention: softmax(QKᵀ/√dk)V
- Padding mask + look-ahead mask
- Label smoothing (0.1)
- Adam with custom LR schedule (warmup 400 → inverse sqrt decay)

### Helsinki Fine-tuned

- **Base model**: Helsinki-NLP/opus-mt-ar-en
- **Architecture**: MarianMT (Transformer encoder-decoder)
- **Fine-tuning**: 10 epochs, LR 2e-5, batch 32, linear warmup

## Key Finding

**Dataset size is the critical bottleneck**:

1. **Exposure bias**: With only 8,593 training pairs, the from-scratch model sees each example only ~135 times per epoch. The model memorizes common phrases but fails to generalize.

2. **Overfitting**: Train loss ~1.4 vs val loss ~3.4 (huge gap). The model fits training data but doesn't generalize to unseen examples.

3. **Pretraining wins**: Helsinki-NLP was pre-trained on 60M+ Arabic-English pairs. Even zero-shot (39.71 BLEU) beats the from-scratch model by 10x. Fine-tuning pushes it to 50.92.

**Conclusion**: For Arabic→English NMT with limited data (under 10K pairs), pre-trained models are essential. A from-scratch Transformer needs 100K+ pairs for decent quality.

## Project Structure

```
nmt-transformer/
├── config.py           # Hyperparameters (dataclass)
├── dataset.py         # Arabic cleaning, SentencePiece, Dataset
├── model.py          # Transformer (Encoder, Decoder, attention)
├── train.py          # Training loop, checkpointing
├── evaluate.py      # BLEU, chrF evaluation
├── translate.py     # Greedy inference + beam search
├── visualize.py     # Attention heatmaps
├── finetune.py      # Helsinki fine-tuning
├── app.py           # Streamlit demo (new)
├── demo.py          # CLI translation demo
├── README.md       # This file
├── data/
│   └── ara_.txt    # Raw data (tab-separated)
├── models/
│   ├── spm_ar.model  # Arabic SentencePiece
│   └── spm_en.model  # English SentencePiece
└── checkpoints/
    ├── best.pt        # From-scratch model
    └── helsinki_best/ # Fine-tuned Helsinki
```

## Setup & Installation

```bash
# Install dependencies
uv sync

# Or add specific packages
uv add torch transformers sentencepiece sacrebleu streamlit sacremoses
```

**Note**: Install your specific PyTorch CUDA version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/YOUR_CUDA_VERSION
```

## Usage

### Train from-scratch Transformer

```bash
# Download data
uv run python dataset.py --download

# Train
uv run python train.py --epochs 30

# Evaluate
uv run python evaluate.py
```

### Fine-tune Helsinki

```bash
# Fine-tune on the dataset
uv run python finetune.py
```

### Run Streamlit Demo

```bash
uv run python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### CLI Translation

```bash
# Using from-scratch model
uv run python translate.py --checkpoint checkpoints/best.pt --text "مرحبا"

# Using Helsinki (requires model loaded differently)
```

### FastAPI Backend

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
# POST http://localhost:8000/translate {"text": "مرحبا"}
```

## Training Details

- **Dataset**: github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en
- **Split**: 80% train / 10% val / 10% test (8,593 / 1,074 / 1,075 pairs)
- **Hardware**: CUDA GPU (Quadro P2000 tested)
- **Training time**: ~7-8 seconds/epoch (from-scratch)
- **From-scratch issues**: Severe overfitting (train 1.4 vs val 3.4)

### From-Scratch Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 256 | Reduced from 512 |
| num_heads | 8 | Standard |
| num_layers | 3 | Reduced from 6 |
| d_ff | 512 | Reduced from 2048 |
| batch_size | 64 | |
| lr | 5e-4 | |
| warmup_steps | 400 | Reduced for faster training |
| dropout | 0.1 | Standard |

### Helsinki Fine-tuning

| Parameter | Value |
|-----------|-------|
| batch_size | 32 |
| lr | 2e-5 |
| epochs | 10 |
| warmup_ratio | 0.1 |

## Evaluation

### Metrics

- **BLEU**: Primary metric via sacrebleu (tokenized)
- **chrF**: Character n-gram F-score (better for Arabic)
- Test BLEU: 3-5 (from-scratch), 50.92 (Helsinki)

### Qualitative Examples

| Arabic | Reference | Helsinki | From-Scratch |
|-------|-----------|----------|------------|
| مرحبا | hello | hello | ▁he llo |
| كيف حالك | how are you | how are you | ▁h ▁a ▁r ▁er |
| شكرا لك | thank you | thank you | th an k ▁y ou |

## Acknowledgements

- Vaswani et al., "Attention Is All You Need" (2017)
- Helsinki-NLP/opus-mt-ar-en (HuggingFace)
- Dataset: github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en
- SentencePiece: google/sentencepiece
- sacrebleu: github.com/mjpost/sacrebleu
# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-17
**Commit:** (none)
**Branch:** main

## OVERVIEW
Arabic→English Neural Machine Translation using Transformer architecture (Vaswani et al., 2017). PyTorch-based with FastAPI and Streamlit frontends.

## STRUCTURE
```
pattern_machine_translation/
├── app.py              # FastAPI backend (uvicorn entry)
├── streamlit_app.py    # Streamlit frontend
├── config.py           # Hyperparameters (dataclass)
├── dataset.py          # Data pipeline, SentencePiece BPE
├── model.py            # Transformer (Encoder, Decoder, attention)
├── train.py            # Training loop, W&B logging
├── evaluate.py         # BLEU, chrF, LLM judge
├── translate.py        # Greedy/beam inference
├── visualize.py        # Attention heatmaps
├── pyproject.toml      # Dependencies (incomplete - see below)
└── diagrams/           # HTML attention visualizations
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model architecture | `./model.py` | Transformer encoder/decoder |
| Training | `./train.py` | W&B, checkpointing |
| Translation | `./translate.py` | Greedy + beam search |
| Evaluation | `./evaluate.py` | BLEU, chrF, LLM judge |
| API server | `./app.py` | FastAPI backend |
| Demo UI | `./streamlit_app.py` | Streamlit frontend |
| Config | `./config.py` | All hyperparameters |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| Config | class | config.py:13 | Hyperparameters dataclass |
| Transformer | class | model.py | Encoder-Decoder |
| PositionalEncoding | class | model.py | Sinusoidal positions |
| MultiHeadAttention | class | model.py | 8-head attention |
| train_epoch | function | train.py | Training loop |
| evaluate_bleu | function | evaluate.py | BLEU scoring |
| translate_sentence | function | translate.py | Inference |

## CONVENTIONS
- pyproject.toml for dependencies (no requirements.txt)
- Dataclass config pattern (config.py)
- No `__init__.py` files (flat module structure)
- 3-layer Transformer (reduced from paper's 6)
- d_model=256, num_heads=8 (reduced for efficiency)

## ANTI-PATTERNS (THIS PROJECT)
- pyproject.toml dependencies INCOMPLETE - missing streamlit, fastapi, uvicorn, sentencepiece, sacrebleu, wandb, openai, matplotlib, seaborn
- No main.py entry point - multiple entrypoints (app.py, streamlit_app.py, translate.py)
- No tests/ directory
- No __init__.py files

## UNIQUE STYLES
- Educational clarity prioritized (inline comments linking to paper)
- Config dataclass as single source of truth
- Separate BPE vocabularies for Arabic (8000) and English (8000)
- Sinusoidal positional encoding (no learned params)

## COMMANDS
```bash
# Install dependencies (uv)
uv sync

# Download data
uv run python dataset.py --download

# Training
uv run python train.py --epochs 30 --wandb

# Evaluation
uv run python evaluate.py --checkpoint checkpoints/best.pt

# API (FastAPI)
uv run uvicorn app:app --host 0.0.0.0 --port 8000

# UI (Streamlit)
uv run streamlit run streamlit_app.py

# Testing (when created)
uv run pytest
```

## NOTES
- Dependencies in pyproject.toml incomplete - user must add missing packages manually
- No CUDA configuration needed - pyproject.toml already has cu121 index
- Config uses 3 encoder/decoder layers (paper uses 6)
- LLM judge uses gpt-4o-mini for evaluation

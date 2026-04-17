"""
FastAPI Backend for Arabic→English Translation API.

Features:
1. POST /translate endpoint
2. Load model at startup
3. JSON request/response

Usage:
    uv run python app.py

Then POST to http://localhost:8000/translate:
    {"text": "مرحبا"}
"""

import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import config
from model import Transformer
from translate import translate as translate_fn


# ========== App ==========

app = FastAPI(
    title="Arabic→English NMT API",
    description="Neural Machine Translation API using Transformer from scratch",
    version="0.1.0",
)


# ========== Models ==========


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translation: str
    source: str


# ========== State ==========

model: Optional[Transformer] = None
sp_ar = None
sp_en = None
device: str = "cpu"


def load_model():
    """Load model and SPMs at startup."""
    global model, sp_ar, sp_en, device

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SentencePiece models
    from dataset import load_sentencepiece

    sp_model_path = Path(config.spm_ar_path)
    if not sp_model_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Arabic SPM not found at {sp_model_path}. Run train.py first.",
        )

    sp_en_path = Path(config.spm_en_path)
    if not sp_en_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"English SPM not found at {sp_en_path}. Run train.py first.",
        )

    print("Loading SentencePiece models...")
    sp_ar = load_sentencepiece(config.spm_ar_path)
    sp_en = load_sentencepiece(config.spm_en_path)

    # Load Transformer
    checkpoint_path = config.checkpoint_dir / config.checkpoint_best
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint not found at {checkpoint_path}. Run train.py first.",
        )

    print("Loading Transformer model...")
    model = Transformer(
        src_vocab_size=config.vocab_size_ar,
        tgt_vocab_size=config.vocab_size_en,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully!")


# ========== Endpoints ==========


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Model will be loaded on first request if not available.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Arabic→English NMT API",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate Arabic text to English.

    Request:
        {"text": "مرحبا"}

    Response:
        {"translation": "hello", "source": "مرحبا"}
    """
    global model, sp_ar, sp_en, device

    # Load model if not loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Validate input
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Translate
        translation = translate_fn(
            model,
            request.text,
            sp_ar,
            sp_en,
            device,
        )

        return TranslateResponse(
            translation=translation,
            source=request.text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Entry Point ==========

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

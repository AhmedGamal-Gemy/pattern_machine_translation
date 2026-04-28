"""
Fine-tuning Helsinki-NLP/opus-mt-en-ar on the English-Arabic parallel corpus.

Run:
    uv run python finetune.py

Requirements:
    uv add transformers sentencepiece sacrebleu datasets torch
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import sacrebleu

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"
DATA_PATH = Path("data/parallel_corpus.json")
CHECKPOINT = Path("checkpoints/helsinki_en_ar_best.pt")
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
MAX_LEN = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Dataset ─────────────────────────────────────────────────────────────────


class ArEnDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=50):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src = self.pairs[idx]["en"]  # English as source
        tgt = self.pairs[idx]["ar"]  # Arabic as target
        return src, tgt


def collate_fn(batch, tokenizer, max_len):
    srcs, tgts = zip(*batch)

    # Tokenize source (English)
    model_inputs = tokenizer(
        list(srcs),
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Tokenize target (English) — Marian uses same tokenizer for both
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            list(tgts),
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids

    # Replace padding token id with -100 so loss ignores it
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    return model_inputs


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_bleu(model, tokenizer, pairs, device, n=200):
    """Compute BLEU on first n pairs."""
    model.eval()
    hyps, refs = [], []

    samples = pairs[:n]
    for item in samples:
        src = item["ar"]
        ref = item["en"]

        inputs = tokenizer(
            [src],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)

        with torch.no_grad():
            translated = model.generate(**inputs, max_new_tokens=MAX_LEN)

        hyp = tokenizer.decode(translated[0], skip_special_tokens=True)
        hyps.append(hyp)
        refs.append(ref)

    result = sacrebleu.corpus_bleu(hyps, [refs])
    return round(result.score, 2)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    print(f"Device: {DEVICE}")

    # Load data
    print("Loading data...")
    data = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                data.append({"ar": parts[1].strip(), "en": parts[0].strip()})
    print(f"Loaded {len(data)} pairs")

    # Split 80/10/10
    n = len(data)
    train_data = data[: int(n * 0.8)]
    val_data = data[int(n * 0.8) : int(n * 0.9)]
    test_data = data[int(n * 0.9) :]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    # Load pretrained model + tokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Dataloaders
    from functools import partial

    collate = partial(collate_fn, tokenizer=tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(
        ArEnDataset(train_data, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        ArEnDataset(val_data, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Training loop
    best_bleu = 0.0
    CHECKPOINT.parent.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)

        # ── Evaluate ──
        bleu = evaluate_bleu(model, tokenizer, val_data, DEVICE)
        print(f"\nEpoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f} | Val BLEU: {bleu}")

        # ── Save best ──
        if bleu > best_bleu:
            best_bleu = bleu
            model.save_pretrained(CHECKPOINT.parent / "helsinki_best")
            tokenizer.save_pretrained(CHECKPOINT.parent / "helsinki_best")
            print(
                f"  ✓ New best BLEU: {bleu} — saved to {CHECKPOINT.parent}/helsinki_best"
            )

    print(f"\nTraining complete. Best BLEU: {best_bleu}")

    # ── Final test BLEU ──
    print("\nRunning final test BLEU...")
    model_best = MarianMTModel.from_pretrained(CHECKPOINT.parent / "helsinki_best").to(
        DEVICE
    )
    test_bleu = evaluate_bleu(
        model_best, tokenizer, test_data, DEVICE, n=len(test_data)
    )
    print(f"Test BLEU: {test_bleu}")


if __name__ == "__main__":
    main()

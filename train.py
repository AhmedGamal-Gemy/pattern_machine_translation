"""
Training Loop for Arabic→English NMT Transformer.

Features:
1. Adam optimizer with original Transformer learning rate schedule
2. Gradient clipping
3. Validation with loss computation
4. Checkpoint saving (best model by val loss)
5. Weights & Biases integration

Design rationale:
- LR schedule: warmup 4000 steps → inverse sqrt decay (per paper)
- Early stopping: prevents overfitting, saves best checkpoint
- Teacher forcing: always use ground truth (no scheduled sampling)
"""

import os
import math
import time
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from config import config
from model import Transformer


# ========== Learning Rate Schedule ==========


def get_lr_schedule(warmup_steps: int, d_model: int):
    """
    Original Transformer learning rate schedule (Vaswani et al., Section 5.3).

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Why this schedule:
    - Warmup: allows model to explore before committing to gradients
    - After warmup: decays slower than 1/sqrt(step) for better convergence

    Args:
        warmup_steps: Number of warmup steps (4000 in paper)
        d_model: Model dimension

    Returns:
        Lambda function for learning rate
    """
    d_model_scale = d_model**-0.5

    def lr_lambda(step: int) -> float:
        step = max(1, step)  # Avoid division by zero
        return d_model_scale * min(step**-0.5, step * warmup_steps**-1.5)

    return lr_lambda


# ========== Loss Function ==========


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss (Vaswani et al., Section 3.4).

    Why label smoothing:
    - Prevents model from becoming overconfident
    - Acts as regularization
    - Smoothing factor 0.1 is standard

    Instead of:
        CrossEntropyLoss where target is exact (one-hot with 1.0 at correct class)
    We use:
        CrossEntropyLoss where target is soft (1.0 - epsilon distributed across classes)
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

        # Confidence (1 - smoothing) for correct class
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            logits: Model output, shape [batch, seq_len, vocab_size]
            target: Target IDs, shape [batch, seq_len]

        Returns:
            Scalar loss
        """
        # Flatten for crossentropy: [batch, seq_len, vocab] → [batch*seq_len, vocab]
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.contiguous().view(-1, vocab_size)

        # Flatten target
        target = target.contiguous().view(-1)

        # Create smooth labels
        # Start with uniform distribution: smoothing / vocab_size for all classes
        true_dist = torch.full_like(logits, self.smoothing / (vocab_size - 1))

        # Set confidence for correct class
        # Use scatter: true_dist[batch_idx, true_class] = confidence
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Ignore padding index
        if self.ignore_index is not None:
            # Zero out distribution for padding (will be masked in KL divergence)
            true_dist[target == self.ignore_index] = 0

        # KL divergence loss (equivalent to cross-entropy with smooth targets)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = (-true_dist * log_probs).sum(dim=-1)

        # Mask loss for padding
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


# ========== Training ==========


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[LambdaLR] = None,
    criterion: nn.Module = None,
    grad_clip_norm: float = 1.0,
    device: str = "cuda",
) -> float:
    """
    Train for one epoch.

    Args:
        model: Transformer model
        dataloader: Training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        grad_clip_norm: Gradient clipping norm
        device: Device

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)

        # Target input (everything except last token) for teacher forcing
        tgt_input = tgt[:, :-1]
        # Target label (everything except first token) for loss
        tgt_label = tgt[:, 1:]

        # Forward pass
        logits = model(src, tgt_input)

        # Compute loss
        if criterion is None:
            criterion = nn.CrossEntropyLoss(
                ignore_index=config.PAD_ID, label_smoothing=0.1
            )

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        # Step scheduler after each batch (per paper)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # Log batch progress
        if (batch_idx + 1) % 10 == 0:
            lr = (
                scheduler.get_last_lr()[0]
                if scheduler
                else optimizer.param_groups[0]["lr"]
            )
            print(
                f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, LR: {lr:.6f}"
            )

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module = None,
    device: str = "cuda",
) -> float:
    """
    Validate model.

    Args:
        model: Transformer model
        dataloader: Validation data
        criterion: Loss function
        device: Device

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            logits = model(src, tgt_input)

            if criterion is None:
                criterion = nn.CrossEntropyLoss(
                    ignore_index=config.PAD_ID, label_smoothing=0.1
                )

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        val_loss: Validation loss
        path: Save path
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    path: Path,
    device: str = "cuda",
) -> dict:
    """
    Load checkpoint.

    Args:
        model: Model
        optimizer: Optimizer (optional)
        path: Checkpoint path
        device: Device

    Returns:
        Checkpoint dict
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


# ========== Main Training Loop ==========


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 30,
    lr: float = 5e-4,
    warmup_steps: int = 4000,
    grad_clip_norm: float = 1.0,
    early_stopping_patience: int = 5,
    use_wandb: bool = True,
    checkpoint_dir: Path = None,
    device: str = "cuda",
) -> Transformer:
    """
    Full training loop.

    Args:
        model: Transformer model
        train_loader: Training data
        val_loader: Validation data
        epochs: Number of epochs
        lr: Learning rate
        warmup_steps: LR warmup steps
        grad_clip_norm: Gradient clipping
        early_stopping_patience: Early stopping patience
        use_wandb: Use Weights & Biases
        checkpoint_dir: Checkpoint directory
        device: Device

    Returns:
        Trained model
    """
    if checkpoint_dir is None:
        checkpoint_dir = config.checkpoint_dir

    # Setup Weights & Biases
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config={
                    "d_model": config.d_model,
                    "num_heads": config.num_heads,
                    "num_layers": config.num_layers,
                    "d_ff": config.d_ff,
                    "dropout": config.dropout,
                    "batch_size": config.batch_size,
                    "lr": lr,
                    "warmup_steps": warmup_steps,
                    "epochs": epochs,
                },
            )
            use_wandb = True
        except ImportError:
            print("wandb not installed, skipping")
            use_wandb = False

    # Optimizer (Adam with original Transformer betas)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )

    # Learning rate scheduler
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=get_lr_schedule(warmup_steps, config.d_model),
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID, label_smoothing=0.1)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            grad_clip_norm,
            device,
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Timing
        epoch_time = time.time() - start_time
        lr_current = scheduler.get_last_lr()[0]

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_time:.1f}s, LR: {lr_current:.6f}")

        # Log to W&B
        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": lr_current,
                    "epoch": epoch,
                }
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_path = checkpoint_dir / config.checkpoint_best
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"New best! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    best_path = checkpoint_dir / config.checkpoint_best
    if best_path.exists():
        load_checkpoint(model, None, best_path, device)
        print(f"Loaded best model from {best_path}")

    # Finish W&B
    if use_wandb:
        wandb.finish()

    return model


# ========== Entry Point ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Set random seeds
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Import dataset functions
    from dataset import (
        download_data,
        load_parallel_data,
        train_sentencepiece,
        load_sentencepiece,
        create_dataloaders,
    )

    # Download data
    print("Downloading data...")
    data = load_parallel_data()
    print(f"Loaded {len(data)} parallel sentences")

    # Train SentencePiece (or load if exists)
    print("Training SentencePiece models...")

    # Extract texts for training
    ar_texts = [item["ar"] for item in data]
    en_texts = [item["en"] for item in data]

    # Train Arabic model
    sp_ar = train_sentencepiece(ar_texts, "ar", config.vocab_size_ar)

    # Train English model
    sp_en = train_sentencepiece(en_texts, "en", config.vocab_size_en)

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(sp_ar, sp_en, data)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("Creating model...")
    model = Transformer(
        src_vocab_size=config.vocab_size_ar,
        tgt_vocab_size=config.vocab_size_en,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    print("Starting training...")
    model = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        use_wandb=args.wandb,
        device=device,
    )

    print("Training complete!")

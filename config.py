"""
Configuration file for Arabic→English NMT Transformer.

All hyperparameters in one place as a dataclass for clean imports and type safety.
Links to paper: Vaswani et al., "Attention Is All You Need" (2017)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """
    Hyperparameter configuration for the Transformer model.

    Design rationale: Centralizing all hyperparameters enables:
    - Easy experimentation (change one place)
    - Reproducibility (share config file)
    - Type checking and validation

    Default values follow the original Transformer paper where applicable,
    with reduced model size for educational/computational efficiency.
    """

    # ========== Data Pipeline ==========
    data_dir: Path = Path("data")
    dataset_url: str = (
        "https://github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en"
    )

    # Vocabulary - SentencePiece BPE
    vocab_size: int = 8000
    vocab_size_ar: int = 8000  # Arabic
    vocab_size_en: int = 8000  # English

    # Special tokens (SentencePiece reserves 0-3)
    PAD_ID: int = 0  # Padding - ignored in loss
    SOS_ID: int = 1  # Start of sequence
    EOS_ID: int = 2  # End of sequence
    UNK_ID: int = 3  # Unknown token

    # Sequence limits
    max_len: int = 50
    max_len_src: int = 50  # Arabic source
    max_len_tgt: int = 50  # English target

    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # ========== Model Architecture ==========
    # Embedding dimension - paper uses 512, reduced for efficiency
    d_model: int = 256

    # Multi-head attention - paper uses 8
    num_heads: int = 8
    # Derived: dk = d_model / num_heads = 256 / 8 = 32 per head

    # Encoder/Decoder stacks - paper uses 6, reduced for efficiency
    num_layers: int = 3
    num_layers_enc: int = 3
    num_layers_dec: int = 3

    # Feed-forward dimension - paper uses 2048, scaled down
    d_ff: int = 512

    # Regularization
    dropout: float = 0.1

    # ========== Training ==========
    # Batch size - paper uses 4096 tokens, we use fixed batch
    batch_size: int = 64

    # Optimizer - original Transformer uses Adam with custom schedule
    lr: float = 5e-4  # Base learning rate
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9

    # Learning rate schedule - original Transformer
    warmup_steps: int = 4000
    # After warmup: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    # Training control
    epochs: int = 30
    early_stopping_patience: int = 5

    # Gradient clipping - prevents exploding gradients
    grad_clip_norm: float = 1.0

    # Teacher forcing (always use ground truth during training)
    teacher_forcing_ratio: float = 1.0

    # ========== Evaluation ==========
    # BLEU thresholds
    bleu_tokenized: bool = True
    bleu_case_sensitive: bool = True

    # LLM judge settings
    llm_judge_samples: int = 50
    llm_judge_model: str = "gpt-4o-mini"

    # ========== Paths ==========
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_best: str = "best.pt"

    model_dir: Path = Path("models")
    spm_ar_path: str = "models/spm_ar.model"
    spm_en_path: str = "models/spm_en.model"

    # ========== Device ==========
    device: str = "cuda"  # Will be set dynamically

    # ========== Reproducibility ==========
    seed: int = 42

    # ========== Logging ==========
    use_wandb: bool = True
    project_name: str = "nmt-transformer-ar-en"
    run_name: str = "experiment"


# Global config instance for easy imports
config = Config()

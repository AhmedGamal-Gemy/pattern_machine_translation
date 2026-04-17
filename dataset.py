"""
Dataset module for Arabic→English NMT.

Features:
1. Arabic text cleaning (diacritics removal, alef normalization)
2. SentencePiece BPE tokenization (separate models for Arabic & English)
3. PyTorch Dataset with train/val/test splits
4. Dynamic padding via collate_fn

Design rationale:
- Separate source/target tokenizers allows optimal subword segmentation per language
- Dynamic padding minimizes compute waste in attention (padded positions masked out anyway)
- Cleaning is CRITICAL for Arabic: diacritics (tashkeel) add noise not present in modern text
"""

import re
import random
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from config import config


# ========== Arabic Text Cleaning ==========


def clean_arabic(text: str) -> str:
    """
    Clean Arabic text for NMT.

    Why cleaning matters:
    - Diacritics (tashkeel): vowel marks that aid reading but add noise to translation
    - Alef variants: أ, إ, آ all normalize to ا for consistency
    - Tatweel (kashida): elongation character adds no meaning

    Operations performed (in order):
    1. Remove Arabic diacritics (tashkeel marks)
    2. Normalize alef variants → ا
    3. Remove tatweel (elongation)
    4. Remove punctuation (keep Arabic/English chars only)
    5. Normalize whitespace

    Reference: Arabic NLP standard practices
    """

    # Step 1: Remove Arabic diacritics (tashkeel)
    # These are combining marks in Unicode ranges: U+064B-065F, U+0670
    diacritics_pattern = re.compile(r"[\u064B-\u065F\u0670]")
    text = diacritics_pattern.sub("", text)

    # Step 2: Normalize alef variants → ا (alef)
    # أ (alef with hamza below), إ (alef with hamza above), آ (alef with madda)
    # all map to ا (alef) for consistent vocabulary
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    # Step 3: Remove tatweel (kashida) — horizontal elongation character
    # Serves only as visual formatter, carries no semantic weight
    text = text.replace("\u0640", "")  # كashima (tatweel)

    # Step 4: Remove punctuation, keep only Arabic letters and basic punctuation
    # Keep: Arabic chars (\u0600-\u06FF), English alphanumerics, spaces
    # This removes: Arabic punctuation (،؟！), symbols, emoji
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z\s\.\,\!\?\'\"\-\:]", "", text)

    # Step 5: Normalize whitespace (collapse multiple spaces, trim)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_english(text: str) -> str:
    """
    Clean English text for NMT.

    Simpler than Arabic:
    - Lowercase
    - Remove punctuation (keep alphanumeric, spaces)
    - Normalize whitespace
    """

    # Lowercase
    text = text.lower()

    # Remove punctuation except spaces, basic punctuation
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\'\"\-\:]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ========== Data Loading ==========


def download_data(force: bool = False) -> Path:
    """
    Download parallel corpus from GitHub.

    Source: https://github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en
    Format: JSON with 'ar' and 'en' fields

    Design rationale: Download once, reuse. Check existence before re-downloading.
    """
    data_dir = config.data_dir
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / "parallel_corpus.json"

    if data_file.exists() and not force:
        print(f"Data already exists at {data_file}")
        return data_file

    # Download from GitHub raw URL
    # Note: This assumes the repo has a specific format
    # If unavailable, creates sample data for development
    import urllib.request

    urls = [
        "https://raw.githubusercontent.com/SamirMoustafa/nmt-with-attention-for-ar-to-en/main/data.json",
    ]

    for url in urls:
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, data_file)
            print(f"Downloaded to {data_file}")
            return data_file
        except Exception as e:
            print(f"Failed to download: {e}")

    # If download fails, create sample data
    print("Creating sample data for development...")
    sample_data = [
        {"ar": "مرحبا", "en": "hello"},
        {"ar": "كيف حالك", "en": "how are you"},
        {"ar": "أنا جيد", "en": "i am good"},
        {"ar": "شكرا لك", "en": "thank you"},
        {"ar": "مع السلامة", "en": "goodbye"},
    ]

    # Create more development samples (expand for training)
    ar_phrases = [
        "مرحبا",
        "كيف حالك",
        "أنا جيد",
        "شكرا لك",
        "مع السلامة",
        "ما اسمك",
        "أنا اسمي",
        "من أين أنت",
        "أنا من",
        "ما هذا",
        "أحب هذا",
        "لا أفهم",
        "أعتذر",
        "نعم",
        "لا",
        "صباح الخير",
        "مساء الخير",
        "ليلة سعيدة",
        "أهلاً",
        "يا صديقي",
    ]
    en_phrases = [
        "hello",
        "how are you",
        "i am good",
        "thank you",
        "goodbye",
        "what is your name",
        "my name is",
        "where are you from",
        "i am from",
        "what is this",
        "i like this",
        "i do not understand",
        "i am sorry",
        "yes",
        "no",
        "good morning",
        "good evening",
        "good night",
        "hello my friend",
        "oh my friend",
    ]

    # Generate more parallel pairs
    sample_data = []
    for ar, en in zip(ar_phrases, en_phrases):
        sample_data.append({"ar": ar, "en": en})

    # Expand to create more training examples
    for i in range(100):
        idx = i % len(ar_phrases)
        sample_data.append({"ar": ar_phrases[idx], "en": en_phrases[idx]})

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"Created {len(sample_data)} parallel samples at {data_file}")
    return data_file


def load_parallel_data(data_path: Optional[Path] = None) -> list[dict]:
    """
    Load parallel Arabic-English corpus from JSON file.

    Returns: List of dicts with 'ar' (Arabic) and 'en' (English) fields
    """
    if data_path is None:
        data_path = download_data()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ========== SentencePiece Training ==========


def train_sentencepiece(
    texts: list[str], lang: str, vocab_size: int = 8000
) -> spm.SentencePieceProcessor:
    """
    Train SentencePiece BPE model on texts.

    Why SentencePiece:
    - Handles unknown characters gracefully (subword segmentation)
    - No pre-tokenization needed (handles raw text)
    - Reversible: can reconstruct original from pieces

    Design rationale:
    - Separate models for Arabic vs English (different scripts, different optimal segmentations)
    - Vocabulary size 8000 is sweet spot: enough granularity, not too many pieces

    Args:
        texts: List of text strings to train on
        lang: Language code ('ar' or 'en')
        vocab_size: Target vocabulary size

    Returns:
        Trained SentencePieceProcessor
    """

    # Create temporary text file for SentencePiece training
    # SentencePiece requires input as text file (one sentence per line)
    temp_file = config.data_dir / f"temp_{lang}.txt"

    with open(temp_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    # Train SentencePiece model
    # --input: input text file
    # --model_prefix: output model path (no extension)
    # --vocab_size: vocabulary size
    # --character_coverage: fraction of chars to cover (0.9995 for diverse scripts)
    # --model_type: bpe (byte-pair encoding)
    # --pad_id: padding token ID
    # --unk_id: unknown token ID
    # --bos_id: beginning-of-sentence token ID
    # --eos_id: end-of-sentence token ID

    model_prefix = config.model_dir / f"spm_{lang}"
    config.model_dir.mkdir(exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(temp_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.SOS_ID,
        eos_id=config.EOS_ID,
    )

    # Load and return the trained model
    model_path = f"{model_prefix}.model"
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    # Clean up temp file
    temp_file.unlink()

    print(f"Trained SentencePiece model: {model_path} (vocab_size={vocab_size})")
    return sp


def load_sentencepiece(model_path: str) -> spm.SentencePieceProcessor:
    """
    Load pre-trained SentencePiece model.

    Args:
        model_path: Path to .model file

    Returns:
        Loaded SentencePieceProcessor
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


# ========== PyTorch Dataset ==========


class ArabicEnglishDataset(Dataset):
    """
    PyTorch Dataset for Arabic→English NMT.

    Design rationale:
    - Inherits from torch.utils.data.Dataset for standard DataLoader compatibility
    - Stores token IDs directly (not raw text) for efficiency
    - Handles split logic internally

    Returns:
        Dictionary with 'src_ids' (Arabic token IDs) and 'tgt_ids' (English token IDs)
    """

    def __init__(
        self,
        data: list[dict],
        sp_ar: spm.SentencePieceProcessor,
        sp_en: spm.SentencePieceProcessor,
        split: str = "train",
        max_len_src: int = None,
        max_len_tgt: int = None,
    ):
        """
        Initialize dataset.

        Args:
            data: List of dicts with 'ar' and 'en' keys
            sp_ar: Arabic SentencePiece processor
            sp_en: English SentencePiece processor
            split: 'train', 'val', or 'test'
            max_len_src: Max source sequence length (None uses config value)
            max_len_tgt: Max target sequence length (None uses config value)
        """
        self.sp_ar = sp_ar
        self.sp_en = sp_en
        self.split = split

        # Use config defaults if not specified
        self.max_len_src = max_len_src or config.max_len_src
        self.max_len_tgt = max_len_tgt or config.max_len_tgt

        # Clean and tokenize all texts, filter by max length
        self.samples = []

        for item in data:
            # Clean texts
            ar_clean = clean_arabic(item["ar"])
            en_clean = clean_english(item["en"])

            # Tokenize (returns IDs, not pieces)
            ar_ids = sp_ar.encode(ar_clean)
            en_ids = sp_en.encode(en_clean)

            # Add BOS/EOS tokens for target (standard NMT practice)
            # Encoder gets source; Decoder gets: BOS + target + EOS
            en_ids_with_bos_eos = [config.SOS_ID] + en_ids + [config.EOS_ID]

            # Filter by max length (drop sequences that are too long)
            if (
                len(ar_ids) <= self.max_len_src
                and len(en_ids_with_bos_eos) <= self.max_len_tgt
            ):
                self.samples.append(
                    {
                        "src": ar_ids,
                        "tgt": en_ids_with_bos_eos,
                    }
                )

        print(f"Dataset {split}: {len(self.samples)} samples (filtered by max_len)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            dict with 'src_ids' and 'tgt_ids' as tensors
        """
        sample = self.samples[idx]

        return {
            "src_ids": torch.tensor(sample["src"], dtype=torch.long),
            "tgt_ids": torch.tensor(sample["tgt"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader.

    Why dynamic padding:
    - Standard padding pads ALL sequences to max batch length (wasteful)
    - Dynamic padding pads each to its own max, minimizing compute
    - Attention mask handles the variable lengths

    This function:
    1. Finds max length in current batch (not global max)
    2. Pads sequences to that length (not longer)
    3. Returns stacked tensors + actual lengths (for mask computation)

    Args:
        batch: List of dicts from ArabicEnglishDataset.__getitem__

    Returns:
        dict with:
        - src_ids: [batch, src_len] padded
        - tgt_ids: [batch, tgt_len] padded
        - src_len: actual source lengths
        - tgt_len: actual target lengths
    """

    # Extract individual samples
    src_batch = [item["src_ids"] for item in batch]
    tgt_batch = [item["tgt_ids"] for item in batch]

    # Find max lengths in this batch (dynamic)
    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_batch)

    # Pad sequences to batch max
    # PyTorch pad_sequence: pad to max length, batch_first=True
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=config.PAD_ID
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=config.PAD_ID
    )

    # Store actual lengths (useful for mask generation)
    src_lens = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_batch], dtype=torch.long)

    return {
        "src_ids": src_padded,  # [batch, max_src_len]
        "tgt_ids": tgt_padded,  # [batch, max_tgt_len]
        "src_lens": src_lens,  # [batch]
        "tgt_lens": tgt_lens,  # [batch]
    }


def create_dataloaders(
    sp_ar: spm.SentencePieceProcessor,
    sp_en: spm.SentencePieceProcessor,
    data: list[dict] = None,
    batch_size: int = None,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        sp_ar: Arabic SentencePiece processor
        sp_en: English SentencePiece processor
        data: Optional parallel data (loads if None)
        batch_size: Batch size (uses config default if None)

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """

    if data is None:
        data = load_parallel_data()

    if batch_size is None:
        batch_size = config.batch_size

    # Shuffle and split data
    random.seed(config.seed)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Create datasets
    train_dataset = ArabicEnglishDataset(train_data, sp_ar, sp_en, split="train")
    val_dataset = ArabicEnglishDataset(val_data, sp_ar, sp_en, split="val")
    test_dataset = ArabicEnglishDataset(test_data, sp_ar, sp_en, split="test")

    # Create dataloaders with collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


# ========== Main ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download data")
    args = parser.parse_args()

    if args.download:
        download_data(force=True)

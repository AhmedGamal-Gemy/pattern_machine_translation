# Neural Machine Translation: English → Arabic

## Overview

This project demonstrates English→Arabic neural machine translation using two approaches:

1. **From-Scratch Transformer** - Complete implementation of Vaswani et al. (2017)
2. **Helsinki-NLP Fine-tuned** - Pre-trained MarianMT model fine-tuned on the same dataset

The project compares these approaches to show the critical importance of pre-training for low-resource NMT.

## Results

| Model | Test BLEU | Parameters | Training Data |
|-------|-----------|------------|--------------|
| From-Scratch | ~3-5 | 10.1M | 8,593 pairs |
| Helsinki Zero-shot | 39.71 | 74M | 60M+ pairs (pre-trained) |
| Helsinki Fine-tuned | 50.92 | 74M | 60M+ → 8,593 pairs |

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
- **Parameters**: ~74M
- **Vocabulary**: 60k+ (BPE)
- **Layers**: 6 encoder + 6 decoder
- **d_model**: 512
- **Attention heads**: 8
- **Fine-tuning**: 10 epochs, LR 2e-5, batch 32, linear warmup

**Why MarianMT**:
- Pre-trained on 60M+ Arabic-English sentence pairs
- Strong zero-shot transfer to new domains
- Light fine-tuning on limited data (8.5K pairs) yields 50+ BLEU

## Key Finding

**Dataset size is the critical bottleneck**:

1. **Best Model: Helsinki Zero-shot**: Without any fine-tuning, Helsinki achieves **39.71 BLEU** - the best result. The pre-trained model already knows Arabic→English translation from 60M+ pairs.

2. **Fine-tuning Hurt**: When fine-tuned on this small dataset (8,593 pairs), the model degraded. Fine-tuning on insufficient data caused the model to overfit to the limited patterns, resulting in worse generalization.

3. **From-Scratch = Near Hallucination**: With only 8,593 pairs, the from-scratch Transformer (~10M parameters) cannot learn real translation. It essentially memorizes training data and produces near-random output (BLEU ~3-5), essentially hallucinating translations.

4. **The Lesson**: For low-resource NMT (<10K pairs), never train from scratch. Use pre-trained models as-is. Fine-tuning on small data hurts more than helps.

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
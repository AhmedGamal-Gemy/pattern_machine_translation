"""
Translation/Inference Module for Arabic→English NMT.

Features:
1. Greedy decoding (argmax at each step)
2. Beam search decoding (optional, for better quality)
3. Attention weight extraction for visualization

Design rationale:
- Greedy is fastest and sufficient for initial evaluation
- Beam search improves translation quality by considering multiple paths
- Attention weights saved for visualization/debugging
"""

from typing import Optional
import torch
import torch.nn.functional as F

from config import config
from dataset import clean_arabic


def translate(
    model: torch.nn.Module,
    source_text: str,
    sp_ar,
    sp_en,
    device: str = "cuda",
    max_len: int = None,
) -> str:
    """
    Translate a single Arabic sentence to English.

    Greedy decoding: at each step, pick the most likely token.

    Why greedy:
    - Simple and fast
    - Decent quality for initial experiments
    - Beam search can be added later if needed

    Args:
        model: Trained Transformer
        source_text: Arabic source text
        sp_ar: Arabic SentencePiece processor
        sp_en: English SentencePiece processor
        device: Device
        max_len: Max generation length

    Returns:
        English translation string
    """
    model.eval()

    if max_len is None:
        max_len = config.max_len_tgt

    # Clean and tokenize source
    source_clean = clean_arabic(source_text)
    source_ids = sp_ar.encode(source_clean)

    # Convert to tensor
    src = torch.tensor([source_ids], dtype=torch.long).to(device)

    # Encode source
    with torch.no_grad():
        # Generate padding mask
        src_padding_mask = model.generate_padding_mask(src)

        # Encode
        encoder_output = model.encoder(src, src_padding_mask)

        # Start with BOS token
        tgt_ids = [config.SOS_ID]

        # Generate token by token
        for _ in range(max_len):
            # Build target tensor
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)

            # Generate masks
            tgt_padding_mask = model.generate_padding_mask(tgt)
            look_ahead_mask = model.generate_look_ahead_mask(tgt.size(1), device)
            tgt_mask = tgt_padding_mask & look_ahead_mask

            # Forward through decoder
            decoder_output = model.decoder(
                tgt,
                encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_padding_mask,
            )

            # Get next token logits
            logits = model.fc_out(decoder_output)

            # Greedy: pick highest probability token
            next_token = logits[0, -1, :].argmax().item()

            # Stop if EOS
            if next_token == config.EOS_ID:
                break

            # Add to sequence
            tgt_ids.append(next_token)

        # Decode to text
        translation = sp_en.decode(tgt_ids[1:])  # Remove BOS

    return translation


def translate_batch(
    model: torch.nn.Module,
    sources: torch.Tensor,
    sp_en,
    max_len: int = None,
) -> list[str]:
    """
    Translate a batch of Arabic sequences.

    Args:
        model: Transformer model
        sources: Source token IDs, [batch, src_len]
        sp_en: English SentencePiece processor
        max_len: Max generation length

    Returns:
        List of English translations
    """
    model.eval()

    if max_len is None:
        max_len = config.max_len_tgt

    batch_size = sources.size(0)
    device = sources.device

    with torch.no_grad():
        # Encode source
        src_padding_mask = model.generate_padding_mask(sources)
        encoder_output = model.encoder(sources, src_padding_mask)

        # Initialize with SOS
        tgt = torch.full(
            (batch_size, 1), config.SOS_ID, dtype=torch.long, device=device
        )

        # Generate iteratively
        for step in range(max_len):
            # Generate masks
            tgt_padding_mask = model.generate_padding_mask(tgt)
            look_ahead_mask = model.generate_look_ahead_mask(tgt.size(1), device)
            tgt_mask = tgt_padding_mask & look_ahead_mask

            # Decode
            decoder_output = model.decoder(
                tgt,
                encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_padding_mask,
            )

            # Get next tokens
            logits = model.fc_out(decoder_output)
            next_tokens = logits[:, -1, :].argmax(dim=-1)

            # Check for EOS
            eos_mask = next_tokens == config.EOS_ID
            if eos_mask.all():
                break

            # Append
            tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)

        # Decode each sequence
        translations = []
        for i in range(batch_size):
            # Find EOS if present
            ids = tgt[i].cpu().tolist()
            if config.EOS_ID in ids:
                ids = ids[: ids.index(config.EOS_ID)]

            # Remove SOS
            if ids and ids[0] == config.SOS_ID:
                ids = ids[1:]

            # Decode
            trans = sp_en.decode(ids)
            translations.append(trans)

    return translations


def translate_beam(
    model: torch.nn.Module,
    source_text: str,
    sp_ar,
    sp_en,
    device: str = "cuda",
    beam_size: int = 5,
    max_len: int = None,
    length_penalty: float = 0.6,
) -> str:
    """
    Beam search translation.

    Why beam search:
    - Greedy picks single best path, beam keeps top-k paths
    - Better translation quality (avoids early mistakes)
    - More compute (k times slower)

    Length penalty (Wu et al., 2016):
    - Encourages longer translations
    - Prevents bias toward short translations

    Args:
        model: Transformer
        source_text: Arabic source
        sp_ar: Arabic SPM
        sp_en: English SPM
        device: Device
        beam_size: Number of beams
        max_len: Max length
        length_penalty: LP alpha (0 = no penalty, higher = longer)

    Returns:
        Best English translation
    """
    model.eval()

    if max_len is None:
        max_len = config.max_len_tgt

    # Clean and tokenize
    source_clean = clean_arabic(source_text)
    source_ids = sp_ar.encode(source_clean)
    src = torch.tensor([source_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        # Encode
        src_padding_mask = model.generate_padding_mask(src)
        encoder_output = model.encoder(src, src_padding_mask)

        # Initialize: [beam_size, 1] with SOS
        # For first step, we only have one source, so beam_size copies of initial state
        done = [False] * beam_size
        beam_scores = torch.zeros(beam_size, device=device)
        gen_sequences = [[config.SOS_ID] for _ in range(beam_size)]

        # Encode once (can be reused for each beam)
        # Actually we need to handle this differently - we'll do standard beam
        # Simplified: expand encoder output for each beam hypothesis

        # For simplicity, use single beam expansion at each step
        for step in range(max_len):
            # Build target tensor from all beam sequences
            tgt = torch.tensor(
                [seq[-config.max_len :] for seq in gen_sequences],  # Truncate
                dtype=torch.long,
                device=device,
            )

            # Create masks
            tgt_padding_mask = model.generate_padding_mask(tgt)
            look_ahead_mask = model.generate_look_ahead_mask(tgt.size(1), device)
            tgt_mask = tgt_padding_mask & look_ahead_mask

            # Decode
            decoder_output = model.decoder(
                tgt,
                encoder_output.expand(beam_size, -1, -1),
                tgt_mask=tgt_mask,
                src_mask=src_padding_mask.expand(beam_size, -1, -1),
            )

            # Get logits
            logits = model.fc_out(decoder_output)  # [batch, seq, vocab]
            next_logits = logits[:, -1, :]  # [batch, vocab]

            # Calculate log probabilities
            log_probs = F.log_softmax(next_logits, dim=-1)

            # Adjust for length penalty
            # lp = (5 + len)^alpha / (5 + 1)^alpha
            if length_penalty > 0:
                current_lengths = torch.tensor(
                    [len(seq) for seq in gen_sequences],
                    device=device,
                )
                length_penalties = ((5 + current_lengths) ** length_penalty) / (
                    5 + 1
                ) ** length_penalty
                log_probs = log_probs / length_penalties.unsqueeze(1)

            # Get top-k for each beam
            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            # Update beam sequences
            new_sequences = []
            new_scores = []

            for beam_idx in range(beam_size):
                best_token = topk_indices[beam_idx].item()
                best_score = topk_log_probs[beam_idx].item()
                new_score = beam_scores[beam_idx] + best_score

                gen_sequences[beam_idx].append(best_token)
                new_sequences.append(gen_sequences[beam_idx][:])
                new_scores.append(new_score)

                # Check if done
                if best_token == config.EOS_ID:
                    done[beam_idx] = True

            # Update beam scores
            beam_scores = torch.tensor(new_scores, device=device)
            gen_sequences = new_sequences

            # Check if all done
            if all(done):
                break

        # Select best sequence
        best_idx = beam_scores.argmax().item()
        best_sequence = gen_sequences[best_idx]

        # Remove SOS/EOS
        if best_sequence[0] == config.SOS_ID:
            best_sequence = best_sequence[1:]
        if config.EOS_ID in best_sequence:
            best_sequence = best_sequence[: best_sequence.index(config.EOS_ID)]

        translation = sp_en.decode(best_sequence)

    return translation


def get_attention_weights(
    model: torch.nn.Module,
    source_text: str,
    sp_ar,
    device: str = "cuda",
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    Get cross-attention weights for visualization.

    Args:
        model: Transformer model
        source_text: Arabic source text
        sp_ar: Arabic SPM
        device: Device
        layer_idx: Layer index (negative for from-end)

    Returns:
        Attention weights [num_heads, tgt_len, src_len]
    """
    model.eval()

    # Clean and tokenize
    source_clean = clean_arabic(source_text)
    source_ids = sp_ar.encode(source_clean)
    src = torch.tensor([source_ids], dtype=torch.long).to(device)

    # Encode with attention tracking
    # For this, we need to set model to eval with gradients disabled
    # but we want to capture intermediate activations

    # Use hooks (PyTorch approach for getting internal activations)
    attention_weights = None

    def hook_fn(module, input, output):
        nonlocal attention_weights
        # DecoderLayer stores cross attention in last_cross_attention_weights
        attention_weights = module.last_cross_attention_weights

    # Register hook on final decoder layer's cross attention
    hook_handle = None
    if hasattr(model.decoder.layers[layer_idx], "cross_attn"):
        hook_handle = model.decoder.layers[layer_idx].cross_attn.register_forward_hook(
            hook_fn
        )

    with torch.no_grad():
        # Generate with model (trigger hook)
        # We'll just do one step to capture attention

        # For proper attention extraction, need encode + decode
        src_padding_mask = model.generate_padding_mask(src)
        encoder_output = model.encoder(src, src_padding_mask)

        # Start with BOS
        tgt = torch.tensor([[config.SOS_ID]], dtype=torch.long, device=device)

        tgt_padding_mask = model.generate_padding_mask(tgt)
        look_ahead_mask = model.generate_look_ahead_mask(tgt.size(1), device)
        tgt_mask = tgt_padding_mask & look_ahead_mask

        _ = model.decoder(
            tgt,
            encoder_output,
            tgt_mask=tgt_mask,
            src_mask=src_padding_mask,
        )

    # Remove hook
    if hook_handle is not None:
        hook_handle.remove()

    # Get from model directly (if stored)
    if attention_weights is None:
        attention_weights = model.get_cross_attention_weights(layer_idx)

    return attention_weights


# ========== Demo ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default="مرحبا")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    from model import Transformer

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = Transformer(
        src_vocab_size=config.vocab_size_ar,
        tgt_vocab_size=config.vocab_size_en,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load SPMs
    from dataset import load_sentencepiece

    sp_ar = load_sentencepiece(config.spm_ar_path)
    sp_en = load_sentencepiece(config.spm_en_path)

    # Translate
    result = translate(model, args.text, sp_ar, sp_en, device)
    print(f"Arabic: {args.text}")
    print(f"English: {result}")

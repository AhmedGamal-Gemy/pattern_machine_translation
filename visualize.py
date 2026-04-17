"""
Attention Visualization for Arabic→English NMT.

Features:
1. Cross-attention heatmaps (decoder → encoder)
2. Self-attention heatmaps (within encoder/decoder)
3. Multi-head visualization (show all heads separately)
4. Aggregated visualization (average across heads)

Design rationale:
- Visualizing attention helps understand how model aligns languages
- Arabic→English: attention often shows monotonic alignment
- Different heads may capture different linguistic phenomena
- Useful for debugging and model analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from config import config


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    title: str = "Cross-Attention",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot attention heatmap.

    Args:
        attention_weights: [num_heads, tgt_len, src_len]
        source_tokens: List of source tokens (Arabic)
        target_tokens: List of target tokens (English)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        weights = attention_weights.cpu().numpy()
    else:
        weights = attention_weights

    # Get dimensions
    num_heads, tgt_len, src_len = weights.shape

    # Select a specific head (first head often shows clearest alignment)
    head_idx = 0
    attn = weights[head_idx]  # [tgt_len, src_len]

    # Trim to actual lengths
    attn = attn[: len(target_tokens), : len(source_tokens)]

    # Create figure
    plt.figure(figsize=figsize)

    # Plot heatmap
    # Rows = target (English), Columns = source (Arabic)
    sns.heatmap(
        attn,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap="YlOrRd",
        cbar_kws={"label": "Attention Weight"},
        annot=False,
    )

    plt.title(f"{title} (Head {head_idx})")
    plt.xlabel("Source (Arabic)")
    plt.ylabel("Target (English)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close()


def plot_multihead_attention(
    attention_weights: torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    num_heads: int = 8,
    title: str = "Multi-Head Attention",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot all attention heads in a grid.

    Why show all heads:
    - Different heads attend to different aspects
    - Some may focus on syntax, others on semantics
    - Aggregating shows overall pattern

    Args:
        attention_weights: [num_heads, tgt_len, src_len]
        source_tokens: Source tokens
        target_tokens: Target tokens
        num_heads: Number of heads (for grid rows)
        title: Plot title
        save_path: Save path
    """
    weights = attention_weights.cpu().numpy()

    # Create grid
    rows = (num_heads + 3) // 4  # 4columns
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 4))
    axes = axes.flatten() if num_heads > 4 else [axes]

    for h in range(num_heads):
        ax = axes[h]

        # Get head weights
        attn = weights[h, : len(target_tokens), : len(source_tokens)]

        # Plot
        sns.heatmap(
            attn,
            xticklabels=False,
            yticklabels=False,
            cmap="YlOrRd",
            cbar=False,
            ax=ax,
        )

        ax.set_title(f"Head {h}")

    # Hide unused subplots
    for h in range(num_heads, len(axes)):
        axes[h].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close()


def plot_attention_aggregate(
    attention_weights: torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    title: str = "Aggregated Attention",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot attention aggregated across all heads.

    Averaging across heads shows overall alignment pattern
    without noise from individual heads.

    Args:
        attention_weights: [num_heads, tgt_len, src_len]
        source_tokens: Source tokens
        target_tokens: Target tokens
        title: Title
        save_path: Save path
    """
    weights = attention_weights.cpu().numpy()

    # Average across heads
    attn_avg = weights.mean(axis=0)  # [tgt_len, src_len]

    # Trim
    attn_avg = attn_avg[: len(target_tokens), : len(source_tokens)]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        attn_avg,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap="YlOrRd",
        cbar_kws={"label": "Avg Attention"},
    )

    plt.title(title)
    plt.xlabel("Source (Arabic)")
    plt.ylabel("Target (English)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close()


def visualize_translation_attention(
    model: torch.nn.Module,
    source_text: str,
    translation: str,
    sp_ar,
    sp_en,
    device: str = "cuda",
    layer_idx: int = -1,
    save_dir: Path = None,
) -> dict:
    """
    Visualize attention for a translation example.

    Args:
        model: Transformer model
        source_text: Arabic source text
        translation: English translation
        sp_ar: Arabic SPM
        sp_en: English SPM
        device: Device
        layer_idx: Layer to visualize
        save_dir: Directory to save figures

    Returns:
        dict with paths to saved figures
    """
    from translate import get_attention_weights
    from dataset import clean_arabic

    # Get tokens (for visualization labels)
    source_clean = clean_arabic(source_text)
    source_tokens = sp_ar.encode(source_clean)
    source_pieces = [sp_ar.id_to_piece(t) for t in source_tokens]

    # Translation tokens (already generated, need to encode)
    # For now, decode the translation to get tokens
    # This is a simplification - ideally we track tokens during generation
    tgt_ids = sp_en.encode(translation)
    target_pieces = [sp_en.id_to_piece(t) for t in tgt_ids]

    # Get attention weights
    attn_weights = get_attention_weights(model, source_text, sp_ar, device, layer_idx)

    if attn_weights is None:
        print("Could not extract attention weights")
        return {}

    # Save figures
    save_paths = {}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Single head heatmap
        save_paths["single_head"] = save_dir / "attention_single.png"
        plot_attention_heatmap(
            attn_weights,
            source_pieces,
            target_pieces,
            title=f"Cross-Attention (Layer {layer_idx})",
            save_path=save_paths["single_head"],
        )

        # Multi-head grid
        save_paths["multi_head"] = save_dir / "attention_multihead.png"
        plot_multihead_attention(
            attn_weights,
            source_pieces,
            target_pieces,
            num_heads=config.num_heads,
            title=f"All Heads (Layer {layer_idx})",
            save_path=save_paths["multi_head"],
        )

        # Aggregated
        save_paths["aggregated"] = save_dir / "attention_aggregated.png"
        plot_attention_aggregate(
            attn_weights,
            source_pieces,
            target_pieces,
            title=f"Aggregated Across Heads (Layer {layer_idx})",
            save_path=save_paths["aggregated"],
        )

    return {
        "source_text": source_text,
        "translation": translation,
        "source_tokens": source_pieces,
        "target_tokens": target_pieces,
        "attention_weights": attn_weights,
        "save_paths": save_paths,
    }


def create_attention_gallery(
    model: torch.nn.Module,
    examples: list[dict],
    sp_ar,
    sp_en,
    device: str = "cuda",
    output_dir: Path = None,
) -> list[dict]:
    """
    Create attention visualizations for multiple examples.

    Args:
        model: Transformer model
        examples: List of {'ar': str, 'en': str} examples
        sp_ar: Arabic SPM
        sp_en: English SPM
        device: Device
        output_dir: Output directory

    Returns:
        List of results dicts
    """
    from translate import translate

    results = []

    for i, example in enumerate(examples):
        print(f"Processing example {i + 1}/{len(examples)}: {example['ar']}")

        # Translate
        translation = translate(
            model,
            example["ar"],
            sp_ar,
            sp_en,
            device,
        )

        # Visualize
        save_dir = output_dir / f"example_{i}" if output_dir else None
        vis_result = visualize_translation_attention(
            model,
            example["ar"],
            translation,
            sp_ar,
            sp_en,
            device,
            save_dir=save_dir,
        )

        results.append(
            {
                "index": i,
                "source": example["ar"],
                "reference": example["en"],
                "translation": translation,
                **vis_result,
            }
        )

    return results


# ========== Entry Point ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--examples", type=str, nargs="+", default=["مرحبا", "كيف حالك"]
    )
    parser.add_argument("--output-dir", type=str, default="attention_viz")
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

    # Create examples
    examples = [{"ar": ex, "en": ""} for ex in args.examples]

    # Create gallery (won't have reference, translate gets it)
    from translate import translate

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results = []

    for i, ex in enumerate(examples):
        print(f"\n=== Example {i + 1} ===")

        # Translate
        translation = translate(model, ex["ar"], sp_ar, sp_en, device)
        print(f"Arabic: {ex['ar']}")
        print(f"English: {translation}")

        # Visualize
        vis_result = visualize_translation_attention(
            model,
            ex["ar"],
            translation,
            sp_ar,
            sp_en,
            device,
            save_dir=output_dir / f"ex_{i}",
        )

        results.append(
            {
                "source": ex["ar"],
                "translation": translation,
                "paths": vis_result.get("save_paths", {}),
            }
        )

    print(f"\nSaved visualizations to {output_dir}")

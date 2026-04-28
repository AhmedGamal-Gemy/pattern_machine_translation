"""
Streamlit App for Arabic→English NMT Demo.

Features:
1. Model selector (Helsinki fine-tuned vs From-Scratch Transformer)
2. Translation with both models
3. Model comparison side-by-side
4. About/Info tab

Usage:
    uv run streamlit run app.py
"""

import streamlit as st
import torch
import time
from pathlib import Path
from typing import Optional, Tuple

# ========== Page Config ==========
st.set_page_config(
    page_title="Arabic→English NMT",
    page_icon="🈚",
    layout="wide",
)


# ========== Model Loading ==========


@st.cache_resource
def load_from_scratch_model() -> Tuple[
    Optional[any], Optional[any], Optional[any], str
]:
    """
    Load from-scratch Transformer model.

    Returns:
        Tuple of (model, sp_ar, sp_en, device) or (None, None, None, device) on failure
    """
    from config import config
    from model import Transformer
    from dataset import load_sentencepiece

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if checkpoint exists
    checkpoint_path = Path("checkpoints/best.pt")
    sp_ar_path = Path("models/spm_ar.model")
    sp_en_path = Path("models/spm_en.model")

    if not checkpoint_path.exists():
        return None, None, None, device
    if not sp_ar_path.exists() or not sp_en_path.exists():
        return None, None, None, device

    try:
        # Load SentencePiece
        sp_ar = load_sentencepiece(str(sp_ar_path))
        sp_en = load_sentencepiece(str(sp_en_path))

        # Load Transformer
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

        return model, sp_ar, sp_en, device
    except Exception as e:
        st.error(f"From-scratch model error: {e}")
        return None, None, None, device


@st.cache_resource
def load_helsinki_model(
    finetuned: bool = True,
) -> Tuple[Optional[any], Optional[any], str]:
    """
    Load Helsinki model - either fine-tuned or zero-shot.

    Args:
        finetuned: If True, load from helsinki_best. If False, load base pretrained model.

    Returns:
        Tuple of (model, tokenizer, device) or (None, None, device) on failure
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if finetuned:
        checkpoint_path = Path("checkpoints/helsinki_best")
        if not checkpoint_path.exists():
            return None, None, device
    else:
        # Zero-shot: use base pretrained model
        checkpoint_path = "Helsinki-NLP/opus-mt-ar-en"

    try:
        from transformers import MarianMTModel, MarianTokenizer

        model = MarianMTModel.from_pretrained(str(checkpoint_path)).to(device)
        tokenizer = MarianTokenizer.from_pretrained(str(checkpoint_path))
        model.eval()

        return model, tokenizer, device
    except Exception as e:
        st.error(f"Helsinki model error: {e}")
        return None, None, device


# ========== Translation Functions ==========


def translate_from_scratch(
    model, source_text: str, sp_ar, sp_en, device: str, max_len: int = 50
) -> str:
    """
    Translate using from-scratch Transformer with greedy decoding.

    Includes repetition detection (stop if same token 3x in a row).
    """
    from config import config
    from dataset import clean_arabic

    # Clean and tokenize source
    source_clean = clean_arabic(source_text)
    source_ids = sp_ar.encode(source_clean)
    src = torch.tensor([source_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        # Encode
        src_padding_mask = model.generate_padding_mask(src)
        encoder_output = model.encoder(src, src_padding_mask)

        # Start with BOS
        tgt_ids = [config.SOS_ID]

        # Generate token by token
        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)

            tgt_padding_mask = model.generate_padding_mask(tgt)
            look_ahead_mask = model.generate_look_ahead_mask(tgt.size(1), device)
            tgt_mask = tgt_padding_mask & look_ahead_mask

            decoder_output = model.decoder(
                tgt, encoder_output, tgt_mask=tgt_mask, src_mask=src_padding_mask
            )
            logits = model.fc_out(decoder_output)

            # Greedy: pick highest probability token
            next_token = logits[0, -1, :].argmax().item()

            # Stop if EOS
            if next_token == config.EOS_ID:
                break

            # Repetition detection - stop after 3 repeats
            if len(tgt_ids) > 1 and tgt_ids[-1] == tgt_ids[-2]:
                if tgt_ids.count(tgt_ids[-1]) > 3:
                    break

            tgt_ids.append(next_token)

        # Remove special tokens
        tgt_ids_clean = [
            t for t in tgt_ids if t not in (config.SOS_ID, config.EOS_ID, config.PAD_ID)
        ]

        # Decode
        translation = sp_en.decode(tgt_ids_clean)

    return translation


def translate_helsinki(
    model, tokenizer, source_text: str, device: str, max_len: int = 50
) -> str:
    """Translate using Helsinki model."""
    inputs = tokenizer(
        [source_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    with torch.no_grad():
        translated = model.generate(**inputs, max_new_tokens=max_len)

    return tokenizer.decode(translated[0], skip_special_tokens=True)


# ========== UI ==========

st.title("🈚 Arabic → English Neural Machine Translation")
st.markdown(
    "Compare **From-Scratch Transformer** vs **Helsinki-NLP Fine-tuned** models"
)


# ========== Sidebar ==========
with st.sidebar:
    st.header("Model Selector")
    model_choice = st.radio(
        "Choose model:",
        [
            "Helsinki (Fine-tuned) ★",
            "Helsinki (Zero-shot)",
            "From-Scratch Transformer",
            "Compare All",
        ],
    )

    st.header("Model Info")

    if "Fine-tuned" in model_choice:
        st.markdown("""
        **Helsinki-NLP (Fine-tuned)**
        
        - Architecture: MarianMT (Transformer)
        - Parameters: ~74M
        - Layers: 6 encoder + 6 decoder
        - d_model: 512, heads: 8
        - Vocab: 60k+ (BPE)
        - BLEU: 50.92 (test)
        - Pre-trained on 60M+ pairs → fine-tuned on 8,593
        """)
    elif "Zero-shot" in model_choice:
        st.markdown("""
        **Helsinki-NLP (Zero-shot)**
        
        - Architecture: MarianMT (Transformer)
        - Parameters: ~74M
        - Layers: 6 encoder + 6 decoder
        - d_model: 512, heads: 8
        - Vocab: 60k+ (BPE)
        - BLEU: 39.71 (test)
        - Direct inference, no fine-tuning
        """)
    elif model_choice == "From-Scratch Transformer":
        st.markdown("""
        **Custom Transformer (Vaswani 2017)**
        
        - Architecture: Transformer (from scratch)
        - Parameters: 10.1M
        - Layers: 3 encoder + 3 decoder
        - d_model: 256, heads: 8, d_ff: 512
        - Vocab: 8k (SentencePiece BPE)
        - BLEU: ~3-5
        """)
    else:
        st.markdown("""
        **Comparison Mode**
        
        Translate the same text with both models and compare outputs.
        """)


# ========== Main Tabs ==========
tab_translate, tab_compare, tab_viz, tab_about = st.tabs(
    ["Translation", "Model Comparison", "Visualizations", "About"]
)


# ======== Tab 1: Translation ========
with tab_translate:
    st.subheader("Translate Arabic to English")

    # Arabic input with RTL styling
    col1, col2 = st.columns([3, 1])

    with col1:
        source_text = st.text_area(
            "Enter Arabic text:",
            placeholder="أدخل النص بالعربية هنا...\nمرحبا بك",
            height=150,
            key="source_input",
        )

    with col2:
        st.write("")
        st.write("")
        translate_btn = st.button(
            "Translate ➡️", type="primary", use_container_width=True
        )

    # Example buttons
    st.markdown("**Examples:**")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    examples = [
        ("مرحبا", "hello"),
        ("كيف حالك", "how are you"),
        ("شكرا لك", "thank you"),
    ]

    for i, (ar, en) in enumerate(examples):
        with [ex_col1, ex_col2, ex_col3][i]:
            if st.button(f"{ar} → {en}", key=f"ex_{ar}"):
                source_text = ar

    # Translation
    if translate_btn and source_text:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with st.spinner("Translating..."):
            start_time = time.time()

            if "Fine-tuned" in model_choice:
                helsinki_model, helsinki_tokenizer, _ = load_helsinki_model(
                    finetuned=True
                )
                if helsinki_model:
                    translation = translate_helsinki(
                        helsinki_model, helsinki_tokenizer, source_text, device
                    )
                else:
                    st.error("Helsinki fine-tuned not found. Run finetune.py first.")
                    translation = None

            elif "Zero-shot" in model_choice:
                helsinki_model, helsinki_tokenizer, _ = load_helsinki_model(
                    finetuned=False
                )
                if helsinki_model:
                    translation = translate_helsinki(
                        helsinki_model, helsinki_tokenizer, source_text, device
                    )
                else:
                    st.error("Helsinki zero-shot failed to load.")
                    translation = None

            elif model_choice == "From-Scratch Transformer":
                scr_model, sp_ar, sp_en, _ = load_from_scratch_model()
                if scr_model:
                    translation = translate_from_scratch(
                        scr_model, source_text, sp_ar, sp_en, device
                    )
                else:
                    st.error("From-scratch model not found. Run train.py first.")
                    translation = None
            else:
                translation = None

            elapsed = (time.time() - start_time) * 1000

        if translation:
            st.success(f"Translation ({elapsed:.0f}ms):")
            st.markdown(f"### {translation}")


# ======== Tab 2: Comparison ========
with tab_compare:
    st.subheader("Compare All Models")

    compare_text = st.text_input(
        "Enter Arabic text to compare:",
        placeholder="أدخل النص بالعربية",
        key="compare_input",
    )

    # Example buttons
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button("مرحبا"):
            compare_text = "مرحبا"
    with ex_col2:
        if st.button("كيف حالك"):
            compare_text = "كيف حالك"

    compare_btn = st.button("Compare All Models", type="primary")

    if compare_btn and compare_text:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = {}

        # Helsinki Fine-tuned
        with st.spinner("Helsinki (Fine-tuned)..."):
            model, tok, _ = load_helsinki_model(finetuned=True)
            if model:
                results["Helsinki\n(Fine-tuned)"] = translate_helsinki(
                    model, tok, compare_text, device
                )

        # Helsinki Zero-shot
        with st.spinner("Helsinki (Zero-shot)..."):
            model, tok, _ = load_helsinki_model(finetuned=False)
            if model:
                results["Helsinki\n(Zero-shot)"] = translate_helsinki(
                    model, tok, compare_text, device
                )

        # From-Scratch
        with st.spinner("From-Scratch..."):
            model, sp_ar, sp_en, _ = load_from_scratch_model()
            if model:
                results["From-Scratch\nTransformer"] = translate_from_scratch(
                    model, compare_text, sp_ar, sp_en, device
                )

        # Display results
        st.markdown("---")
        st.markdown(f"**Input:** {compare_text}")
        st.markdown("---")
        for name, trans in results.items():
            st.markdown(f"**{name}:** {trans}")

        # Reference (if available)
        st.markdown("---")
        st.caption(f"Reference depends on input")


# ======== Tab 3: Visualizations ========
with tab_viz:
    st.subheader("Attention Heatmaps")

    # Check if from-scratch model exists
    scr_model, _, _, _ = load_from_scratch_model()

    if scr_model is None:
        st.error("From-scratch model not found. Run train.py first.")
    else:
        st.info("Generate attention heatmaps from the from-scratch Transformer")

        # Input text
        viz_text = st.text_input(
            "Enter Arabic text for attention visualization:",
            placeholder="مرحبا",
            key="viz_input",
        )

        # Example buttons
    if st.button("Example: مرحبا"):
        viz_text = "مرحبا"
    if st.button("Example: كيف حالك"):
        viz_text = "كيف حالك"

    generate_btn = st.button("Generate Heatmap", type="primary")

    if generate_btn and viz_text:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Import required modules
        from config import config
        from dataset import load_sentencepiece, clean_arabic
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from io import BytesIO
        import base64

        try:
            # Load tokenizer
            sp_ar = load_sentencepiece("models/spm_ar.model")
            sp_en = load_sentencepiece("models/spm_en.model")

            # Encode source
            source_clean = clean_arabic(viz_text)
            source_ids = sp_ar.encode(source_clean)
            src = torch.tensor([source_ids], dtype=torch.long).to(device)

            # First: run inference to capture attention weights
            src_padding_mask = scr_model.generate_padding_mask(src)
            encoder_output = scr_model.encoder(src, src_padding_mask)

            # Do one decode step to trigger attention
            tgt = torch.tensor([[config.SOS_ID]], dtype=torch.long).to(device)
            tgt_padding_mask = scr_model.generate_padding_mask(tgt)
            look_ahead_mask = scr_model.generate_look_ahead_mask(tgt.size(1), device)
            tgt_mask = tgt_padding_mask & look_ahead_mask

            _ = scr_model.decoder(
                tgt,
                encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_padding_mask,
            )

            # Now get the attention weights
            attn = scr_model.get_cross_attention_weights(0)

            if attn is not None:
                # attn shape: [num_heads, tgt_len, src_len]
                avg_attn = attn.mean(dim=0)[0].detach().cpu().numpy()

                # Decode tokens
                src_tokens = sp_ar.decode(source_ids).split()
                tgt_tokens = ["<s>"]

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    avg_attn,
                    cmap="viridis",
                    ax=ax,
                    xticklabels=src_tokens,
                    yticklabels=tgt_tokens,
                )

                # RTL for Arabic
                ax.xaxis.set_tick_params(labelrotation=0)
                ax.set_xlabel("Source (Arabic) - RTL", fontfamily="Arial")
                ax.set_ylabel("Target (English)")
                ax.set_title(f"Cross-Attention: {viz_text}")

                # Convert to image
                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode()
                st.image(f"data:image/png;base64,{img_str}")
                plt.close()
            else:
                st.warning(
                    "No attention weights captured. Model may not support attention extraction."
                )
        except Exception as e:
            st.error(f"Error: {e}")

    # Architecture section
    st.markdown("---")
    st.subheader("Architecture Diagrams")

    # Use expanders for better organization
    with st.expander("📐 MarianMT Architecture", expanded=True):
        if Path("diagrams/marianmt_architecture.html").exists():
            with open(
                "diagrams/marianmt_architecture.html", "r", encoding="utf-8"
            ) as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)

    with st.expander("🏗️ Transformer Architecture"):
        if Path("diagrams/transformer_architecture.html").exists():
            with open(
                "diagrams/transformer_architecture.html", "r", encoding="utf-8"
            ) as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)

    with st.expander("🔲 Encoder Layer"):
        if Path("diagrams/encoder_layer.html").exists():
            with open("diagrams/encoder_layer.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)

    with st.expander("🔄 Decoder Layer"):
        if Path("diagrams/decoder_layer.html").exists():
            with open("diagrams/decoder_layer.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)

    with st.expander("👁️ Attention Mechanism"):
        if Path("diagrams/attention_mechanism.html").exists():
            with open("diagrams/attention_mechanism.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)

    with st.expander("🔄 Training Loop"):
        if Path("diagrams/training_loop.html").exists():
            with open("diagrams/training_loop.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)

    with st.expander("🎭 Masks"):
        if Path("diagrams/masks.html").exists():
            with open("diagrams/masks.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)


# ======== Tab 4: About ========
with tab_about:
    st.subheader("About This Project")

    st.markdown("""
    ## Arabic → English Neural Machine Translation
    
    This project demonstrates two approaches to machine translation:
    
    ### 1. From-Scratch Transformer
    Complete implementation of the Transformer architecture from the ground up:
    - Encoder-Decoder architecture (Vaswani et al., 2017)
    - Multi-head attention with 8 heads
    - Sinusoidal positional encoding
    - SentencePiece BPE tokenization
    
    **Result**: BLEU ~3-5 - essentially hallucination due to tiny dataset
    
    ### 2. Helsinki Model
    Pre-trained Helsinki-NLP/opus-mt-ar-en (MarianMT):
    - Pre-trained on 60M+ Arabic-English pairs
    - Zero-shot BLEU: 39.71 (no fine-tuning!)
    - Fine-tuned BLEU: 50.92 (on 8,593 pairs)
    
    ## Key Findings
    
    1. **Best for General Use**: Helsinki Zero-shot (39.71 BLEU)
       - No fine-tuning needed, best generalization
    
    2. **Fine-tuning on Small Data Hurts Generalization**
       - Fine-tuning (50.92) beats zero-shot on TEST SET
       - BUT may generalize worse to out-of-domain data
       - Only fine-tune if you have 100K+ pairs
    
    3. **From-Scratch = Near Hallucination**
       - 8,593 pairs is too small for 10M parameters
       - Model memorizes, doesn't translate
       - Never train from scratch with <10K pairs
    """)

    # Training summary
    st.markdown("---")
    st.subheader("Training Summary")

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Train Pairs", "8,593")
    met2.metric("From-Scratch BLEU", "~3-5")
    met3.metric("Helsinki BLEU", "50.92")
    met4.metric("From-Scratch Params", "10.1M")

    # Architecture details
    st.markdown("---")
    st.subheader("Architecture Spec")

    st.markdown("""
    | Component | From-Scratch | Helsinki |
    |-----------|--------------|-----------|
    | d_model | 256 | 512 |
    | heads | 8 | 8 |
    | layers | 3 | 6 |
    | d_ff | 512 | 2048 |
    | vocab | 8000 (BPE) | 60k+ |
    """)


# ========== Other Scripts ==========
st.markdown("""
### Other Useful Scripts

- **CLI translation** (`demo.py`): `uv run python demo.py`
- **Test eval** (`zero_shot.py`): `uv run python zero_shot.py`
- **Train from-scratch**: `uv run python train.py --epochs 30`
- **Fine-tune Helsinki**: `uv run python finetune.py`
""")


# ========== Footer ==========
st.markdown("---")
st.caption(
    "Built with PyTorch • Transformer architecture • "
    "Helsinki-NLP/opus-mt-ar-en | "
    "Dataset: github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en"
)

"""
Streamlit Frontend for Arabic→English Translation Demo.

Features:
1. Simple text input
2. Translate button
3. Translation output display
4. Optional attention visualization

Usage:
    uv run streamlit run streamlit_app.py

Then open http://localhost:8501 in browser.
"""

import streamlit as st
import torch
from pathlib import Path

from config import config
from model import Transformer
from dataset import load_sentencepiece
from translate import translate as translate_fn


# ========== Page Config ==========

st.set_page_config(
    page_title="Arabic→English NMT",
    page_icon="🈚",
    layout="wide",
)


# ========== State ==========


@st.cache_resource
def load_model_and_tokenizers():
    """Load model and tokenizers (cached for performance)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SentencePiece
    sp_ar = load_sentencepiece(config.spm_ar_path)
    sp_en = load_sentencepiece(config.spm_en_path)

    # Load Transformer
    checkpoint_path = config.checkpoint_dir / config.checkpoint_best
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


# ========== UI ==========

st.title("🈚 Arabic → English NMT")
st.markdown("Neural Machine Translation using **Transformer from Scratch** (PyTorch)")

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
**Arabic→English NMT**

Built with:
- Transformer architecture (Vaswani et al., 2017)
- PyTorch (no pretrained models)
- SentencePiece BPE tokenization
- Clean, educational code
""")

st.sidebar.header("Model Info")
st.sidebar.markdown(f"""
- **d_model**: {config.d_model}
- **num_heads**: {config.num_heads}
- **num_layers**: {config.num_layers}
- **d_ff**: {config.d_ff}
- **vocab_size**: {config.vocab_size}
""")

# Try to load model
try:
    model, sp_ar, sp_en, device = load_model_and_tokenizers()
    st.success(f"Model loaded on {device}")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info("Run train.py first to train the model.")
    model = None


# Input
st.subheader("Translate")
col1, col2 = st.columns([2, 1])

with col1:
    source_text = st.text_input(
        "Arabic text to translate:",
        placeholder="أدخل النص بالعربية هنا...",
        key="source_input",
    )

with col2:
    translate_button = st.button("Translate ➡️", type="primary")


# Translation
if translate_button and source_text and model:
    with st.spinner("Translating..."):
        try:
            translation = translate_fn(
                model,
                source_text,
                sp_ar,
                sp_en,
                device,
            )

            st.success("Translation:")
            st.markdown(f"### {translation}")

        except Exception as e:
            st.error(f"Translation error: {e}")


# Example inputs
st.subheader("Examples")
examples = [
    ("مرحبا", "hello"),
    ("كيف حالك", "how are you"),
    ("أنا جيد", "i am good"),
    ("شكرا لك", "thank you"),
    ("مع السلامة", "goodbye"),
]

for ex_ar, ex_en in examples:
    if st.button(f"{ex_ar} →", key=f"ex_{ex_ar}"):
        st.session_state.source_input = ex_ar


# Statistics (when model loaded)
if model:
    st.subheader("Model Statistics")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Parameters", f"{n_params:,}")
    col2.metric("Trainable", f"{n_trainable:,}")
    col3.metric("Device", device.upper())


# Footer
st.markdown("---")
st.caption("Built with PyTorch • Transformer architecture • Educational clarity")

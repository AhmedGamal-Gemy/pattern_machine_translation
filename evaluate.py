"""
Evaluation module for Arabic→English NMT.

Metrics:
1. BLEU (sacrebleu) - Primary metric
2. chrF (sacrebleu) - Secondary metric
3. LLM Judge (GPT-4o-mini) - Qualitative evaluation

Design rationale:
- BLEU is standard NMT metric but captures only n-gram overlap
- chrF captures character-level quality better for morphologically rich languages
- LLM judge provides human-like evaluation of accuracy/fluency/completeness
"""

import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from config import config
from model import Transformer


# ========== BLEU Evaluation ==========


def compute_bleu(
    references: list[str],
    predictions: list[str],
    tokenized: bool = True,
    case_sensitive: bool = True,
) -> dict:
    """
    Compute BLEU score using sacrebleu.

    BLEU: Bilingual Evaluation Understudy
    Paper: Papinemi et al. (2002)

    Why BLEU:
    - Standard metric for machine translation
    - Correlates with human judgment (at corpus level)
    - Fast to compute

    Limitations:
    - Doesn't capture meaning or fluency
    - Favors short translations (penalty for being too long/short)
    - No direct alignment to source

    Args:
        references: List of reference translations
        predictions: List of predicted translations
        tokenized: Use tokenizer (recommended)
        case_sensitive: Case-sensitive matching

    Returns:
        dict with 'score', 'precision', 'recall', etc.
    """
    try:
        import sacrebleu

        # Format for sacrebleu
        refs = [
            [ref] for ref in references
        ]  # List of lists (each ref is list of refs for that sample)

        # Compute BLEU
        bleu = sacrebleu.corpus_bleu(
            predictions,
            refs,
            tokenize="intl" if tokenized else "none",
            case_sensitive=case_sensitive,
        )

        return {
            "score": bleu.score,
            "precision": bleu.precision,
            "recall": bleu.recall,
            "bp": bleu.bp,  # Brevity penalty
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
        }
    except ImportError:
        return {"score": 0.0, "error": "sacrebleu not installed"}


def compute_chrf(
    references: list[str],
    predictions: list[str],
    char_order: int = 6,
    word_order: int = 2,
) -> dict:
    """
    Compute chrF score using sacrebleu.

    chrF: Character n-gram F-score
    Paper: Popovic (2016)

    Why chrF:
    - Better for morphologically rich languages (Arabic)
    - Captures word fragments and inflections
    - More sensitive than BLEU for translation quality

    Args:
        references: List of reference translations
        predictions: List of predicted translations
        char_order: Character n-gram order (1-6)
        word_order: Word n-gram order (0-2)

    Returns:
        dict with 'score', 'precision', 'recall', 'f_score'
    """
    try:
        import sacrebleu

        refs = [[ref] for ref in references]

        chrf = sacrebleu.corpus_chrf(
            predictions,
            refs,
            char_order=char_order,
            word_order=word_order,
        )

        return {
            "score": chrf.score,
            "precision": chrf.precision,
            "recall": chrf.recall,
            "f_score": chrf.fscore,
        }
    except ImportError:
        return {"score": 0.0, "error": "sacrebleu not installed"}


# ========== LLM Judge ==========


def evaluate_with_llm_judge(
    references: list[str],
    predictions: list[str],
    source_texts: list[str],
    model_name: str = "gpt-4o-mini",
    num_samples: int = 50,
) -> dict:
    """
    Evaluate translations using LLM as judge.

    Rubric:
    1. Accuracy (1-5): Does translation preserve original meaning?
    2. Fluency (1-5): Is the English natural and grammatical?
    3. Completeness (1-5): Is anything missing from the translation?
    4. Overall (1-5): Overall quality

    Why LLM judge:
    - Captures semantic quality BLEU misses
    - Evaluates fluency and naturalness
    - Can identify missing/added content

    Args:
        references: Reference translations (ground truth)
        predictions: Model predictions
        source_texts: Source Arabic texts
        model_name: LLM model to use
        num_samples: Number of samples to evaluate

    Returns:
        dict with average scores and per-sample evaluations
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed"}

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    # Sample if too many
    if len(references) > num_samples:
        import random

        random.seed(config.seed)
        indices = random.sample(range(len(references)), num_samples)
        references = [references[i] for i in indices]
        predictions = [predictions[i] for i in indices]
        source_texts = [source_texts[i] for i in indices]

    # Evaluate each sample
    results = []

    for i, (src, ref, hyp) in enumerate(zip(source_texts, references, predictions)):
        prompt = f"""You are an expert translator evaluating Arabic-to-English translation quality.

Source (Arabic): {src}
Reference (English): {ref}
Hypothesis (Your model): {hyp}

Evaluate the hypothesis on a scale of 1-5 for each criterion:

1. Accuracy (1-5): Does the translation preserve the original meaning? Consider:
   - All content translated accurately?
   - No mis-translations or hallucinations?
   - Proper handling of named entities?

2. Fluency (1-5): Is the English natural and grammatical? Consider:
   - Natural word order?
   - Correct grammar and agreement?
   - Idiomatic expression?

3. Completeness (1-5): Is anything missing from the translation? Consider:
   - No content omitted?
   - No truncated translations?
   - All nuance preserved?

4. Overall (1-5): Overall translation quality.

Respond ONLY in this format (JSON):
{{"accuracy": N, "fluency": N, "completeness": N, "overall": N, "reason": "brief explanation"}}
"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )

            result_text = response.choices[0].message.content

            # Parse JSON from response
            result = json.loads(result_text)
            results.append(result)

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # Compute averages
    if not results:
        return {"error": "No evaluations completed"}

    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
    avg_fluency = sum(r["fluency"] for r in results) / len(results)
    avg_completeness = sum(r["completeness"] for r in results) / len(results)
    avg_overall = sum(r["overall"] for r in results) / len(results)

    return {
        "num_samples": len(results),
        "accuracy": avg_accuracy,
        "fluency": avg_fluency,
        "completeness": avg_completeness,
        "overall": avg_overall,
        "individual": results,
    }


# ========== Evaluation Function ==========


def evaluate_model(
    model: Transformer,
    test_loader: torch.utils.data.DataLoader,
    sp_ar,
    sp_en,
    data: list[dict],
    use_bleu: bool = True,
    use_chrf: bool = True,
    use_llm_judge: bool = True,
    checkpoint_path: Path = None,
    device: str = "cuda",
) -> dict:
    """
    Full model evaluation.

    Args:
        model: Trained Transformer model
        test_loader: Test data
        sp_ar: Arabic SentencePiece model
        sp_en: English SentencePiece model
        data: Original parallel data
        use_bleu: Compute BLEU
        use_chrf: Compute chrF
        use_llm_judge: Use LLM judge
        checkpoint_path: Optional checkpoint to load
        device: Device

    Returns:
        dict with all metrics
    """
    model.eval()

    # Generate predictions
    from translate import translate_batch

    all_predictions = []
    all_references = []
    all_sources = []

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src_ids"].to(device)
            src_lens = batch["src_lens"]

            # Translate
            predictions = translate_batch(
                model,
                src,
                sp_en,
                max_len=config.max_len_tgt,
            )

            all_predictions.extend(predictions)

            # Get references (need to decode)
            # For each sample in batch, get corresponding reference
            for i, src_len in enumerate(src_lens):
                # Find the English reference for this sample
                # (Note: This is approximate - ideally use dataset indices)
                pass

    # If we can't match references easily, compute loss-based evaluation instead
    # For proper BLEU, need aligned references
    results = {
        "note": "Generation-based evaluation - need proper reference alignment for BLEU",
    }

    # Compute BLEU if we have aligned references
    if use_bleu and all_references:
        bleu_results = compute_bleu(all_references, all_predictions)
        results["bleu"] = bleu_results

    if use_chrf and all_references:
        chrf_results = compute_chrf(all_references, all_predictions)
        results["chrf"] = chrf_results

    # LLM judge
    if use_llm_judge and all_references and all_predictions and all_sources:
        llm_results = evaluate_with_llm_judge(
            all_references,
            all_predictions,
            all_sources,
        )
        results["llm_judge"] = llm_results

    return results


def evaluate_manual(
    model: Transformer,
    sp_ar,
    sp_en,
    test_pairs: list[dict],
    use_llm: bool = True,
    device: str = "cuda",
) -> dict:
    """
    Manual evaluation on specific examples.

    Args:
        model: Transformer model
        sp_ar: Arabic SPM
        sp_en: English SPM
        test_pairs: List of {'ar': str, 'en': str} pairs
        use_llm: Use LLM judge
        device: Device

    Returns:
        dict with predictions, references, and metrics
    """
    from translate import translate

    predictions = []
    references = []
    sources = []

    model.eval()

    for pair in test_pairs:
        src = pair["ar"]
        ref = pair["en"]

        # Translate
        hyp = translate(model, src, sp_ar, sp_en, device)

        predictions.append(hyp)
        references.append(ref)
        sources.append(src)

        print(f"Source:  {src}")
        print(f"Ref:     {ref}")
        print(f"Hyp:     {hyp}")
        print()

    # Compute BLEU
    if predictions and references:
        bleu = compute_bleu(references, predictions)
        chrf = compute_chrf(references, predictions)

    # LLM judge
    llm_results = None
    if use_llm:
        llm_results = evaluate_with_llm_judge(
            references,
            predictions,
            sources,
        )

    return {
        "predictions": predictions,
        "references": references,
        "sources": sources,
        "bleu": bleu if predictions else None,
        "chrf": chrf if predictions else None,
        "llm_judge": llm_results,
    }


# ========== Entry Point ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    model = Transformer(
        src_vocab_size=config.vocab_size_ar,
        tgt_vocab_size=config.vocab_size_en,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
    ).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Load data and SPM
    from dataset import load_parallel_data, load_sentencepiece

    data = load_parallel_data()
    sp_ar = load_sentencepiece(config.spm_ar_path)
    sp_en = load_sentencepiece(config.spm_en_path)

    # Test on examples
    test_examples = [
        {"ar": "مرحبا", "en": "hello"},
        {"ar": "كيف حالك", "en": "how are you"},
    ]

    results = evaluate_manual(model, sp_ar, sp_en, test_examples, device=device)

    print("\n=== Results ===")
    for key, value in results.items():
        if key not in ("predictions", "references", "sources"):
            print(f"{key}: {value}")

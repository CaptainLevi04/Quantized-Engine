# =============================================================================
# [EVALUATION: PERPLEXITY BENCHMARK]
# Evaluates language model Perplexity (PPL) on WikiText-2 (wikitext-2-raw-v1)
# for Qwen2.5 Base vs. AWQ Quantized models using the custom CUDA engine.
# =============================================================================

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

import engine as E
from run import _resolve_model_dir
from tokenizer import Qwen2Tokenizer


# =============================================================================
# [DATASET LOADER]
# Loads WikiText-2 test split cleanly across different HF datasets versions
# =============================================================================

def load_wikitext_test_text() -> str:
    """Loads the WikiText-2 raw test split and joins paragraphs."""
    try:
        wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        # Fallback for newer huggingface_hub namespace requirement
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(wt["text"])


# =============================================================================
# [PERPLEXITY COMPUTATION]
# Implements sliding window / chunked NLL accumulation matching the reference setup
# =============================================================================

def compute_ppl(
    model: E.Engine,
    tokenizer: Qwen2Tokenizer,
    max_length: int = 1024,
    stride: int = 1024,
    limit_chunks: int | None = None,
    text: str | None = None,
) -> tuple[float, dict]:
    """
    Computes WikiText-2 Perplexity matching the reference formula:
        out = model(input_ids)
        neg_log_likelihood = out.loss * trg_len
        ppl = exp(sum(nlls) / n_tokens)
    """
    if text is None:
        text = load_wikitext_test_text()

    # Encode full corpus into token IDs
    token_ids = tokenizer.encode(text)
    encodings_input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    seq_len = encodings_input_ids.size(1)

    nlls = []
    n_tokens = 0
    prev_end = 0

    chunk_starts = list(range(0, seq_len, stride))
    if limit_chunks:
        chunk_starts = chunk_starts[:limit_chunks]

    t0 = time.perf_counter()

    for begin in tqdm(chunk_starts, desc=f"Computing Perplexity ({getattr(model, '_name', 'Model')})"):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        input_ids = encodings_input_ids[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            out = model.forward(input_ids)
            # Shift so that token at i predicts token at i+1
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            # Compute cross-entropy in float32 to prevent FP16 accumulator overflow
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Negative log likelihood for this window
            neg_log_likelihood = loss.item() * trg_len

        nlls.append(neg_log_likelihood)
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break

    elapsed = time.perf_counter() - t0
    mean_nll = sum(nlls) / n_tokens
    ppl = math.exp(mean_nll)

    stats = {
        "Perplexity": round(ppl, 4),
        "Mean NLL": round(mean_nll, 4),
        "Total Evaluated Tokens": n_tokens,
        "Total Corpus Tokens": seq_len,
        "Chunks Evaluated": len(nlls),
        "Chunk Length (max_length)": max_length,
        "Stride": stride,
        "Wall Time (s)": round(elapsed, 2),
        "Throughput (tok/s)": round(n_tokens / elapsed, 1) if elapsed > 0 else 0.0,
    }
    return ppl, stats


# =============================================================================
# [EVALUATION & COMPARISON RUNNERS]
# =============================================================================

def evaluate_model_ppl(
    model_name_or_dir: str,
    max_length: int = 1024,
    stride: int = 1024,
    limit_chunks: int | None = None,
    text: str | None = None,
) -> dict:
    model_dir = _resolve_model_dir(model_name_or_dir)
    model_name = os.path.basename(model_dir)

    print(f"\n{'=' * 75}")
    print(f"Loading Model: {model_name} from {model_dir}")
    print(f"{'=' * 75}")

    tok = Qwen2Tokenizer.from_pretrained(model_dir)
    eng = E.Engine.load(model_dir)
    eng._name = model_name

    if text is None:
        print("Loading WikiText-2 (wikitext-2-raw-v1, split='test')...")
        text = load_wikitext_test_text()

    ppl, stats = compute_ppl(
        model=eng,
        tokenizer=tok,
        max_length=max_length,
        stride=stride,
        limit_chunks=limit_chunks,
        text=text,
    )

    print(f"\n{'=' * 75}")
    print(f"Perplexity Result: {model_name}")
    print(f"{'=' * 75}")
    print(f"  WikiText-2 Perplexity (PPL) : {ppl:.4f}")
    for k, v in stats.items():
        if k != "Perplexity":
            print(f"  {k:28}: {v}")
    print(f"{'=' * 75}\n")

    stats["Model"] = model_name
    return stats


def compare_models_ppl(
    max_length: int = 1024,
    stride: int = 1024,
    limit_chunks: int | None = None,
    output_csv: str | None = "ppl_comparison.csv",
) -> pd.DataFrame:
    print(f"\n{'#' * 75}")
    print("Starting Perplexity Comparison: Base (FP16) vs. AWQ (Quantized)")
    print(f"{'#' * 75}")

    print("Loading WikiText-2 dataset once for both models...")
    text = load_wikitext_test_text()

    # 1. Base Model
    base_stats = evaluate_model_ppl(
        model_name_or_dir="Qwen2.5-0.5B",
        max_length=max_length,
        stride=stride,
        limit_chunks=limit_chunks,
        text=text,
    )

    torch.cuda.empty_cache()

    # 2. Quantized AWQ Model
    quant_stats = evaluate_model_ppl(
        model_name_or_dir="Qwen2.5-0.5B-AWQ",
        max_length=max_length,
        stride=stride,
        limit_chunks=limit_chunks,
        text=text,
    )

    # 3. Build summary DataFrame
    df = pd.DataFrame([base_stats, quant_stats]).set_index("Model")
    cols = ["Perplexity", "Mean NLL", "Total Evaluated Tokens", "Chunks Evaluated", "Wall Time (s)", "Throughput (tok/s)"]
    df = df[[c for c in cols if c in df.columns]]

    print(f"\n{'=' * 75}")
    print("FINAL WIKITEXT-2 PERPLEXITY COMPARISON")
    print(f"{'=' * 75}")
    print(df.to_string())
    print(f"{'=' * 75}\n")

    if output_csv:
        df.to_csv(output_csv)
        print(f"Saved comparison to: {output_csv}\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate WikiText-2 Perplexity using standalone CUDA engine")
    parser.add_argument("--model", type=str, default=None, help="Model name or dir (Qwen2.5-0.5B or Qwen2.5-0.5B-AWQ)")
    parser.add_argument("--compare", action="store_true", help="Compare both Base and AWQ models")
    parser.add_argument("--max_length", type=int, default=1024, help="Window max length (default: 1024)")
    parser.add_argument("--stride", type=int, default=1024, help="Window stride (default: 1024)")
    parser.add_argument("--limit_chunks", type=int, default=None, help="Limit number of chunks to evaluate (e.g. 50)")
    parser.add_argument("--output", type=str, default="ppl_comparison.csv", help="CSV output path")
    args = parser.parse_args()

    if args.compare or args.model is None:
        compare_models_ppl(
            max_length=args.max_length,
            stride=args.stride,
            limit_chunks=args.limit_chunks,
            output_csv=args.output,
        )
    else:
        evaluate_model_ppl(
            model_name_or_dir=args.model,
            max_length=args.max_length,
            stride=args.stride,
            limit_chunks=args.limit_chunks,
        )


if __name__ == "__main__":
    main()

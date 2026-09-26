# =============================================================================
# [EVALUATION: IFEVAL BENCHMARK]
# Evaluates Qwen2.5 models (Original FP16 or AWQ Quantized) on IFEval
# (Instruction-Following Evaluation) using the custom standalone CUDA engine.
# =============================================================================

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from tqdm import tqdm

from instruction_following_eval import get_examples, evaluate_instruction_following
import engine as E
from run import _resolve_model_dir
from tokenizer import Qwen2Tokenizer


# =============================================================================
# [IFEVAL SCORING UTILITIES]
# Matches the exact scoring logic from the reference notebook
# =============================================================================

def score_ifeval_results(results: list[dict], model_name: str) -> dict:
    examples_for_eval = [
        {
            "key": r["key"],
            "instruction_id_list": r["instruction_id_list"],
            "prompt": r["prompt"],
            "kwargs": r["kwargs"],
        }
        for r in results
    ]
    responses = [r["response"] for r in results]

    metrics = evaluate_instruction_following(examples_for_eval, responses)

    # 4 standard sub-scores: prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose
    p_strict = metrics.get("prompt_level_strict_accuracy", 0.0)
    i_strict = metrics.get("inst_level_strict_accuracy", 0.0)
    p_loose = metrics.get("prompt_level_loose_accuracy", 0.0)
    i_loose = metrics.get("inst_level_loose_accuracy", 0.0)

    overall_avg = (p_strict + i_strict + p_loose + i_loose) / 4.0

    return {
        "Model": model_name,
        "Samples": len(results),
        "Prompt Strict (%)": round(100.0 * p_strict, 2),
        "Inst Strict (%)": round(100.0 * i_strict, 2),
        "Prompt Loose (%)": round(100.0 * p_loose, 2),
        "Inst Loose (%)": round(100.0 * i_loose, 2),
        "IFEval Overall (%)": round(100.0 * overall_avg, 2),
    }


# =============================================================================
# [INFERENCE RUNNER]
# Generates responses sample-by-sample through the standalone CUDA engine
# =============================================================================

def run_ifeval_model(
    model_dir: str,
    num_samples: int = 50,
    max_tokens: int = 256,
    output_json: str | None = None,
) -> tuple[dict, list[dict]]:
    model_dir = _resolve_model_dir(model_dir)
    model_name = os.path.basename(model_dir)

    print(f"\n{'=' * 75}")
    print(f"Loading Model: {model_name} from {model_dir}")
    print(f"{'=' * 75}")

    tok = Qwen2Tokenizer.from_pretrained(model_dir)
    eng = E.Engine.load(model_dir)

    examples = get_examples()[:num_samples]
    print(f"Loaded {len(examples)} IFEval prompts. Max new tokens: {max_tokens}\n")

    results = []
    total_gen_tokens = 0
    start_wall = time.perf_counter()

    eos_ids = [tok.eos_token_id, 151645]  # <|endoftext|> and <|im_end|>

    for idx, ex in enumerate(tqdm(examples, desc=f"Evaluating IFEval ({model_name})")):
        prompt = ex["prompt"]
        ids = torch.tensor([tok.encode(prompt)], device="cuda")

        t0 = time.perf_counter()
        out = eng.generate(ids, max_new_tokens=max_tokens, eos_token_id=eos_ids)
        elapsed = time.perf_counter() - t0

        response = tok.decode(out, skip_special_tokens=True).strip()
        total_gen_tokens += len(out)

        record = {
            "key": ex["key"],
            "instruction_id_list": ex["instruction_id_list"],
            "prompt": prompt,
            "kwargs": ex["kwargs"],
            "response": response,
            "model": model_name,
            "benchmark": "ifeval",
            "prompt_tokens": ids.shape[1],
            "generated_tokens": len(out),
            "latency_s": round(elapsed, 4),
            "tokens_per_second": round(len(out) / elapsed, 2) if elapsed > 0 else 0.0,
        }
        results.append(record)

    total_wall = time.perf_counter() - start_wall
    summary = score_ifeval_results(results, model_name)
    summary["Wall Time (s)"] = round(total_wall, 2)
    summary["Avg Throughput (tok/s)"] = round(total_gen_tokens / total_wall, 2) if total_wall > 0 else 0.0

    print(f"\n{'=' * 75}")
    print(f"IFEval Benchmark Summary: {model_name}")
    print(f"{'=' * 75}")
    for k, v in summary.items():
        print(f"  {k:25}: {v}")
    print(f"{'=' * 75}\n")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed output saved to: {output_json}\n")

    return summary, results


def run_comparison(num_samples: int = 50, max_tokens: int = 256):
    print(f"\n{'#' * 75}")
    print(f"Starting IFEval Benchmark Comparison ({num_samples} samples)")
    print(f"{'#' * 75}")

    # 1. Base Model
    base_summary, base_results = run_ifeval_model(
        model_dir="Qwen2.5-0.5B",
        num_samples=num_samples,
        max_tokens=max_tokens,
        output_json="base_ifeval_results.json",
    )

    # Clean GPU memory between runs
    torch.cuda.empty_cache()

    # 2. Quantized AWQ Model
    quant_summary, quant_results = run_ifeval_model(
        model_dir="Qwen2.5-0.5B-AWQ",
        num_samples=num_samples,
        max_tokens=max_tokens,
        output_json="awq_ifeval_results.json",
    )

    # 3. Comparison Table
    df = pd.DataFrame([base_summary, quant_summary]).set_index("Model")
    print(f"\n{'=' * 75}")
    print(f"FINAL IFEVAL COMPARISON TABLE ({num_samples} samples)")
    print(f"{'=' * 75}")
    print(df.to_string())
    print(f"{'=' * 75}\n")

    # Save summary table
    df.to_csv("ifeval_comparison.csv")
    print("Comparison saved to: ifeval_comparison.csv\n")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 on IFEval benchmark using standalone CUDA engine")
    parser.add_argument("--model", type=str, default=None, help="Model name or dir (Qwen2.5-0.5B or Qwen2.5-0.5B-AWQ)")
    parser.add_argument("--compare", action="store_true", help="Run both Base and AWQ models and print comparison table")
    parser.add_argument("--samples", type=int, default=50, help="Number of IFEval prompt samples (default: 50)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum generated tokens per sample (default: 256)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.compare or args.model is None:
        run_comparison(num_samples=args.samples, max_tokens=args.max_tokens)
    else:
        run_ifeval_model(
            model_dir=args.model,
            num_samples=args.samples,
            max_tokens=args.max_tokens,
            output_json=args.output,
        )


if __name__ == "__main__":
    main()

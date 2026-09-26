# =============================================================================
# [EVALUATION: MMLU BENCHMARK]
# Evaluates Qwen2.5 models (Original FP16 or AWQ Quantized) on MMLU
# using the custom standalone CUDA engine.
# =============================================================================

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import load_dataset
from tqdm import tqdm

import engine as E
from run import _resolve_model_dir
from tokenizer import Qwen2Tokenizer

# =============================================================================
# [MMLU PROMPT & PARSING UTILITIES]
# Matches the exact evaluation setup from the reference notebook
# =============================================================================

def build_mmlu_prompt(sample: dict) -> tuple[str, str]:
    choices = "\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample["choices"])
    )
    prompt = f"""Multiple-choice question.
Answer with ONLY one letter (A, B, C, or D).

Question:
{sample["question"]}

Choices:
{choices}

Answer:
"""
    ground_truth = chr(65 + sample["answer"])
    return prompt, ground_truth


def extract_mcq_answer(prediction: str) -> str | None:
    prediction = prediction.strip()
    if not prediction:
        return None
    first_word = prediction.split(maxsplit=1)[0]
    match = re.fullmatch(r"([A-D])[\.\):]?", first_word)
    return match.group(1) if match else None


def score_mcq(results: list[dict], model_name: str) -> dict:
    correct = wrong = invalid = 0
    for sample in results:
        pred = extract_mcq_answer(sample["prediction"])
        if pred is None:
            invalid += 1
            continue
        if pred == sample["ground_truth"]:
            correct += 1
        else:
            wrong += 1

    evaluated = correct + wrong
    accuracy = 100.0 * correct / evaluated if evaluated > 0 else 0.0

    return {
        "Model": model_name,
        "Correct": correct,
        "Wrong": wrong,
        "Invalid": invalid,
        "Evaluated": evaluated,
        "Total": len(results),
        "Accuracy (%)": round(accuracy, 2),
    }


# =============================================================================
# [EVALUATION RUNNER]
# Runs inference question-by-question through the standalone CUDA engine
# =============================================================================

def evaluate_mmlu(
    model_dir: str,
    num_samples: int = 100,
    seed: int = 42,
    output_json: str | None = None,
    max_tokens: int = 8,
) -> dict:
    model_dir = _resolve_model_dir(model_dir)
    model_name = os.path.basename(model_dir)

    print(f"\n{'=' * 70}")
    print(f"Loading Model: {model_name} from {model_dir}")
    print(f"{'=' * 70}")

    tok = Qwen2Tokenizer.from_pretrained(model_dir)
    eng = E.Engine.load(model_dir)

    print(f"\nLoading MMLU test split (seed={seed}, samples={num_samples})...")
    mmlu_dataset = (
        load_dataset("cais/mmlu", "all", split="test")
        .shuffle(seed=seed)
        .select(range(num_samples))
    )

    results = []
    total_gen_tokens = 0
    start_time = time.perf_counter()

    for idx, sample in enumerate(tqdm(mmlu_dataset, desc=f"Evaluating {model_name}")):
        prompt, ground_truth = build_mmlu_prompt(sample)
        ids = torch.tensor([tok.encode(prompt)], device="cuda")

        t0 = time.perf_counter()
        out = eng.generate(ids, max_new_tokens=max_tokens, eos_token_id=tok.eos_token_id)
        elapsed = time.perf_counter() - t0

        prediction = tok.decode(out, skip_special_tokens=True).strip()
        parsed_choice = extract_mcq_answer(prediction)
        is_correct = (parsed_choice == ground_truth) if parsed_choice else False

        total_gen_tokens += len(out)
        results.append({
            "id": idx,
            "subject": sample.get("subject", "unknown"),
            "question": sample["question"],
            "ground_truth": ground_truth,
            "prediction_raw": prediction,
            "prediction_parsed": parsed_choice,
            "is_correct": is_correct,
            "prompt_tokens": ids.shape[1],
            "generated_tokens": len(out),
            "latency_s": round(elapsed, 4),
        })

    total_wall_time = time.perf_counter() - start_time
    summary = score_mcq(
        [{"prediction": r["prediction_raw"], "ground_truth": r["ground_truth"]} for r in results],
        model_name,
    )
    summary["Wall Time (s)"] = round(total_wall_time, 2)
    summary["Throughput (tok/s)"] = round(total_gen_tokens / total_wall_time, 2) if total_wall_time > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"MMLU Evaluation Results: {model_name}")
    print(f"{'=' * 70}")
    for k, v in summary.items():
        print(f"  {k:20}: {v}")
    print(f"{'=' * 70}\n")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {output_json}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 on MMLU using standalone CUDA engine")
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B-AWQ", help="Model name or directory (e.g. Qwen2.5-0.5B or Qwen2.5-0.5B-AWQ)")
    parser.add_argument("--samples", type=int, default=100, help="Number of MMLU samples to evaluate (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path for detailed predictions")
    args = parser.parse_args()

    evaluate_mmlu(
        model_dir=args.model,
        num_samples=args.samples,
        seed=args.seed,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CTF Model Evaluator — run trained models against CTF challenges and report results.

Usage:
  uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora
  uv run src/eval.py --compare gemma4:outputs/gemma4-ctf/lora qwen35:outputs/qwen35-ctf/lora
  uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora --category pwn --difficulty easy
  uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora --output data/eval/results/
"""
import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch


def load_benchmarks(bench_path: str = None) -> list[dict]:
    if bench_path is None:
        bench_path = Path(__file__).parent.parent / "data" / "eval" / "ctf_bench.jsonl"
    with open(bench_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% CI by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = count / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def load_model(model_key: str, adapter_path: str):
    """Load model with LoRA adapter for inference."""
    from unsloth import get_chat_template

    if model_key.startswith("gemma"):
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(
            model_name=_model_name(model_key),
            load_in_4bit=True,
        )
        processor = get_chat_template(processor, "gemma-4")
        tokenizer = processor.tokenizer
        model = FastVisionModel.get_for_inference(model)
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=_model_name(model_key),
            load_in_4bit=True,
            dtype=None,
        )
        tokenizer = get_chat_template(tokenizer, "chatml")
        model = FastLanguageModel.get_for_inference(model)

    # Load LoRA adapter
    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded LoRA adapter from {adapter_path}")

    return model, tokenizer


def _model_name(model_key: str) -> str:
    """Map model key to HuggingFace model name."""
    from src.train import load_config
    config = load_config(model_key)
    return config.get("model", config).get("name", model_key)


def generate_response(model, tokenizer, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = 512) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    inputs = actual_tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy; kills sampling variance
        )
    response = actual_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def grade_flag(response: str, expected: str) -> bool:
    """Check if response contains the expected flag."""
    flags = re.findall(r'flag\{[^}]+\}', response)
    return expected in flags


def grade_mcq(response: str, expected: str) -> bool:
    """Check if response matches the expected MCQ answer."""
    # Layered matcher: look for explicit answer patterns first
    pat = re.compile(r'(?i)(?:answer\s*(?:is|:)?\s*|\*?answer\*?\s*:?\s*|\()\s*([A-D])\b')
    m = pat.search(response)
    if m:
        return m.group(1).upper() == expected.upper()
    # Fallback: last A-D letter in response
    last = re.findall(r'(?i)\b([A-D])\b', response)
    return bool(last) and last[-1].upper() == expected.upper()


def grade_code(response: str, reference: str = None) -> bool:
    """Check if response contains syntactically valid code with reference keywords."""
    code_blocks = re.findall(r'```(?:python|c|bash|py)?\n(.*?)```', response, re.DOTALL)
    if not code_blocks:
        # Try to find code without markdown fences
        lines = response.split('\n')
        code_lines = [l for l in lines if any(kw in l for kw in ['import ', 'from ', 'def ', 'for ', 'if ', 'print(', 'p ='])]
        if code_lines:
            code_blocks = ['\n'.join(code_lines)]

    if not code_blocks:
        return False

    candidate = code_blocks[0]
    try:
        compile(candidate.strip(), '<eval>', 'exec')
    except SyntaxError:
        return False

    if reference is None:
        return True

    # Require key tokens from reference to appear in generated code
    needed = re.findall(r'[A-Za-z_][A-Za-z_0-9]{4,}', reference)
    return any(tok in candidate for tok in needed[:3])


def grade(challenge: dict, response: str) -> bool:
    """Grade a response based on task type."""
    task_type = challenge["task_type"]
    expected = challenge["expected"]

    if task_type == "flag_extraction":
        return grade_flag(response, expected)
    elif task_type == "multiple_choice":
        return grade_mcq(response, expected)
    elif task_type == "code_generation":
        return grade_code(response, challenge.get("reference"))
    return False


def run_evaluation(model_key: str, adapter_path: str, bench_path: str = None,
                   category: str = None, difficulty: str = None) -> dict:
    """Run full evaluation and return results."""
    print(f"\nLoading model: {model_key}")
    t0 = time.time()
    model, tokenizer = load_model(model_key, adapter_path)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    benchmarks = load_benchmarks(bench_path)
    if category:
        benchmarks = [b for b in benchmarks if b["category"] == category]
    if difficulty:
        benchmarks = [b for b in benchmarks if b["difficulty"] == difficulty]

    print(f"  Running {len(benchmarks)} challenges...\n")
    results = []
    for i, ch in enumerate(benchmarks):
        t0 = time.time()
        response = generate_response(model, tokenizer, ch["system_prompt"], ch["prompt"])
        correct = grade(ch, response)
        elapsed = time.time() - t0

        results.append({
            "id": ch["id"],
            "category": ch["category"],
            "difficulty": ch["difficulty"],
            "task_type": ch["task_type"],
            "correct": correct,
            "response": response,
            "expected": ch["expected"],
            "time": elapsed,
        })

        mark = "✓" if correct else "✗"
        print(f"  [{i+1}/{len(benchmarks)}] {mark} {ch['id']} ({ch['difficulty']}) {elapsed:.1f}s")

    return {
        "model": model_key,
        "adapter": adapter_path,
        "results": results,
    }


def print_results(eval_result: dict):
    """Print formatted evaluation results with Wilson 95% CI."""
    results = eval_result["results"]
    model_key = eval_result["model"]

    # Group by category-difficulty bucket
    buckets = defaultdict(lambda: {"correct": 0, "total": 0, "times": []})
    for r in results:
        key = f"{r['category']}-{r['difficulty']}"
        buckets[key]["total"] += 1
        if r["correct"]:
            buckets[key]["correct"] += 1
        buckets[key]["times"].append(r["time"])

    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    mean_lat = sum(r["time"] for r in results) / total if total else 0

    print(f"\n{'='*70}")
    print(f"CTF Evaluation Results")
    print(f"{'='*70}")
    print(f"Model:   {model_key}")
    print(f"Dataset: {total} questions")
    print()

    # Bucket breakdown with Wilson CI
    header = f"{'Bucket':<16} {'acc':>5} {'n':>4} {'CI95_lo':>7} {'CI95_hi':>7} {'mean_lat':>9}"
    print(header)
    print("-" * len(header))

    categories = sorted(set(r["category"] for r in results))
    difficulties = ["easy", "medium", "hard"]

    for cat in categories:
        for d in difficulties:
            key = f"{cat}-{d}"
            b = buckets[key]
            if b["total"] == 0:
                continue
            acc = b["correct"] / b["total"]
            lo, hi = wilson_ci(b["correct"], b["total"])
            avg_t = sum(b["times"]) / len(b["times"])
            print(f"{key:<16} {acc:>4.0%} {b['total']:>4} {lo:>6.0%} {hi:>6.0%} {avg_t:>7.1f}s")

    # Overall
    lo, hi = wilson_ci(total_correct, total)
    print("-" * len(header))
    print(f"{'overall':<16} {total_correct/total:>4.0%} {total:>4} {lo:>6.0%} {hi:>6.0%} {mean_lat:>7.1f}s")
    print()

    # Per-question breakdown
    print("Per-question breakdown:")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} {r['id']:<12} ({r['difficulty']:<6}) ", end="")
        if r["correct"]:
            print(f"{r['expected']}")
        else:
            resp_preview = r['response'][:60].replace('\n', ' ')
            print(f"got: \"{resp_preview}...\"")


def print_comparison(eval_results: list[dict]):
    """Print side-by-side comparison with McNemar's test and per-question diff."""
    print(f"\n{'='*70}")
    print(f"CTF Model Comparison")
    print(f"{'='*70}")

    # Build comparison table
    categories = sorted(set(r["category"] for er in eval_results for r in er["results"]))

    # Header
    header = f"{'Category':<10}"
    for er in eval_results:
        name = er["model"]
        header += f" {name:<15}"
    print(header)
    print("-" * len(header))

    # Per-category
    for cat in categories:
        row = f"{cat:<10}"
        for er in eval_results:
            cat_results = [r for r in er["results"] if r["category"] == cat]
            correct = sum(1 for r in cat_results if r["correct"])
            total = len(cat_results)
            lo, hi = wilson_ci(correct, total)
            row += f" {correct}/{total} ({lo:.0%}-{hi:.0%}){'':<2}"
        print(row)

    # Overall
    print("-" * len(header))
    row = f"{'Overall':<10}"
    for er in eval_results:
        correct = sum(1 for r in er["results"] if r["correct"])
        total = len(er["results"])
        lo, hi = wilson_ci(correct, total)
        row += f" {correct}/{total} ({lo:.0%}-{hi:.0%}){'':<2}"
    print(row)
    print()

    # McNemar's test for 2-model comparison
    if len(eval_results) == 2:
        r1 = {r["id"]: r["correct"] for r in eval_results[0]["results"]}
        r2 = {r["id"]: r["correct"] for r in eval_results[1]["results"]}
        ids = sorted(set(r1.keys()) & set(r2.keys()))

        both_right = sum(1 for i in ids if r1[i] and r2[i])
        both_wrong = sum(1 for i in ids if not r1[i] and not r2[i])
        a_only = sum(1 for i in ids if r1[i] and not r2[i])
        b_only = sum(1 for i in ids if not r1[i] and r2[i])
        n = len(ids)

        # McNemar's chi-squared with continuity correction
        if a_only + b_only > 0:
            chi2 = (abs(a_only - b_only) - 1)**2 / (a_only + b_only)
            # p-value from chi2 with 1 df (approximate)
            p_value = math.exp(-chi2 / 2)  # rough approximation
        else:
            chi2 = 0.0
            p_value = 1.0

        a_name = eval_results[0]["model"]
        b_name = eval_results[1]["model"]

        print("McNemar's Test:")
        print(f"  {a_name} wins: {a_only}  |  {b_name} wins: {b_only}  |  ties: {both_right + both_wrong}")
        print(f"  chi² = {chi2:.2f}, p ≈ {p_value:.4f}")
        if n < 25:
            print(f"  Note: n={n} < 25, test has low power")
        elif p_value < 0.05:
            winner = a_name if a_only > b_only else b_name
            print(f"  Decision: {winner} is significantly better (p < 0.05)")
        else:
            print(f"  Decision: no significant difference (p ≥ 0.05)")
        print()

        # Per-question diff
        print("Per-question diff:")
        for i in ids:
            if r1[i] != r2[i]:
                a_mark = "✓" if r1[i] else "✗"
                b_mark = "✓" if r2[i] else "✗"
                ch = next(r for r in eval_results[0]["results"] if r["id"] == i)
                print(f"  {i:<12} ({ch['category']:<6}) {a_name}: {a_mark}  {b_name}: {b_mark}")
        print()


def save_results(eval_results: list[dict], output_dir: str):
    """Save evaluation results as timestamped JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(eval_results) == 1:
        r = eval_results[0]
        filename = f"eval_{r['model']}_{ts}.json"
    else:
        filename = f"eval_compare_{ts}.json"

    filepath = output_path / filename
    data = {
        "timestamp": datetime.now().isoformat(),
        "models": [er["model"] for er in eval_results],
        "results": [],
    }
    for er in eval_results:
        correct = sum(1 for r in er["results"] if r["correct"])
        total = len(er["results"])
        lo, hi = wilson_ci(correct, total)
        data["results"].append({
            "model": er["model"],
            "adapter": er["adapter"],
            "accuracy": correct / total if total else 0,
            "wilson_ci95": [lo, hi],
            "correct": correct,
            "total": total,
            "questions": er["results"],
        })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="CTF Model Evaluator")
    parser.add_argument("--model", choices=["gemma4", "gemma4-12b", "qwen35", "qwen35-4b"],
                        help="Model to evaluate")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    parser.add_argument("--bench", help="Path to benchmark JSONL (default: data/eval/ctf_bench.jsonl)")
    parser.add_argument("--category", choices=["pwn", "rev", "crypto", "web"],
                        help="Filter by category")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--compare", nargs="+", metavar="MODEL:ADAPTER",
                        help="Compare multiple models (e.g., gemma4:outputs/gemma4-ctf/lora qwen35:outputs/qwen35-ctf/lora)")
    parser.add_argument("--output", help="Directory to save results JSON (default: no save)")

    args = parser.parse_args()

    if args.compare:
        eval_results = []
        for spec in args.compare:
            model_key, adapter = spec.split(":", 1)
            result = run_evaluation(model_key, adapter, args.bench, args.category, args.difficulty)
            eval_results.append(result)
        print_comparison(eval_results)
    elif args.model:
        result = run_evaluation(args.model, args.adapter or "", args.bench, args.category, args.difficulty)
        eval_results = [result]
        print_results(result)
    else:
        parser.print_help()
        return

    if args.output:
        save_results(eval_results, args.output)


if __name__ == "__main__":
    main()

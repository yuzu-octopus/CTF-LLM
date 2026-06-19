#!/usr/bin/env python3
"""
CTF Model Evaluator — run trained models against CTF challenges and report results.

Usage:
  uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora
  uv run src/eval.py --compare gemma4:outputs/gemma4-ctf/lora qwen35:outputs/qwen35-ctf/lora
  uv run src/eval.py --model gemma4 --adapter outputs/gemma4-ctf/lora --category pwn --difficulty easy
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch


def load_benchmarks(bench_path: str = None) -> list[dict]:
    if bench_path is None:
        bench_path = Path(__file__).parent.parent / "data" / "eval" / "ctf_bench.jsonl"
    with open(bench_path) as f:
        return [json.loads(line) for line in f if line.strip()]


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
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
    response = actual_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def grade_flag(response: str, expected: str) -> bool:
    """Check if response contains the expected flag."""
    flags = re.findall(r'flag\{[^}]+\}', response)
    return expected in flags


def grade_mcq(response: str, expected: str) -> bool:
    """Check if response matches the expected MCQ answer."""
    match = re.search(r'\b([A-D])\b', response)
    return bool(match and match.group(1) == expected)


def grade_code(response: str, reference: str = None) -> bool:
    """Check if response contains syntactically valid code."""
    code_blocks = re.findall(r'```(?:python|c|bash|py)?\n(.*?)```', response, re.DOTALL)
    if not code_blocks:
        # Try to find code without markdown fences
        lines = response.split('\n')
        code_lines = [l for l in lines if any(kw in l for kw in ['import ', 'from ', 'def ', 'for ', 'if ', 'print(', 'p ='])]
        if code_lines:
            code_blocks = ['\n'.join(code_lines)]

    for block in code_blocks:
        try:
            compile(block.strip(), '<eval>', 'exec')
            return True
        except SyntaxError:
            continue
    return False


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
    """Print formatted evaluation results."""
    results = eval_result["results"]
    model_key = eval_result["model"]

    # Group by category and difficulty
    cat_diff = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0}))
    for r in results:
        cat_diff[r["category"]][r["difficulty"]]["total"] += 1
        if r["correct"]:
            cat_diff[r["category"]][r["difficulty"]]["correct"] += 1

    categories = sorted(cat_diff.keys())
    difficulties = ["easy", "medium", "hard"]

    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)

    print(f"\n{'='*60}")
    print(f"CTF Evaluation Results")
    print(f"{'='*60}")
    print(f"Model:   {model_key}")
    print(f"Dataset: {total} questions")

    # Category counts
    cat_counts = defaultdict(int)
    for r in results:
        cat_counts[r["category"]] += 1
    cat_str = " ".join(f"{k}({v})" for k, v in sorted(cat_counts.items()))
    print(f"         {cat_str}")
    print()

    # Header
    header = f"{'Category':<10}"
    for d in difficulties:
        header += f" {d.capitalize():<10}"
    header += f" {'Overall':<12}"
    print(header)
    print("-" * len(header))

    # Per-category rows
    for cat in categories:
        row = f"{cat:<10}"
        cat_total = 0
        cat_correct = 0
        for d in difficulties:
            stats = cat_diff[cat][d]
            cat_total += stats["total"]
            cat_correct += stats["correct"]
            if stats["total"] > 0:
                row += f" {stats['correct']}/{stats['total']:<8}"
            else:
                row += f" {'—':<10}"
        pct = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        row += f" {cat_correct}/{cat_total} ({pct:.0f}%)"
        print(row)

    # Overall
    pct = (total_correct / total * 100) if total > 0 else 0
    print("-" * len(header))
    print(f"{'Overall':<10}" + " " * (len(header) - 10 - len(f"{total_correct}/{total} ({pct:.0f}%)")) + f"{total_correct}/{total} ({pct:.0f}%)")
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
    """Print side-by-side comparison of multiple models."""
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
            pct = (correct / total * 100) if total > 0 else 0
            row += f" {correct}/{total} ({pct:.0f}%){'':<6}"
        print(row)

    # Overall
    print("-" * len(header))
    row = f"{'Overall':<10}"
    for er in eval_results:
        correct = sum(1 for r in er["results"] if r["correct"])
        total = len(er["results"])
        pct = (correct / total * 100) if total > 0 else 0
        row += f" {correct}/{total} ({pct:.0f}%){'':<6}"
    print(row)
    print()


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
        print_results(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

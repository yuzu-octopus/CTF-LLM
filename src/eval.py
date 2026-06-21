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

try:
    import torch
except ImportError:
    torch = None


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
    from src.model_utils import load_model_for_inference
    return load_model_for_inference(model_key, adapter_path)


def _model_name(model_key: str) -> str:
    """Map model key to HuggingFace model name."""
    from src.model_utils import _model_name as _mn
    return _mn(model_key)


def generate_response(model, tokenizer, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = 512, n_samples: int = 1) -> list[str]:
    """Generate response(s) from the model. Returns list of responses."""
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

    responses = []
    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(n_samples > 1),
                temperature=0.6 if n_samples > 1 else None,
            )
        responses.append(actual_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip())
    return responses


def grade_flag(response: str, expected: str) -> tuple[bool, str]:
    """Check if response contains the expected flag."""
    flags = re.findall(r'flag\{[^}]+\}', response)
    if not flags:
        return False, "No flag{...} pattern found"
    if expected in flags:
        return True, f"Matched: {expected}"
    return False, f"Found {flags} but expected {expected}"


def grade_mcq(response: str, expected: str) -> tuple[bool, str]:
    """Check if response matches the expected MCQ answer."""
    pat = re.compile(r'(?i)(?:answer\s*(?:is|:)?\s*|\*?answer\*?\s*:?\s*|\()\s*([A-D])\b')
    m = pat.search(response)
    if m:
        correct = m.group(1).upper() == expected.upper()
        return correct, f"Explicit match: {m.group(1).upper()} (expected {expected.upper()})"
    last = re.findall(r'(?i)\b([A-D])\b', response)
    if not last:
        return False, "No A-D letter found"
    correct = last[-1].upper() == expected.upper()
    return correct, f"Last-letter fallback: {last[-1].upper()} (expected {expected.upper()})"


def grade_code(response: str, reference: str = None, test_cases: list = None) -> tuple[bool, str]:
    """Check if response contains syntactically valid code with reference keywords."""
    code_blocks = re.findall(r'```(?:python|c|bash|py)?\n(.*?)```', response, re.DOTALL)
    if not code_blocks:
        lines = response.split('\n')
        code_lines = [l for l in lines if any(kw in l for kw in ['import ', 'from ', 'def ', 'for ', 'if ', 'print(', 'p ='])]
        if code_lines:
            code_blocks = ['\n'.join(code_lines)]

    if not code_blocks:
        return False, "No code block found"

    candidate = code_blocks[0]
    try:
        compile(candidate.strip(), '<eval>', 'exec')
    except SyntaxError as e:
        return False, f"SyntaxError: {str(e)[:80]}"

    # No test_cases: legacy reference-token fallback
    if not test_cases:
        if reference is None:
            return True, "Syntax valid (no reference)"
        needed = re.findall(r'[A-Za-z_][A-Za-z_0-9]{4,}', reference)
        if any(tok in candidate for tok in needed[:3]):
            return True, "Reference tokens matched"
        return False, "Reference tokens not found"

    # Functional test cases — restricted exec with safe builtins
    try:
        safe_builtins = {
            "range": range, "len": len, "int": int, "float": float,
            "str": str, "bool": bool, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "print": print, "True": True,
            "False": False, "None": None, "abs": abs, "min": min,
            "max": max, "sum": sum, "sorted": sorted, "reversed": reversed,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        }
        local_env = {"__builtins__": safe_builtins}
        exec(candidate, local_env)
        for i, tc in enumerate(test_cases):
            if "setup" in tc:
                exec(tc["setup"], local_env)
            if not eval(tc["assert"], local_env):
                return False, f"Test {i+1} failed: {tc['assert']}"
        return True, f"All {len(test_cases)} tests passed"
    except Exception as e:
        return False, f"Runtime error: {str(e)[:80]}"


def grade_patch(response: str, banned: list = None, required: list = None) -> tuple[bool, str]:
    """Grade a patch: check for banned tokens and required tokens."""
    code_blocks = re.findall(r'```(?:python|c|bash|py)?\n(.*?)```', response, re.DOTALL)
    candidate = code_blocks[0] if code_blocks else response

    banned = banned or []
    required = required or []

    if any(t in candidate for t in banned):
        return False, f"Used banned token: {next(t for t in banned if t in candidate)}"
    if not all(t in candidate for t in required):
        missing = [t for t in required if t not in candidate]
        return False, f"Missing required: {missing}"
    return True, "Patch passes token checks"


def grade_vuln_id(response: str, expected: str) -> tuple[bool, str]:
    """Grade vulnerability identification — reuse grade_mcq logic."""
    return grade_mcq(response, expected)


def grade_exploit_trace(response: str, required_steps: list = None) -> tuple[bool, str]:
    """Grade exploit trace by checking for required steps."""
    if not required_steps:
        return True, "No steps required"

    matched = sum(1 for step in required_steps if re.search(step, response, re.I))
    if matched == len(required_steps):
        return True, f"All {len(required_steps)} steps found"
    return False, f"Found {matched}/{len(required_steps)} steps"


def grade(challenge: dict, response: str) -> tuple[bool, str]:
    """Grade a response based on task type. Returns (correct, feedback)."""
    task_type = challenge["task_type"]
    expected = challenge["expected"]

    if task_type == "flag_extraction":
        return grade_flag(response, expected)
    elif task_type == "multiple_choice":
        return grade_mcq(response, expected)
    elif task_type == "code_generation":
        return grade_code(response, challenge.get("reference"), challenge.get("test_cases"))
    elif task_type == "vulnerability_identification":
        return grade_vuln_id(response, expected)
    elif task_type == "patch_generation":
        return grade_patch(response, challenge.get("banned_tokens"), challenge.get("required_tokens"))
    elif task_type == "exploit_trace":
        return grade_exploit_trace(response, challenge.get("required_steps"))
    return False, f"Unknown task_type: {task_type}"


def grade_with_subtasks(response: str, challenge: dict) -> tuple[float, str]:
    """Grade a response with optional subtask partial credit. Returns (score, feedback)."""
    subtasks = challenge.get("subtasks", [])
    if not subtasks:
        correct, fb = grade(challenge, response)
        return float(correct), fb

    total_weight = sum(s.get("weight", 1.0) for s in subtasks)
    matched_weight = 0.0
    details = []
    for st in subtasks:
        pat = st.get("criterion", "")
        hit = bool(re.search(pat, response, re.I)) if pat else False
        details.append({"name": st["name"], "hit": hit, "weight": st.get("weight", 1.0)})
        if hit:
            matched_weight += st.get("weight", 1.0)
    score = matched_weight / total_weight if total_weight else 0
    return score, json.dumps(details)


SUSPICIOUS_MARKERS = [
    "writeup from", "write-up", "originally by", "Author: ",
    "HTB ", "Hack The Box", "from writeup", "picoCTF writeup",
    "walkthrough", "solution from", "according to writeup",
]


import hashlib


def check_contamination(benchmarks: list[dict]):
    """Check for overlap between benchmark and training corpus."""
    try:
        train_path = Path(__file__).parent.parent / "data" / "merged" / "train.jsonl"
        if not train_path.exists():
            return
        def _h(d): return hashlib.sha256((d.get("prompt", "") + d.get("instruction", "")).encode()).hexdigest()[:16]
        bench_hashes = {_h(b) for b in benchmarks}
        train_hashes = {_h(json.loads(l)) for l in open(train_path) if l.strip()}
        overlap = bench_hashes & train_hashes
        print(f"  Bench ∩ Train overlap: {len(overlap)}/{len(bench_hashes)} challenges")
        if len(overlap) / max(len(bench_hashes), 1) > 0.05:
            print("  ⚠  >5% of bench is in training corpus — scores may be inflated")
    except Exception:
        pass


def run_evaluation(model_key: str, adapter_path: str, bench_path: str = None,
                   category: str = None, difficulty: str = None, n_samples: int = 1) -> dict:
    """Run full evaluation and return results."""
    print(f"\nLoading model: {model_key}")
    t0 = time.time()
    model, tokenizer = load_model(model_key, adapter_path)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    benchmarks = load_benchmarks(bench_path)
    check_contamination(benchmarks)
    if category:
        benchmarks = [b for b in benchmarks if b["category"] == category]
    if difficulty:
        benchmarks = [b for b in benchmarks if b["difficulty"] == difficulty]

    print(f"  Running {len(benchmarks)} challenges (k={n_samples})...\n")
    results = []
    for i, ch in enumerate(benchmarks):
        t0 = time.time()
        responses = generate_response(model, tokenizer, ch["system_prompt"], ch["prompt"], n_samples=n_samples)
        elapsed = time.time() - t0

        # Grade all samples, track which passed
        any_correct = False
        first_correct = False
        feedbacks = []
        first_score = 0.0
        any_score = 0.0
        for j, resp in enumerate(responses):
            score, fb = grade_with_subtasks(resp, ch)
            correct = score >= 1.0
            feedbacks.append(fb)
            if j == 0:
                first_score = score
                first_correct = correct
            if correct:
                any_correct = True
                any_score = max(any_score, score)

        cheated = any(m.lower() in responses[0].lower() for m in SUSPICIOUS_MARKERS)

        results.append({
            "id": ch["id"],
            "category": ch["category"],
            "difficulty": ch["difficulty"],
            "task_type": ch["task_type"],
            "correct": first_correct,
            "score": first_score,
            "pass_at_k": any_correct,
            "score_at_k": any_score,
            "feedback": feedbacks[0] if feedbacks else "",
            "response": responses[0],
            "all_responses": responses if n_samples > 1 else [],
            "expected": ch["expected"],
            "time": elapsed,
            "suspicious_memorization": cheated,
        })

        mark = "✓" if first_correct else ("~" if any_correct else "✗")
        print(f"  [{i+1}/{len(benchmarks)}] {mark} {ch['id']} ({ch['difficulty']}) {elapsed:.1f}s")

    return {
        "model": model_key,
        "adapter": adapter_path,
        "results": results,
        "n_samples": n_samples,
    }


def print_results(eval_result: dict):
    """Print formatted evaluation results with Wilson 95% CI."""
    results = eval_result["results"]
    model_key = eval_result["model"]

    # Group by category-difficulty bucket
    buckets = defaultdict(lambda: {"correct": 0, "total": 0, "times": [], "score_sum": 0.0})
    for r in results:
        key = f"{r['category']}-{r['difficulty']}"
        buckets[key]["total"] += 1
        if r["correct"]:
            buckets[key]["correct"] += 1
        buckets[key]["score_sum"] += r.get("score", float(r["correct"]))
        buckets[key]["times"].append(r["time"])

    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    total_score = sum(r.get("score", float(r["correct"])) for r in results)
    mean_lat = sum(r["time"] for r in results) / total if total else 0

    # Pass@k stats
    n_samples = eval_result.get("n_samples", 1)
    pass_at_k = sum(1 for r in results if r.get("pass_at_k", r["correct"]))

    print(f"\n{'='*70}")
    print(f"CTF Evaluation Results")
    print(f"{'='*70}")
    print(f"Model:   {model_key}")
    print(f"Dataset: {total} questions (k={n_samples})")
    if n_samples > 1:
        print(f"pass@1:  {total_correct}/{total} ({total_correct/total:.0%})")
        print(f"pass@{n_samples}: {pass_at_k}/{total} ({pass_at_k/total:.0%})")
    print()

    # Bucket breakdown with Wilson CI
    header = f"{'Bucket':<16} {'acc':>5} {'score':>6} {'n':>4} {'CI95_lo':>7} {'CI95_hi':>7} {'mean_lat':>9}"
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
            avg_score = b["score_sum"] / b["total"]
            lo, hi = wilson_ci(b["correct"], b["total"])
            avg_t = sum(b["times"]) / len(b["times"])
            print(f"{key:<16} {acc:>4.0%} {avg_score:>5.2f} {b['total']:>4} {lo:>6.0%} {hi:>6.0%} {avg_t:>7.1f}s")

    # Overall
    lo, hi = wilson_ci(total_correct, total)
    avg_score = total_score / total if total else 0
    print("-" * len(header))
    print(f"{'overall':<16} {total_correct/total:>4.0%} {avg_score:>5.2f} {total:>4} {lo:>6.0%} {hi:>6.0%} {mean_lat:>7.1f}s")

    # Per-difficulty overall rows
    for d in difficulties:
        d_res = [r for r in results if r["difficulty"] == d]
        if not d_res:
            continue
        d_cor = sum(1 for r in d_res if r["correct"])
        d_score = sum(r.get("score", float(r["correct"])) for r in d_res) / len(d_res)
        lo, hi = wilson_ci(d_cor, len(d_res))
        d_lat = sum(r["time"] for r in d_res) / len(d_res)
        print(f"{'Overall ' + d:<16} {d_cor/len(d_res):>4.0%} {d_score:>5.2f} {len(d_res):>4} {lo:>6.0%} {hi:>6.0%} {d_lat:>7.1f}s")
    print()

    # Per-question breakdown
    print("Per-question breakdown:")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        score = r.get("score", float(r["correct"]))
        print(f"  {mark} {r['id']:<12} ({r['difficulty']:<6}) score={score:.2f} {r.get('feedback', '')[:50]}")

    # Length-bias probe
    correct_lens = [len(r["response"]) for r in results if r["correct"]]
    wrong_lens = [len(r["response"]) for r in results if not r["correct"]]
    if correct_lens and wrong_lens:
        mean_c = sum(correct_lens) / len(correct_lens)
        mean_w = sum(wrong_lens) / len(wrong_lens)
        ratio = mean_c / mean_w if mean_w > 0 else float('inf')
        flag = " ⚠ length bias suspected" if ratio > 1.5 or ratio < 0.67 else " ✓ no obvious length bias"
        print(f"\nLength-bias probe:")
        print(f"  avg correct: {mean_c:.0f} chars | avg wrong: {mean_w:.0f} chars | ratio: {ratio:.2f}{flag}")

    # Cheating detection
    cheated = sum(1 for r in results if r.get("suspicious_memorization"))
    if cheated:
        print(f"  Suspicious memorization: {cheated}/{len(results)} ({cheated/len(results)*100:.0f}%)")

    # Difficulty-balanced accuracy
    all_buckets = [(r["category"], r["difficulty"]) for r in results]
    unique_buckets = list(set(all_buckets))
    bucket_accs = {}
    bucket_scores = {}
    for cat, diff in unique_buckets:
        bq = [r for r in results if r["category"] == cat and r["difficulty"] == diff]
        bucket_accs[(cat, diff)] = sum(1 for r in bq if r["correct"]) / len(bq)
        bucket_scores[(cat, diff)] = sum(r.get("score", float(r["correct"])) for r in bq) / len(bq)
    balanced = sum(bucket_accs.values()) / len(bucket_accs) if bucket_accs else 0
    balanced_score = sum(bucket_scores.values()) / len(bucket_scores) if bucket_scores else 0
    print(f"  Balanced mean acc across {len(bucket_accs)} buckets: {balanced:.0%}")
    print(f"  Balanced mean score across {len(bucket_scores)} buckets: {balanced_score:.2f}")


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
        header += f" {name:<25}"
    print(header)
    print("-" * len(header))

    # Per-category
    for cat in categories:
        row = f"{cat:<10}"
        for er in eval_results:
            cat_results = [r for r in er["results"] if r["category"] == cat]
            correct = sum(1 for r in cat_results if r["correct"])
            total = len(cat_results)
            total_score = sum(r.get("score", float(r["correct"])) for r in cat_results)
            avg_score = total_score / total if total else 0
            lo, hi = wilson_ci(correct, total)
            row += f" {correct}/{total} (s={avg_score:.2f}, {lo:.0%}-{hi:.0%}){'':<2}"
        print(row)

    # Overall
    print("-" * len(header))
    row = f"{'Overall':<10}"
    for er in eval_results:
        correct = sum(1 for r in er["results"] if r["correct"])
        total = len(er["results"])
        total_score = sum(r.get("score", float(r["correct"])) for r in er["results"])
        avg_score = total_score / total if total else 0
        lo, hi = wilson_ci(correct, total)
        row += f" {correct}/{total} (s={avg_score:.2f}, {lo:.0%}-{hi:.0%}){'':<2}"
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
            p_value = math.erfc(math.sqrt(chi2 / 2))  # chi-squared survival, 1 df
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
        total_score = sum(r.get("score", float(r["correct"])) for r in er["results"])
        avg_score = total_score / total if total else 0
        lo, hi = wilson_ci(correct, total)
        cheated = sum(1 for r in er["results"] if r.get("suspicious_memorization"))
        data["results"].append({
            "model": er["model"],
            "adapter": er["adapter"],
            "accuracy": correct / total if total else 0,
            "mean_score": avg_score,
            "wilson_ci95": [lo, hi],
            "correct": correct,
            "total": total,
            "suspicious_memorization": cheated,
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
    parser.add_argument("--samples", type=int, default=1,
                        help="Samples per prompt for pass@k (recommended: 3)")

    args = parser.parse_args()

    if args.compare:
        eval_results = []
        for spec in args.compare:
            model_key, adapter = spec.split(":", 1)
            result = run_evaluation(model_key, adapter, args.bench, args.category, args.difficulty, args.samples)
            eval_results.append(result)
        print_comparison(eval_results)
    elif args.model:
        result = run_evaluation(args.model, args.adapter or "", args.bench, args.category, args.difficulty, args.samples)
        eval_results = [result]
        print_results(result)
    else:
        parser.print_help()
        return

    if args.output:
        save_results(eval_results, args.output)


if __name__ == "__main__":
    main()

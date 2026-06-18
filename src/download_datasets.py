"""
Dataset download and preparation script
Downloads and converts datasets to Alpaca format for Unsloth training
"""
import json
import argparse
import time
from pathlib import Path
from datasets import load_dataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def load_hf_with_fallback(name):
    """Try train split, then test, then default."""
    for split in ["train", "test"]:
        try:
            return load_dataset(name, split=split)
        except Exception:
            continue
    return load_dataset(name)


def extract_qa(item):
    """Extract question/answer from various HF dataset schemas."""
    msgs = item.get("messages")
    if msgs and len(msgs) >= 2:
        q = msgs[0].get("content", "")
        a = msgs[-1].get("content", "")
        if q and a:
            return q, a
    q = item.get("question") or item.get("problem") or item.get("description") or ""
    a = item.get("answer") or item.get("solution") or item.get("flag") or ""
    if q and a:
        return q, a
    text = item.get("text_chunk", "")
    if text:
        return "Explain this CTF writeup and provide the solution", text
    return None, None


def download_ctf_webserver(output_path="data/ctf_webserver.jsonl"):
    """Download Jacqkues/ctf_webserver_v0.1 (web CTF challenges)"""
    print("\n  [1/5] Downloading ctf_webserver_v0.1...")
    start_time = time.time()
    ds = load_dataset("Jacqkues/ctf_webserver_v0.1", split="train")

    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, desc="    ctf_webserver", unit="row")

    with open(output_path, "w") as f:
        for item in data_iter:
            messages = item.get("messages", [])
            if len(messages) >= 2:
                instruction = messages[0].get("content", "")
                output = messages[-1].get("content", "")
                f.write(json.dumps({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                }) + "\n")
                count += 1

    elapsed = time.time() - start_time
    print(f"  ✓ ctf_webserver: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_opencode_reasoning(output_path="data/opencode_reasoning.jsonl", max_samples=10000):
    """Download nvidia/OpenCodeReasoning (competitive programming)"""
    print("\n  [2/4] Downloading nvidia/OpenCodeReasoning...")
    start_time = time.time()
    ds = load_dataset("nvidia/OpenCodeReasoning", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    opencode-reasoning", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            problem = item.get("input", "")
            solution = item.get("solution", "")
            reasoning = item.get("output", "")
            
            if problem and solution:
                f.write(json.dumps({
                    "instruction": f"Solve this competitive programming problem:\n{problem}",
                    "input": "",
                    "output": f"{reasoning}\n\n```python\n{solution}\n```"
                }) + "\n")
                count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ opencode-reasoning: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_fenrir(output_path="data/fenrir_cybersecurity.jsonl", max_samples=10000):
    """Download AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1"""
    print("\n  [3/4] Downloading Cybersecurity Dataset Fenrir v2.1...")
    start_time = time.time()
    ds = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    fenrir", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            messages = item.get("messages", [])
            if len(messages) >= 2:
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                if user_msg and assistant_msg:
                    f.write(json.dumps({
                        "instruction": user_msg,
                        "input": "",
                        "output": assistant_msg
                    }) + "\n")
                    count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ fenrir: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_vulnerability(output_path="data/vulnerability_detection.jsonl", max_samples=10000):
    """Download ayshajavd/code-security-vulnerability-dataset"""
    print("\n  [4/4] Downloading code-security-vulnerability-dataset...")
    start_time = time.time()
    ds = load_dataset("ayshajavd/code-security-vulnerability-dataset", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    vulnerability", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            code = item.get("code", "")
            vulnerability = item.get("vulnerability", "")
            fix = item.get("fix", "")
            cwe = item.get("cwe", "")
            
            if code and vulnerability:
                f.write(json.dumps({
                    "instruction": f"Identify and fix the security vulnerability in this code (CWE: {cwe}):",
                    "input": code[:2000],
                    "output": f"Vulnerability: {vulnerability}\n\nFixed code:\n{fix}"
                }) + "\n")
                count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ vulnerability: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_ctf_solver(output_path="data/raw/ctf_solver.jsonl", max_samples=10000):
    """Download TrueNix/ctf-solver-dataset (CTF solver with solutions)"""
    print("\n  [NEW] Downloading ctf-solver-dataset...")
    start_time = time.time()
    ds = load_dataset("TrueNix/ctf-solver-dataset", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    ctf-solver", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            msgs = item.get("messages", [])
            if len(msgs) >= 2:
                instruction = msgs[0].get("content", "")
                output = msgs[-1].get("content", "")
                if instruction and output:
                    f.write(json.dumps({
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    }) + "\n")
                    count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ ctf-solver: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_summermc_ctf(output_path="data/raw/summermc_ctf.jsonl", max_samples=10000):
    """Download SummerMC/CTF (structured CTF agent training data)"""
    print("\n  [NEW] Downloading SummerMC/CTF...")
    start_time = time.time()
    ds = load_dataset("SummerMC/CTF", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    summermc-ctf", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            task = item.get("task", "")
            ground_truth = item.get("ground_truth", "")
            if task and ground_truth:
                f.write(json.dumps({
                    "instruction": task[:3000],
                    "input": "",
                    "output": str(ground_truth)[:5000]
                }) + "\n")
                count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ summermc-ctf: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_trendyol_cybersec(output_path="data/raw/trendyol_cybersec.jsonl", max_samples=10000):
    """Download Trendyol cybersecurity instruction tuning dataset"""
    print("\n  [NEW] Downloading Trendyol-Cybersecurity...")
    start_time = time.time()
    ds = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    trendyol-cyber", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            user = item.get("user", "")
            assistant = item.get("assistant", "")
            if user and assistant:
                f.write(json.dumps({
                    "instruction": user,
                    "input": "",
                    "output": assistant
                }) + "\n")
                count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ trendyol-cyber: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_ctf_crypto_analysis(output_path="data/raw/ctf_crypto_analysis.jsonl", max_samples=5000):
    """Download Sakana-ctf ctf-crypto-manual-analysis-benchmark"""
    print("\n  [NEW] Downloading ctf-crypto-analysis...")
    start_time = time.time()
    ds = load_dataset("Sakana-ctf/ctf-crypto-manual-analysis-benchmark", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    ctf-crypto", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            msgs = item.get("messages", [])
            if len(msgs) >= 2:
                instruction = msgs[0].get("content", "")
                output = msgs[-1].get("content", "")
                if instruction and output:
                    f.write(json.dumps({
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    }) + "\n")
                    count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ ctf-crypto-analysis: {count} examples ({elapsed:.1f}s)")
    return output_path


def merge_datasets(input_files, output_path="data/merged_train.jsonl"):
    """Merge multiple JSONL files into one"""
    total = 0
    with open(output_path, "w") as f:
        for input_file in input_files:
            if Path(input_file).exists():
                with open(input_file) as inf:
                    for line in inf:
                        f.write(line)
                        total += 1
    
    print(f"Merged {total} examples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download and prepare CTF/coding datasets")
    parser.add_argument("--dataset", choices=[
        "ctf-webserver", "opencode-reasoning", "fenrir", "ctf-solver", "summermc-ctf", "trendyol-cyber", "ctf-crypto-analysis", "all", "merge"
    ], required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    
    start_time = time.time()
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"  Dataset Downloader - dataset: {args.dataset}")
    print(f"  Max samples: {args.max_samples}")
    print(f"{'='*50}")
    
    if args.dataset == "ctf-webserver":
        download_ctf_webserver(f"{args.output_dir}/ctf_webserver.jsonl")
    elif args.dataset == "opencode-reasoning":
        download_opencode_reasoning(f"{args.output_dir}/opencode_reasoning.jsonl", args.max_samples)
    elif args.dataset == "fenrir":
        download_fenrir(f"{args.output_dir}/fenrir_cybersecurity.jsonl", args.max_samples)
    elif args.dataset == "vulnerability":
        download_vulnerability(f"{args.output_dir}/vulnerability_detection.jsonl", args.max_samples)
    elif args.dataset == "all":
        download_ctf_webserver(f"{args.output_dir}/ctf_webserver.jsonl")
        download_opencode_reasoning(f"{args.output_dir}/opencode_reasoning.jsonl", args.max_samples)
        download_fenrir(f"{args.output_dir}/fenrir_cybersecurity.jsonl", args.max_samples)
        download_ctf_solver(f"{args.output_dir}/ctf_solver.jsonl", args.max_samples)
        download_summermc_ctf(f"{args.output_dir}/summermc_ctf.jsonl", args.max_samples)
        download_trendyol_cybersec(f"{args.output_dir}/trendyol_cybersec.jsonl", args.max_samples)
        download_ctf_crypto_analysis(f"{args.output_dir}/ctf_crypto_analysis.jsonl", args.max_samples)
    elif args.dataset == "merge":
        print("\n=== Merging datasets ===")
        files = [
            f"{args.output_dir}/ctf_webserver.jsonl",
            f"{args.output_dir}/opencode_reasoning.jsonl",
            f"{args.output_dir}/fenrir_cybersecurity.jsonl",
            f"{args.output_dir}/ctf_solver.jsonl",
            f"{args.output_dir}/summermc_ctf.jsonl",
            f"{args.output_dir}/trendyol_cybersec.jsonl",
            f"{args.output_dir}/ctf_crypto_analysis.jsonl",
        ]
        merge_datasets(files, f"{args.output_dir}/merged_train.jsonl")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

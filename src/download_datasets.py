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


def download_ctftime(output_path="data/ctftime.jsonl", max_samples=5000):
    """Download justinwangx/CTFtime (CTF writeups)"""
    print("\n  [2/5] Downloading CTFtime writeups...")
    start_time = time.time()
    ds = load_dataset("justinwangx/CTFtime", split="train")
    
    count = 0
    data_iter = ds
    if HAS_TQDM:
        data_iter = tqdm(ds, total=min(len(ds), max_samples), desc="    ctftime", unit="row")
    
    with open(output_path, "w") as f:
        for item in data_iter:
            if count >= max_samples:
                break
            text = item.get("text_chunk", "")
            if len(text) > 100:
                f.write(json.dumps({
                    "instruction": "Explain this CTF challenge solution and provide the exploit code:",
                    "input": text[:2000],
                    "output": "See the analysis above."
                }) + "\n")
                count += 1
    
    elapsed = time.time() - start_time
    print(f"  ✓ ctftime: {count} examples ({elapsed:.1f}s)")
    return output_path


def download_opencode_reasoning(output_path="data/opencode_reasoning.jsonl", max_samples=10000):
    """Download nvidia/OpenCodeReasoning (competitive programming)"""
    print("\n  [3/5] Downloading nvidia/OpenCodeReasoning...")
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
    print("\n  [4/5] Downloading Cybersecurity Dataset Fenrir v2.1...")
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
    print("\n  [5/5] Downloading code-security-vulnerability-dataset...")
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
        "ctf-webserver", "ctftime", "opencode-reasoning", 
        "fenrir", "vulnerability", "all", "merge"
    ], required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    
    start_time = time.time()
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"  Dataset Downloader - dataset: {args.dataset}")
    print(f"  Max samples: {args.max_samples}")
    print(f"{'='*50}")
    
    if args.dataset == "ctf-webserver":
        download_ctf_webserver(f"{args.output_dir}/ctf_webserver.jsonl")
    elif args.dataset == "ctftime":
        download_ctftime(f"{args.output_dir}/ctftime.jsonl", args.max_samples)
    elif args.dataset == "opencode-reasoning":
        download_opencode_reasoning(f"{args.output_dir}/opencode_reasoning.jsonl", args.max_samples)
    elif args.dataset == "fenrir":
        download_fenrir(f"{args.output_dir}/fenrir_cybersecurity.jsonl", args.max_samples)
    elif args.dataset == "vulnerability":
        download_vulnerability(f"{args.output_dir}/vulnerability_detection.jsonl", args.max_samples)
    elif args.dataset == "all":
        download_ctf_webserver(f"{args.output_dir}/ctf_webserver.jsonl")
        download_ctftime(f"{args.output_dir}/ctftime.jsonl", args.max_samples)
        download_opencode_reasoning(f"{args.output_dir}/opencode_reasoning.jsonl", args.max_samples)
        download_fenrir(f"{args.output_dir}/fenrir_cybersecurity.jsonl", args.max_samples)
        download_vulnerability(f"{args.output_dir}/vulnerability_detection.jsonl", args.max_samples)
    elif args.dataset == "merge":
        print("\n=== Merging datasets ===")
        files = [
            f"{args.output_dir}/ctf_webserver.jsonl",
            f"{args.output_dir}/ctftime.jsonl",
            f"{args.output_dir}/opencode_reasoning.jsonl",
            f"{args.output_dir}/fenrir_cybersecurity.jsonl",
            f"{args.output_dir}/vulnerability_detection.jsonl",
        ]
        merge_datasets(files, f"{args.output_dir}/merged_train.jsonl")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

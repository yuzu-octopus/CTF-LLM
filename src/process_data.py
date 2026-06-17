"""
Data Processing Pipeline - Convert datasets to Unsloth-compatible format
Usage:
  uv run src/process_data.py --input data/raw --output data/processed
  uv run src/process_data.py --merge --input data/processed --output data/merged
"""
import json
import argparse
import time
from pathlib import Path
from typing import Iterator

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


SYSTEM_PROMPT_CTF = """You are an expert CTF (Capture The Flag) player and security researcher. You specialize in:
- Binary exploitation (pwn): buffer overflows, ROP, heap exploitation, format strings
- Reverse engineering: analyzing binaries, deobfuscation, decompilation
- Web exploitation: SQL injection, XSS, SSRF, deserialization, JWT attacks
- Cryptography: cryptanalysis, key recovery, side-channel attacks
- Forensics: memory analysis, network capture analysis, steganography

When solving challenges:
1. Analyze the problem systematically
2. Identify the vulnerability or attack vector
3. Provide step-by-step solution with code
4. Explain the underlying concepts"""

SYSTEM_PROMPT_CODING = """You are an expert competitive programmer and software engineer. You specialize in:
- Algorithm design and analysis
- Data structures optimization
- Code optimization and debugging
- Security-aware coding practices

When solving problems:
1. Understand the problem requirements
2. Identify the optimal approach
3. Write clean, efficient code
4. Explain your reasoning"""


def convert_alpaca_to_chat(instruction: str, input_text: str, output: str, category: str = "") -> dict:
    """Convert Alpaca format to chat format"""
    # Determine system prompt based on category
    if any(kw in category.lower() for kw in ["pwn", "rev", "web", "crypto", "ctf", "exploit"]):
        system_prompt = SYSTEM_PROMPT_CTF
    else:
        system_prompt = SYSTEM_PROMPT_CODING
    
    # Build user message
    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
    }


def convert_writeup_to_chat(text_chunk: str, category: str = "") -> dict:
    """Convert raw writeup text to chat format"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_CTF},
            {"role": "user", "content": f"Explain this CTF challenge solution:\n\n{text_chunk[:3000]}"},
            {"role": "assistant", "content": f"Here's the analysis and solution:\n\n{text_chunk[:2000]}"}
        ]
    }


def process_jsonl_file(input_path: Path, output_path: Path) -> int:
    """Process a JSONL file and convert to chat format"""
    count = 0
    total_lines = sum(1 for _ in open(input_path))
    start_time = time.time()
    
    line_iter = open(input_path)
    if HAS_TQDM:
        line_iter = tqdm(open(input_path), total=total_lines, desc=f"    {input_path.name}", unit="line")
    
    with open(output_path, "w") as fout:
        for line in line_iter:
            try:
                item = json.loads(line)
                
                if "messages" in item:
                    chat_data = item
                elif "instruction" in item:
                    chat_data = convert_alpaca_to_chat(
                        item.get("instruction", ""),
                        item.get("input", ""),
                        item.get("output", ""),
                        item.get("category", "")
                    )
                elif "text_chunk" in item:
                    chat_data = convert_writeup_to_chat(
                        item["text_chunk"],
                        item.get("category", "")
                    )
                else:
                    continue
                
                fout.write(json.dumps(chat_data) + "\n")
                count += 1
            except Exception as e:
                continue
    
    elapsed = time.time() - start_time
    print(f"    ✓ {input_path.name}: {count}/{total_lines} examples ({elapsed:.1f}s)")
    return count


def process_directory(input_dir: Path, output_dir: Path) -> int:
    """Process all JSONL files in a directory"""
    total = 0
    start_time = time.time()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob("*.jsonl"))
    total_files = len(files)
    
    print(f"\n  Found {total_files} JSONL files to process\n")
    
    for file_idx, input_file in enumerate(files):
        print(f"  [{file_idx + 1}/{total_files}] Processing {input_file.name}...")
        output_file = output_dir / input_file.name
        count = process_jsonl_file(input_file, output_file)
        total += count
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Processed {total_files} files: {total} total examples ({elapsed:.1f}s)")
    return total


def merge_jsonl_files(input_dir: Path, output_path: Path) -> int:
    """Merge all JSONL files into one"""
    total = 0
    
    with open(output_path, "w") as fout:
        for input_file in sorted(input_dir.glob("*.jsonl")):
            with open(input_file) as fin:
                for line in fin:
                    fout.write(line)
                    total += 1
    
    return total


def main():
    parser = argparse.ArgumentParser(description="Process datasets for Unsloth training")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--merge", action="store_true", help="Merge processed files")
    parser.add_argument("--system-prompt", choices=["ctf", "coding", "auto"], default="auto")
    args = parser.parse_args()
    
    start_time = time.time()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print(f"\n{'='*50}")
    print(f"  Data Processor - mode: {'merge' if args.merge else 'process'}")
    print(f"{'='*50}")
    
    if args.merge:
        print("\n=== Merging processed datasets ===")
        merged_dir = Path("data/merged")
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        total = merge_jsonl_files(output_dir, merged_dir / "train.jsonl")
        print(f"  ✓ Merged {total} examples to data/merged/train.jsonl")
    else:
        print(f"\n=== Processing datasets ===")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        
        total = process_directory(input_dir, output_dir)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

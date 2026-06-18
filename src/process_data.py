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


try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


SYSTEM_PROMPT_CTF = (
    "Expert CTF player. Specialties: pwn, rev, web, crypto, forensics. "
    "Always reason step-by-step before exploit code."
)

SYSTEM_PROMPT_CODING = (
    "Expert competitive programmer. Optimize first; explain after. "
    "Security-aware coding."
)


CTF_KEYWORDS = ["pwn", "rev", "web", "crypto", "ctf", "exploit", "vuln", "shellcode"]


def is_ctf_content(text):
    """Check if text contains CTF-related keywords."""
    return any(k in text.lower() for k in CTF_KEYWORDS)


def convert_alpaca_to_chat(instruction: str, input_text: str, output: str, category: str = "", system_prompt_mode: str = "auto", skip_system_prompt: bool = False) -> dict:
    """Convert Alpaca format to chat format"""
    # Build user message
    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"

    if skip_system_prompt:
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
        }

    # Determine system prompt based on mode
    if system_prompt_mode == "ctf":
        system_prompt = SYSTEM_PROMPT_CTF
    elif system_prompt_mode == "coding":
        system_prompt = SYSTEM_PROMPT_CODING
    else:
        # auto: detect from category
        if is_ctf_content(category):
            system_prompt = SYSTEM_PROMPT_CTF
        else:
            system_prompt = SYSTEM_PROMPT_CODING

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
    }


def process_jsonl_file(input_path: Path, output_path: Path, system_prompt_mode: str = "auto", skip_system_prompt: bool = False) -> int:
    """Process a JSONL file and convert to chat format"""
    count = 0
    with open(input_path) as f:
        total_lines = sum(1 for _ in f)
    start_time = time.time()

    with open(input_path) as line_iter:
        if HAS_TQDM:
            line_iter = tqdm(line_iter, total=total_lines, desc=f"    {input_path.name}", unit="line")

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
                            item.get("category", ""),
                            system_prompt_mode=system_prompt_mode,
                            skip_system_prompt=skip_system_prompt,
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


def process_directory(input_dir: Path, output_dir: Path, system_prompt_mode: str = "auto", skip_system_prompt: bool = False) -> int:
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
        count = process_jsonl_file(input_file, output_file, system_prompt_mode=system_prompt_mode, skip_system_prompt=skip_system_prompt)
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
    parser.add_argument("--no-system-prompt", action="store_true", help="Omit system role from output messages. Default: ON (system prompt written).")
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
        
        total = merge_jsonl_files(input_dir, merged_dir / "train.jsonl")
        print(f"  ✓ Merged {total} examples to data/merged/train.jsonl")
    else:
        print(f"\n=== Processing datasets ===")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  System prompt mode: {args.system_prompt}")
        if args.no_system_prompt:
            print(f"  System prompts: SKIPPED (set at training time)")

        total = process_directory(input_dir, output_dir, system_prompt_mode=args.system_prompt, skip_system_prompt=args.no_system_prompt)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

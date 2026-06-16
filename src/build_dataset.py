"""
Custom Dataset Builder - Scrapes GitHub repos and docs, converts to Alpaca format
Usage:
  uv run src/build_dataset.py --source writeups
  uv run src/build_dataset.py --source docs
  uv run src/build_dataset.py --source all
"""
import json
import re
import argparse
from pathlib import Path
import tempfile

import requests
from git import Repo

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed. HuggingFace datasets will be skipped.")


# GitHub repos to scrape for CTF writeups
CTF_WRITEUP_REPOS = [
    # picoCTF
    {"url": "https://github.com/Cajac/picoCTF-Writeups", "name": "picoctf-cajac", "category": "picoctf"},
    {"url": "https://github.com/vivian-dai/PicoCTF2021-Writeup", "name": "picoctf-2021", "category": "picoctf"},
    
    # CryptoHack
    {"url": "https://github.com/DarkCodeOrg/CryptoHack", "name": "cryptohack-darkcode", "category": "cryptohack"},
    {"url": "https://github.com/AyushSingh-c/Cryptohack", "name": "cryptohack-ayush", "category": "cryptohack"},
    
    # pwnCollege
    {"url": "https://github.com/H3xKatana/pwncollege-writeups", "name": "pwncollege-h3x", "category": "pwncollege"},
    {"url": "https://github.com/prettyb0iisam/pwncollege-writeups", "name": "pwncollege-prettyb0i", "category": "pwncollege"},
    {"url": "https://github.com/id-none/pwncollege_writeup", "name": "pwncollege-idnone", "category": "pwncollege"},
    
    # Multi-category CTF
    {"url": "https://github.com/Adamkadaban/CTFs", "name": "ctfs-adamkadaban", "category": "multi"},
    {"url": "https://github.com/Cryptogenic/Exploit-Writeups", "name": "exploit-writeups", "category": "pwn"},
    {"url": "https://github.com/ffffffff0x/1earn", "name": "0x1earn", "category": "multi"},
]

# Documentation repos/files to scrape
DOC_SOURCES = [
    # pwntools
    {"type": "github_docs", "url": "https://github.com/Gallopsled/pwntools", "path": "docs", "name": "pwntools-docs"},
    {"type": "github_file", "url": "https://raw.githubusercontent.com/Gallopsled/pwntools-write-ups/master/README.md", "name": "pwntools-writeups-readme"},
    
    # angr
    {"type": "github_docs", "url": "https://github.com/angr/angr-doc", "path": "docs", "name": "angr-docs"},
    
    # z3
    {"type": "github_docs", "url": "https://github.com/Z3Prover/z3", "path": "doc", "name": "z3-docs"},
    
    # pwndbg
    {"type": "github_file", "url": "https://raw.githubusercontent.com/pwndbg/pwndbg/dev/FEATURES.md", "name": "pwndbg-features"},
    
    # ROP gadgets
    {"type": "github_readme", "url": "https://github.com/JonathanSalwan/ROPgadget", "name": "ropgadget-readme"},
    {"type": "github_readme", "url": "https://github.com/sashs/Ropper", "name": "ropper-readme"},
    
    # One gadget
    {"type": "github_readme", "url": "https://github.com/david942j/one_gadget", "name": "onegadget-readme"},
    
    # Math/crypto libraries
    {"type": "github_docs", "url": "https://github.com/sympy/sympy", "path": "docs", "name": "sympy-docs"},
    {"type": "github_readme", "url": "https://github.com/aleaxit/gmpy2", "name": "gmpy2-readme"},
    {"type": "github_docs", "url": "https://github.com/sagemath/sage", "path": "doc", "name": "sagemath-docs"},
]


def clone_repo(url: str, dest: str) -> bool:
    """Clone a git repo using gitpython"""
    try:
        Repo.clone_from(url, dest, depth=1)
        return Path(dest).exists()
    except Exception as e:
        print(f"  Failed to clone {url}: {e}")
        return False


def extract_code_blocks(content: str) -> list:
    """Extract code blocks from markdown content"""
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    code_blocks = []
    for lang, code in matches:
        code = code.strip()
        if len(code) > 10:  # Skip trivial blocks
            code_blocks.append({"lang": lang or "text", "code": code})
    return code_blocks


def extract_challenge_description(content: str) -> str:
    """Extract challenge description from writeup markdown"""
    # Remove code blocks for description extraction
    cleaned = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    # Take first ~1500 chars as description
    if len(cleaned) > 1500:
        cleaned = cleaned[:1500] + "..."
    return cleaned


def extract_writeups_from_repo(repo_path: str, category: str) -> list:
    """Extract writeups from a cloned repo (markdown + code)"""
    examples = []
    repo_path = Path(repo_path)
    
    # Find all markdown files
    md_files = list(repo_path.rglob("*.md")) + list(repo_path.rglob("*.MD"))
    
    # Instruction templates based on content type
    INSTRUCTION_TEMPLATES = [
        "Analyze this binary for vulnerabilities and provide an exploit",
        "Explain this CTF challenge and solve it step by step",
        "What is the vulnerability in this code? Write an exploit.",
        "Reverse engineer this challenge and find the flag",
        "Analyze the cryptography used and break the encryption",
        "Write a solution script for this challenge",
        "Identify the security flaw and demonstrate exploitation",
    ]
    
    for md_file in md_files:
        try:
            content = md_file.read_text(errors='ignore')
            if len(content) < 100:  # Skip very short files
                continue
            
            # Extract code blocks from markdown
            code_blocks = extract_code_blocks(content)
            
            # Also try to find associated code files
            parent = md_file.parent
            code_files = list(parent.glob("*.py")) + list(parent.glob("*.c")) + list(parent.glob("*.sh"))
            for cf in code_files[:3]:  # Max 3 code files
                try:
                    code_content = cf.read_text(errors='ignore')
                    if len(code_content) > 10:
                        code_blocks.append({"lang": cf.suffix[1:] or "text", "code": code_content})
                except:
                    pass
            
            # Extract challenge name from filename or path
            challenge_name = md_file.stem.replace("-", " ").replace("_", " ").title()
            
            # Extract challenge description (text without code)
            description = extract_challenge_description(content)
            
            # Build output from code blocks
            if code_blocks:
                output_parts = [f"## Challenge: {challenge_name}\n"]
                output_parts.append(f"Category: {category}\n")
                output_parts.append(f"### Description\n{description}\n")
                output_parts.append("### Solution\n")
                for i, block in enumerate(code_blocks[:3]):  # Max 3 blocks
                    lang = block["lang"]
                    code = block["code"][:2000]
                    output_parts.append(f"```{lang}\n{code}\n```\n")
                output_text = "\n".join(output_parts)
            else:
                # No code found - create summary-based output
                output_text = f"## Challenge: {challenge_name}\nCategory: {category}\n\n{description}\n"
            
            # Select instruction based on content hints
            lower_content = content.lower()
            if any(kw in lower_content for kw in ["binary", "pwn", "buffer overflow", "heap", "stack"]):
                instruction = "Analyze this binary for vulnerabilities and provide an exploit"
            elif any(kw in lower_content for kw in ["crypto", "rsa", "aes", "encrypt", "decrypt"]):
                instruction = "Analyze the cryptography used and break the encryption"
            elif any(kw in lower_content for kw in ["reverse", "flag", "hidden", "obfuscated"]):
                instruction = "Reverse engineer this challenge and find the flag"
            elif code_blocks:
                instruction = "Write a solution script for this challenge"
            else:
                instruction = "Explain this CTF challenge and solve it step by step"
            
            examples.append({
                "instruction": instruction,
                "input": f"Challenge: {challenge_name}\nCategory: {category}\n\n{description}",
                "output": output_text
            })
        except Exception as e:
            continue
    
    return examples


def extract_from_huggingface(dataset_name: str, max_samples: int = 5000) -> list:
    """Extract and convert a HuggingFace dataset to Alpaca format"""
    examples = []
    
    if not HAS_DATASETS:
        print(f"  Skipping {dataset_name} (datasets library not installed)")
        return examples
    
    try:
        ds = load_dataset(dataset_name, split="train")
        
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            
            # Try common field names
            question = item.get("question", item.get("problem", item.get("description", "")))
            answer = item.get("answer", item.get("solution", item.get("flag", "")))
            category = item.get("category", item.get("type", ""))
            
            if question and answer:
                examples.append({
                    "instruction": question,
                    "input": "",
                    "output": f"Category: {category}\n\n{answer}" if category else answer
                })
    except Exception as e:
        print(f"  Failed to load {dataset_name}: {e}")
    
    return examples


def scrape_documentation(url: str, name: str) -> list:
    """Scrape a single documentation file using requests"""
    examples = []
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        if len(content) < 100:
            return examples
        
        # Split into sections (heuristic: look for ## headers)
        sections = re.split(r'\n(?=## )', content)
        
        for section in sections:
            if len(section) < 50:
                continue
            
            # Extract section title
            title_match = re.match(r'##\s+(.+)', section)
            title = title_match.group(1) if title_match else "Documentation"
            
            examples.append({
                "instruction": f"Explain how to use {name} for: {title}",
                "input": section[:2000],
                "output": f"Based on {name} documentation:\n\n{section[:2000]}"
            })
    except Exception as e:
        print(f"  Failed to scrape {url}: {e}")
    
    return examples


def build_writeups_dataset(output_path: str, max_per_repo: int = 500):
    """Build dataset from CTF writeup repos"""
    all_examples = []
    
    print("=== Building CTF Writeups Dataset ===\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for repo in CTF_WRITEUP_REPOS:
            print(f"Cloning {repo['name']}...")
            repo_path = f"{tmpdir}/{repo['name']}"
            
            if clone_repo(repo["url"], repo_path):
                examples = extract_writeups_from_repo(repo_path, repo["category"])
                examples = examples[:max_per_repo]  # Limit per repo
                all_examples.extend(examples)
                print(f"  Extracted {len(examples)} examples")
            else:
                print(f"  Skipped (clone failed)")
    
    # Also add HuggingFace datasets
    print("\nLoading HuggingFace datasets...")
    hf_datasets = [
        ("kyleavery/picoctf", 500),
        ("justinwangx/CTFtime", 2000),
    ]
    
    for ds_name, max_samples in hf_datasets:
        print(f"  Loading {ds_name}...")
        examples = extract_from_huggingface(ds_name, max_samples)
        all_examples.extend(examples)
        print(f"  Extracted {len(examples)} examples")
    
    # Write output
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_examples:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n=== Total: {len(all_examples)} examples saved to {output_path} ===")
    return all_examples


def build_docs_dataset(output_path: str, max_per_doc: int = 200):
    """Build dataset from documentation"""
    all_examples = []
    
    print("=== Building Documentation Dataset ===\n")
    
    for doc in DOC_SOURCES:
        print(f"Scraping {doc['name']}...")
        
        if doc["type"] == "github_file" or doc["type"] == "github_readme":
            examples = scrape_documentation(doc["url"], doc["name"])
        elif doc["type"] == "github_docs":
            # For docs directories, we'd need more complex scraping
            # For now, just get the README
            readme_url = doc["url"].replace("github.com", "raw.githubusercontent.com") + "/dev/README.md"
            examples = scrape_documentation(readme_url, doc["name"])
        else:
            examples = []
        
        examples = examples[:max_per_doc]
        all_examples.extend(examples)
        print(f"  Extracted {len(examples)} examples")
    
    # Write output
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_examples:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n=== Total: {len(all_examples)} examples saved to {output_path} ===")
    return all_examples


def merge_datasets(input_files: list, output_path: str):
    """Merge multiple JSONL files"""
    total = 0
    with open(output_path, "w") as f:
        for input_file in input_files:
            if Path(input_file).exists():
                with open(input_file) as inf:
                    for line in inf:
                        f.write(line)
                        total += 1
    
    print(f"Merged {total} examples to {output_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Build custom CTF/coding datasets")
    parser.add_argument("--source", choices=["writeups", "docs", "all", "merge"], required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-per-repo", type=int, default=500)
    parser.add_argument("--max-per-doc", type=int, default=200)
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.source == "writeups":
        build_writeups_dataset(f"{args.output_dir}/writeups.jsonl", args.max_per_repo)
    elif args.source == "docs":
        build_docs_dataset(f"{args.output_dir}/docs.jsonl", args.max_per_repo)
    elif args.source == "all":
        build_writeups_dataset(f"{args.output_dir}/writeups.jsonl", args.max_per_repo)
        build_docs_dataset(f"{args.output_dir}/docs.jsonl", args.max_per_repo)
    elif args.source == "merge":
        files = [
            f"{args.output_dir}/writeups.jsonl",
            f"{args.output_dir}/docs.jsonl",
            f"{args.output_dir}/ctf_webserver.jsonl",
            f"{args.output_dir}/ctftime.jsonl",
            f"{args.output_dir}/opencode_reasoning.jsonl",
            f"{args.output_dir}/fenrir_cybersecurity.jsonl",
            f"{args.output_dir}/vulnerability_detection.jsonl",
        ]
        merge_datasets(files, f"{args.output_dir}/merged_train.jsonl")
    
    print("Done!")


if __name__ == "__main__":
    main()

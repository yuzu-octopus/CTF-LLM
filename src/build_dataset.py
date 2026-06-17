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
import time
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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


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

# Documentation repos/files to scrape (fixed URLs)
DOC_SOURCES = [
    # pwntools
    {"type": "github_docs", "url": "https://github.com/Gallopsled/pwntools", "path": "docs", "name": "pwntools-docs"},
    {"type": "github_file", "url": "https://raw.githubusercontent.com/Gallopsled/pwntools-write-ups/master/README.md", "name": "pwntools-writeups-readme"},
    
    # angr (fixed: using main README)
    {"type": "github_readme", "url": "https://github.com/angr/angr", "name": "angr-main"},
    
    # z3 (fixed: using main README)
    {"type": "github_readme", "url": "https://github.com/Z3Prover/z3", "name": "z3-main"},
    
    # pwndbg (fixed: using main README)
    {"type": "github_readme", "url": "https://github.com/pwndbg/pwndbg", "name": "pwndbg-main"},
    
    # ROP gadgets
    {"type": "github_readme", "url": "https://github.com/JonathanSalwan/ROPgadget", "name": "ropgadget-readme"},
    {"type": "github_readme", "url": "https://github.com/sashs/Ropper", "name": "ropper-readme"},
    
    # One gadget
    {"type": "github_readme", "url": "https://github.com/david942j/one_gadget", "name": "onegadget-readme"},
    
    # Math/crypto libraries (fixed: using main READMEs)
    {"type": "github_readme", "url": "https://github.com/sympy/sympy", "name": "sympy-main"},
    {"type": "github_readme", "url": "https://github.com/gmpy2/gmpy2", "name": "gmpy2"},
    {"type": "github_readme", "url": "https://github.com/sagemath/sage", "name": "sagemath"},
    
    # Crypto-specific docs (raw GitHub files)
    {"type": "github_file", "url": "https://raw.githubusercontent.com/sympy/sympy/master/sympy/crypto/crypto.py", "name": "sympy-crypto-module"},
]


def clone_repo(url: str, dest: str) -> bool:
    """Clone a git repo using gitpython"""
    import shutil
    
    # Clean up existing directory if present
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    
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
        if len(code) > 10:
            code_blocks.append({"lang": lang or "text", "code": code})
    return code_blocks


def find_solution_boundary(content: str) -> int:
    """Find where the solution/writeup section starts"""
    solution_markers = [
        r'^##\s+(?:Solution|Writeup|Exploit|Answer|Flag|Solution:)',
        r'^###\s+(?:Solution|Writeup|Exploit|Answer|Flag|Solution:)',
        r'^##\s+(?:Solving|Solved|My Solution)',
        r'^###\s+(?:Solving|Solved|My Solution)',
        r'^\*\*(?:Solution|Exploit|Answer|Flag)\*\*',
        r'^>\s*(?:Solution|Exploit|Answer|Flag)',
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for marker in solution_markers:
            if re.match(marker, line, re.IGNORECASE | re.MULTILINE):
                return i
    return len(lines)


def extract_challenge_description(content: str) -> str:
    """Extract challenge description from writeup markdown (before solution)"""
    lines = content.split('\n')
    boundary = find_solution_boundary(content)
    
    description_lines = lines[:boundary]
    description = '\n'.join(description_lines)
    
    cleaned = re.sub(r'```.*?```', '', description, flags=re.DOTALL)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000] + "..."
    return cleaned


def extract_solution_text(content: str) -> str:
    """Extract solution text from writeup markdown (after challenge description)"""
    lines = content.split('\n')
    boundary = find_solution_boundary(content)
    
    if boundary >= len(lines):
        return ""
    
    solution_lines = lines[boundary:]
    solution = '\n'.join(solution_lines)
    solution = re.sub(r'\n{3,}', '\n\n', solution)
    solution = solution.strip()
    
    if len(solution) > 3000:
        solution = solution[:3000] + "..."
    return solution


def extract_writeups_from_repo(repo_path: str, category: str, repo_name: str) -> list:
    """Extract writeups from a cloned repo as proper Q&A pairs"""
    examples = []
    repo_path = Path(repo_path)
    
    md_files = []
    for md_file in repo_path.rglob("*.md"):
        if md_file.parent == repo_path and md_file.name.lower() == "readme.md":
            continue
        if md_file.name.lower() == "readme.md" and len(md_file.parts) - len(repo_path.parts) <= 1:
            continue
        md_files.append(md_file)
    for md_file in repo_path.rglob("*.MD"):
        if md_file.parent == repo_path and md_file.name.lower() == "readme.md":
            continue
        if md_file.name.lower() == "readme.md" and len(md_file.parts) - len(repo_path.parts) <= 1:
            continue
        md_files.append(md_file)
    
    md_files = list(set(md_files))
    
    total_files = len(md_files)
    extracted = 0
    start_time = time.time()
    
    file_iter = enumerate(md_files)
    if HAS_TQDM:
        file_iter = tqdm(enumerate(md_files), total=total_files, desc=f"    {repo_name}", unit="file")
    
    for idx, md_file in file_iter:
        try:
            content = md_file.read_text(errors='ignore')
            if len(content) < 100:
                continue
            
            challenge_name = md_file.stem.replace("-", " ").replace("_", " ").title()
            code_blocks = extract_code_blocks(content)
            
            parent = md_file.parent
            code_files = list(parent.glob("*.py")) + list(parent.glob("*.c")) + list(parent.glob("*.sh"))
            for cf in code_files[:3]:
                try:
                    code_content = cf.read_text(errors='ignore')
                    if len(code_content) > 10:
                        code_blocks.append({"lang": cf.suffix[1:] or "text", "code": code_content})
                except:
                    pass
            
            description = extract_challenge_description(content)
            solution_text = extract_solution_text(content)
            
            if not description or len(description) < 50:
                continue
            
            has_code = len(code_blocks) > 0
            
            if has_code:
                instruction = "Write an exploit/solution for this challenge"
                input_text = f"Challenge: {challenge_name}\nCategory: {category}\n\n{description}"
                
                solution_parts = []
                if solution_text:
                    solution_parts.append(solution_text)
                for block in code_blocks[:3]:
                    lang = block["lang"]
                    code = block["code"][:2000]
                    solution_parts.append(f"```{lang}\n{code}\n```")
                output_text = "\n\n".join(solution_parts)
            else:
                instruction = "Explain how to solve this challenge step by step"
                input_text = f"Challenge: {challenge_name}\nCategory: {category}\n\n{description}"
                
                if solution_text:
                    output_text = solution_text
                else:
                    output_text = f"## Solution for {challenge_name}\n\n{description}"
            
            if output_text and len(output_text) > 50:
                examples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
                extracted += 1
            
            if not HAS_TQDM and ((idx + 1) % 10 == 0 or idx + 1 == total_files):
                print(f"    [{idx + 1}/{total_files}] files processed, {extracted} extracted so far", flush=True)
                
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    print(f"    ✓ {repo_name}: {extracted} examples from {total_files} files ({elapsed:.1f}s)")
    return examples


def extract_from_huggingface(dataset_name: str, max_samples: int = 5000) -> list:
    """Extract and convert a HuggingFace dataset to Alpaca format"""
    examples = []
    
    if not HAS_DATASETS:
        print(f"  Skipping {dataset_name} (datasets library not installed)")
        return examples
    
    start_time = time.time()
    try:
        try:
            ds = load_dataset(dataset_name, split="train")
        except Exception:
            try:
                ds = load_dataset(dataset_name, split="test")
            except Exception:
                ds = load_dataset(dataset_name)
        
        ds_len = min(len(ds), max_samples)
        data_iter = ds
        if HAS_TQDM:
            data_iter = tqdm(ds, total=ds_len, desc=f"    {dataset_name}", unit="row")
        
        for i, item in enumerate(data_iter):
            if i >= max_samples:
                break
            
            question = item.get("question", item.get("problem", item.get("description", "")))
            answer = item.get("answer", item.get("solution", item.get("flag", "")))
            category = item.get("category", item.get("type", ""))
            
            text_chunk = item.get("text_chunk", "")
            if text_chunk and not question:
                question = "Explain this CTF writeup and provide the solution"
                answer = text_chunk
            
            if question and answer:
                examples.append({
                    "instruction": question,
                    "input": "",
                    "output": f"Category: {category}\n\n{answer}" if category else answer
                })
        
        elapsed = time.time() - start_time
        print(f"    ✓ {dataset_name}: {len(examples)} examples ({elapsed:.1f}s)")
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
        
        sections = re.split(r'\n(?=## )', content)
        
        for section in sections:
            if len(section) < 50:
                continue
            
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
    start_time = time.time()
    
    print("\n=== Step 1/2: Building CTF Writeups Dataset ===\n")
    
    total_repos = len(CTF_WRITEUP_REPOS)
    for repo_idx, repo in enumerate(CTF_WRITEUP_REPOS):
        print(f"\n[{repo_idx + 1}/{total_repos}] Cloning {repo['name']}...")
        repo_path = f"{tempfile.gettempdir()}/{repo['name']}"
        
        if clone_repo(repo["url"], repo_path):
            print(f"  Extracting writeups from {repo['name']}...")
            examples = extract_writeups_from_repo(repo_path, repo["category"], repo['name'])
            examples = examples[:max_per_repo]
            all_examples.extend(examples)
        else:
            print(f"  Skipped (clone failed)")
    
    print(f"\nLoading HuggingFace datasets...")
    hf_datasets = [
        ("kyleavery/picoctf", 500),
        ("justinwangx/CTFtime", 2000),
    ]
    
    for ds_name, max_samples in hf_datasets:
        print(f"\n  Loading {ds_name}...")
        examples = extract_from_huggingface(ds_name, max_samples)
        all_examples.extend(examples)
    
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_examples:
            f.write(json.dumps(item) + "\n")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Done: {len(all_examples)} examples saved to {output_path} ({elapsed:.1f}s)")
    return all_examples


def build_docs_dataset(output_path: str, max_per_doc: int = 200):
    """Build dataset from documentation"""
    all_examples = []
    start_time = time.time()
    
    print("\n=== Step 2/2: Building Documentation Dataset ===\n")
    
    total_docs = len(DOC_SOURCES)
    for doc_idx, doc in enumerate(DOC_SOURCES):
        print(f"[{doc_idx + 1}/{total_docs}] Scraping {doc['name']}...")
        
        if doc["type"] == "github_file" or doc["type"] == "github_readme":
            examples = scrape_documentation(doc["url"], doc["name"])
        elif doc["type"] == "github_docs":
            readme_url = doc["url"].replace("github.com", "raw.githubusercontent.com") + "/dev/README.md"
            examples = scrape_documentation(readme_url, doc["name"])
        else:
            examples = []
        
        examples = examples[:max_per_doc]
        all_examples.extend(examples)
        print(f"  ✓ {len(examples)} examples extracted")
    
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_examples:
            f.write(json.dumps(item) + "\n")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Done: {len(all_examples)} examples saved to {output_path} ({elapsed:.1f}s)")
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
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-per-repo", type=int, default=500)
    parser.add_argument("--max-per-doc", type=int, default=200)
    args = parser.parse_args()
    
    start_time = time.time()
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"  Dataset Builder - source: {args.source}")
    print(f"{'='*50}")
    
    if args.source == "writeups":
        build_writeups_dataset(f"{args.output_dir}/writeups.jsonl", args.max_per_repo)
    elif args.source == "docs":
        build_docs_dataset(f"{args.output_dir}/docs.jsonl", args.max_per_repo)
    elif args.source == "all":
        build_writeups_dataset(f"{args.output_dir}/writeups.jsonl", args.max_per_repo)
        build_docs_dataset(f"{args.output_dir}/docs.jsonl", args.max_per_repo)
    elif args.source == "merge":
        print("\n=== Merging datasets ===")
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
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

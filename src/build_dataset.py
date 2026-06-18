"""
Custom Dataset Builder - Scrapes GitHub repos and docs, converts to Alpaca format
Usage:
  uv run src/build_dataset.py --source writeups
  uv run src/build_dataset.py --source docs
  uv run src/build_dataset.py --source all
"""
import json
import os
import re
import argparse
import hashlib
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

# Generous safety cap for extraction length
MAX_OUTPUT_LEN = 20000


def dedup_examples(examples: list) -> list:
    """Remove duplicates by hashing user content."""
    seen = set()
    unique = []
    for ex in examples:
        key = hashlib.md5(ex.get('input', ex.get('instruction', '')).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    dropped = len(examples) - len(unique)
    if dropped > 0:
        print(f"    Deduped: dropped {dropped} duplicates")
    return unique


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

    # Reverse engineering focused
    {"url": "https://github.com/mohitmishra786/reversingBits", "name": "reversingBits", "category": "rev", "max_per_repo": 200},
    {"url": "https://github.com/x86byte/RE-MA-Roadmap", "name": "re-ma-roadmap", "category": "rev", "max_per_repo": 150},

    # Binary exploitation (pwn) focused
    {"url": "https://github.com/Crypto-Cat/CTF", "name": "cryptocat-ctf", "category": "pwn", "max_per_repo": 200},
    {"url": "https://github.com/Bretley/how2exploit_binary", "name": "how2exploit", "category": "pwn", "max_per_repo": 100},
]

# Documentation repos/files to scrape (fixed URLs)
DOC_SOURCES = [
    # pwntools
    {"type": "github_docs", "url": "https://github.com/Gallopsled/pwntools", "branch": "dev", "path": "docs", "name": "pwntools-docs"},
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


def extract_challenge_description(content: str, category: str = "") -> str:
    """Extract challenge description from writeup markdown (before challenge desc)"""
    max_len = MAX_OUTPUT_LEN
    lines = content.split('\n')
    boundary = find_solution_boundary(content)
    
    description_lines = lines[:boundary]
    description = '\n'.join(description_lines)
    
    cleaned = re.sub(r'```.*?```', '', description, flags=re.DOTALL)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "..."
    return cleaned


def extract_solution_text(content: str, category: str = "") -> str:
    """Extract solution text from writeup markdown (after challenge description)"""
    max_len = MAX_OUTPUT_LEN
    lines = content.split('\n')
    boundary = find_solution_boundary(content)
    
    if boundary >= len(lines):
        return ""
    
    solution_lines = lines[boundary:]
    solution = '\n'.join(solution_lines)
    solution = re.sub(r'\n{3,}', '\n\n', solution)
    solution = solution.strip()
    
    if len(solution) > max_len:
        solution = solution[:max_len] + "..."
    return solution


def extract_writeups_from_repo(repo_path: str, category: str, repo_name: str) -> list:
    """Extract writeups from a cloned repo as proper Q&A pairs"""
    examples = []
    repo_path = Path(repo_path)
    
    md_files = list({p for p in repo_path.rglob("*.[mM][dD]")})
    md_files = [p for p in md_files if not (p.name.lower() == "readme.md" and p.parent == repo_path)]
    
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
                except Exception:
                    pass
            
            description = extract_challenge_description(content, category)
            solution_text = extract_solution_text(content, category)
            
            if not description or len(description) < 50:
                continue
            
            has_code = len(code_blocks) > 0
            
            if has_code:
                instruction = "Analyze this CTF challenge, explain your approach step by step, and provide the exploit/solution code."
                input_text = f"Challenge: {challenge_name}\nCategory: {category}\n\n{description}"
                
                solution_parts = []
                if solution_text:
                    solution_parts.append(solution_text)
                for block in code_blocks[:3]:
                    lang = block["lang"]
                    code = block["code"][:MAX_OUTPUT_LEN]
                    solution_parts.append(f"```{lang}\n{code}\n```")
                output_text = "\n\n".join(solution_parts)
            else:
                instruction = "Analyze this CTF challenge step by step: first identify the type and category, then examine the available information, reason through possible approaches, and finally provide your solution with explanation."
                input_text = f"Challenge: {challenge_name}\nCategory: {category}\n\n{description}"
                
                if solution_text:
                    output_text = solution_text
                else:
                    output_text = f"## Solution for {challenge_name}\n\n{description}"
            
            if output_text and len(output_text) > 50:
                examples.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "category": category,
                    "source": repo_name,
                })
                extracted += 1
            
            if not HAS_TQDM and ((idx + 1) % 10 == 0 or idx + 1 == total_files):
                print(f"    [{idx + 1}/{total_files}] files processed, {extracted} extracted so far", flush=True)

        except Exception:
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
                "input": section[:MAX_OUTPUT_LEN],
                "output": f"Based on {name} documentation:\n\n{section[:MAX_OUTPUT_LEN]}"
            })
    except Exception as e:
        print(f"  Failed to scrape {url}: {e}")
    
    return examples


def build_ctfdojo_dataset(output_path: str, max_samples: int = 500) -> list:
    """Load amazon-science/CTF-Dojo SFT trajectories from a local clone."""
    import json as _json
    from pathlib import Path

    ctfdojo_root = Path("./data/raw/ctfdojo")
    sft_dir = ctfdojo_root / "SFT-data"

    if not sft_dir.exists():
        ctfdojo_root.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Cloning amazon-science/CTF-Dojo...")
        Repo.clone_from(
            "https://github.com/amazon-science/CTF-Dojo",
            str(ctfdojo_root),
            depth=1,
        )
        print(f"  Clone complete")

    examples = []
    for jsonl_file in sft_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if len(examples) >= max_samples:
                    break
                item = _json.loads(line)
                task = item.get("task", item.get("question", ""))
                trajectory = item.get("trajectory", item.get("expert_trajectory", ""))
                flag = item.get("flag", "")
                if not task or not trajectory:
                    continue
                examples.append({
                    "instruction": task,
                    "input": "",
                    "output": trajectory + (f"\n\nFlag: {flag}" if flag else ""),
                    "category": item.get("category", "ctf"),
                    "source": "ctfdojo",
                })
        if len(examples) >= max_samples:
            break

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} CTF-Dojo examples to {output_path}")
    return examples


def build_writeups_dataset(output_path: str, max_per_repo: int = 500):
    """Build dataset from CTF writeup repos"""
    import concurrent.futures

    all_examples = []
    start_time = time.time()
    
    print("\n=== Step 1/2: Building CTF Writeups Dataset ===\n")
    
    total_repos = len(CTF_WRITEUP_REPOS)

    # Phase 1: Clone all repos in parallel
    print(f"Cloning {total_repos} repos in parallel...")
    clone_results = {}

    def clone_one(repo):
        repo_path = f"{tempfile.gettempdir()}/{repo['name']}"
        return repo, clone_repo(repo['url'], repo_path), repo_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(clone_one, repo): repo for repo in CTF_WRITEUP_REPOS}
        for future in concurrent.futures.as_completed(futures):
            repo, success, repo_path = future.result()
            clone_results[repo['name']] = (success, repo_path)
            status = "✓" if success else "✗"
            print(f"  {status} {repo['name']}")

    # Phase 2: Extract from cloned repos in parallel (CPU-bound: markdown parsing + regex)
    def _extract_repo(args):
        repo_path, category, repo_name, repo_max = args
        from src.build_dataset import extract_writeups_from_repo, dedup_examples
        examples = extract_writeups_from_repo(repo_path, category, repo_name)
        examples = dedup_examples(examples)
        return examples[:repo_max], repo_name

    extract_args = []
    for repo in CTF_WRITEUP_REPOS:
        success, repo_path = clone_results[repo['name']]
        if success:
            repo_max = repo.get("max_per_repo", max_per_repo)
            extract_args.append((repo_path, repo["category"], repo['name'], repo_max))
        else:
            print(f"  Skipped {repo['name']} (clone failed)")

    max_workers = min(os.cpu_count() or 4, 8)
    print(f"\nExtracting from {len(extract_args)} repos in parallel (workers={max_workers})...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_extract_repo, args): args[2] for args in extract_args}
        for future in concurrent.futures.as_completed(futures):
            repo_name = futures[future]
            try:
                examples, _ = future.result()
                all_examples.extend(examples)
            except Exception as e:
                print(f"  ✗ {repo_name}: {e}")
    
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
            raw_url = doc["url"].replace("github.com", "raw.githubusercontent.com")
            branch = doc.get("branch", "main")
            readme_url = f"{raw_url}/refs/heads/{branch}/{doc.get('path', '')}/README.md"
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
    parser.add_argument("--source", choices=["writeups", "docs", "ctfd", "all", "merge"], required=True)
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
        build_docs_dataset(f"{args.output_dir}/docs.jsonl", args.max_per_doc)
    elif args.source == "ctfd":
        build_ctfdojo_dataset(f"{args.output_dir}/ctfdojo.jsonl", args.max_per_repo)
    elif args.source == "all":
        build_writeups_dataset(f"{args.output_dir}/writeups.jsonl", args.max_per_repo)
        build_docs_dataset(f"{args.output_dir}/docs.jsonl", args.max_per_doc)
        build_ctfdojo_dataset(f"{args.output_dir}/ctfdojo.jsonl", args.max_per_repo)

    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

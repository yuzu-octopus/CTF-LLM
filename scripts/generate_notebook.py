#!/usr/bin/env python3
"""Generate notebooks/qwen4b_self_contained.ipynb from src/*.py functions.
Run: python3 scripts/generate_notebook.py
"""
import json
import re
from pathlib import Path

PROJECT = Path(__file__).parent.parent
SRC = PROJECT / "src"
NB_PATH = PROJECT / "notebooks" / "qwen4b_self_contained.ipynb"

SRC_FILES = {
    "build_dataset": SRC / "build_dataset.py",
    "download_datasets": SRC / "download_datasets.py",
    "process_data": SRC / "process_data.py",
}

src_content = {}
for name, path in SRC_FILES.items():
    src_content[name] = path.read_text()

def extract_function(filepath, func_name):
    content = src_content[filepath]
    lines = content.split("\n")
    func_lines = []
    in_func = False
    func_indent = 0
    for line in lines:
        if re.match(rf"^def {func_name}\(", line):
            in_func = True
            func_indent = len(line) - len(line.lstrip())
            func_lines.append(line)
            continue
        if in_func:
            if line.strip() == "":
                func_lines.append(line)
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent > func_indent:
                func_lines.append(line)
            elif re.match(r"^def |^class ", line):
                break
            else:
                func_lines.append(line)
    return "\n".join(func_lines).rstrip()

def extract_constants(filepath, names):
    content = src_content[filepath]
    lines = content.split("\n")
    result = []
    for name in names:
        for i, line in enumerate(lines):
            if line.startswith(f"{name} =") or line.startswith(f"{name} = "):
                if '"""' in line or "\'\'\'" in line:
                    quote = '"""' if '"""' in line else "\'\'\'"
                    collected = [line]
                    for j in range(i+1, len(lines)):
                        collected.append(lines[j])
                        if quote in lines[j] and j > i:
                            break
                    result.append("\n".join(collected))
                    break
                else:
                    result.append(line)
                    break
    return "\n".join(result)

def adapt_for_notebook(source):
    source = source.replace("from tqdm import tqdm", "from tqdm.notebook import tqdm")
    lines = source.split("\n")
    adapted = []
    skip_indent = None
    for line in lines:
        if line.strip().startswith("if HAS_TQDM:") or line.strip().startswith("if not HAS_TQDM:"):
            skip_indent = len(line) - len(line.lstrip())
            continue
        if skip_indent is not None:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() == "" or current_indent > skip_indent:
                continue
            else:
                skip_indent = None
        if "HAS_TQDM" in line and "= True" not in line and "= False" not in line:
            continue
        adapted.append(line)
    source = "\n".join(adapted)
    source = source.replace("[:3000]", "[:MAX_OUTPUT_LEN]")
    source = source.replace("[:2000]", "[:MAX_OUTPUT_LEN]")
    source = source.replace("[:1500]", "[:MAX_OUTPUT_LEN]")
    return source

src_funcs = {
    "clone_repo": adapt_for_notebook(extract_function("build_dataset", "clone_repo")),
    "find_solution_boundary": adapt_for_notebook(extract_function("build_dataset", "find_solution_boundary")),
    "extract_code_blocks": adapt_for_notebook(extract_function("build_dataset", "extract_code_blocks")),
    "load_hf_with_fallback": adapt_for_notebook(extract_function("download_datasets", "load_hf_with_fallback")),
    "extract_qa": adapt_for_notebook(extract_function("download_datasets", "extract_qa")),
    "scrape_doc": adapt_for_notebook(extract_function("build_dataset", "scrape_documentation")),
    "system_prompts": extract_constants("process_data", ["SYSTEM_PROMPT_CTF", "SYSTEM_PROMPT_CODING"]),
    "is_ctf_content": adapt_for_notebook(extract_function("process_data", "is_ctf_content")),
    "ctf_keywords": adapt_for_notebook(extract_constants("process_data", ["CTF_KEYWORDS"])),
}

with open(NB_PATH) as f:
    current_nb = json.load(f)

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
H = current_nb["cells"]

cells = [
    md(H[0]["source"]), md(H[1]["source"]), code(H[2]["source"]), code(H[3]["source"]),
    md(H[4]["source"]), code(H[5]["source"]), md(H[6]["source"]), code(H[7]["source"]),
    # Cell 8: FROM SRC
    code(["# 3.2 Helper functions (from src/build_dataset.py)\n",
          src_funcs["clone_repo"] + "\n\n" + src_funcs["find_solution_boundary"] + "\n\n" + src_funcs["extract_code_blocks"] + "\n\nprint(\"Helper functions defined\")"]),
    code(H[9]["source"]), code(H[10]["source"]), md(H[11]["source"]), code(H[12]["source"]),
    # Cell 13: FROM SRC
    code(["# 4.2 Download and convert HF datasets to Alpaca format\n# (from src/download_datasets.py)\n",
          src_funcs["load_hf_with_fallback"] + "\n\n" + src_funcs["extract_qa"]]),
    md(H[14]["source"]), code(H[15]["source"]),
    # Cell 16: FROM SRC
    code(["# 5.2 Scrape and parse documentation\n# (from src/build_dataset.py)\n",
          src_funcs["scrape_doc"]]),
    md(H[17]["source"]),
    # Cell 18: FROM SRC
    code(["# 6.1 System prompts (from src/process_data.py)\n",
          src_funcs["system_prompts"] + "\n\n" + src_funcs["ctf_keywords"] + "\n\n" + src_funcs["is_ctf_content"] + "\n\nprint(\"System prompts defined\")"]),
    code(H[19]["source"]), md(H[20]["source"]), code(H[21]["source"]), code(H[22]["source"]),
    md(H[23]["source"]), code(H[24]["source"]), code(H[25]["source"]), md(H[26]["source"]),
    code(H[27]["source"]), code(H[28]["source"]), md(H[29]["source"]), code(H[30]["source"]), code(H[31]["source"]),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 4
}

NB_PATH.write_text(json.dumps(notebook, indent=1))
md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Generated {NB_PATH} ({len(cells)} cells: {md_count} markdown, {code_count} code)")

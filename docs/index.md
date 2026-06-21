---
layout: default
title: CTF-LLM
---

<section class="hero">
  <h1>CTF-LLM</h1>
  <p class="tagline">Fine-tune open-source LLMs to solve CTF challenges</p>
  <p class="subtitle">End-to-end pipeline for cybersecurity CTF and competitive programming using Unsloth + QLoRA on Google Colab's free T4 GPU.</p>
  <div class="cta">
    <a href="https://github.com/yuzu-octopus/CTF-LLM" class="btn btn-primary">View on GitHub</a>
    <a href="#quickstart" class="btn btn-outline">Quick Start</a>
  </div>
</section>

<section class="features">
  <div class="wrapper">
    <h2>Features</h2>
    <div class="grid">
      <div class="card">
        <h3>End-to-end Pipeline</h3>
        <p>Scrapes CTF writeups from 13 GitHub repos, downloads curated HuggingFace datasets, processes to ChatML, trains with Unsloth QLoRA, evaluates on a 210-question benchmark.</p>
      </div>
      <div class="card">
        <h3>4 Models Supported</h3>
        <p>Gemma 4 E4B, Gemma 4 12B, Qwen 3.5 9B, Qwen 3.5 4B — all running 4-bit QLoRA on a free T4 GPU (16GB VRAM).</p>
      </div>
      <div class="card">
        <h3>210-Question CTF Benchmark</h3>
        <p>Curated evaluation set across 4 categories (pwn, rev, crypto, web) and 3 difficulty levels with Wilson 95% CI and McNemar's test.</p>
      </div>
      <div class="card">
        <h3>Colab-Ready</h3>
        <p>Single <code>finetune.sh</code> command handles session creation, dependency install, file upload, training, and model download.</p>
      </div>
    </div>
  </div>
</section>

<section class="models">
  <div class="wrapper">
    <h2>Supported Models</h2>
    <table>
      <thead>
        <tr><th>Model</th><th>Parameters</th><th>LoRA Rank</th><th>VRAM (T4)</th><th>Status</th></tr>
      </thead>
      <tbody>
        <tr><td>Gemma 4 E4B</td><td>~4.5B</td><td>32</td><td>~10GB</td><td>Comfortable</td></tr>
        <tr><td>Gemma 4 12B</td><td>~12B</td><td>32</td><td>~14GB</td><td>Tight</td></tr>
        <tr><td>Qwen 3.5 9B</td><td>~9B</td><td>32</td><td>~12GB</td><td>Comfortable</td></tr>
        <tr><td>Qwen 3.5 4B</td><td>~4B</td><td>8</td><td>~8GB</td><td>Comfortable</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section class="modes">
  <div class="wrapper">
    <h2>Training Modes</h2>
    <table>
      <thead>
        <tr><th>Parameter</th><th>Fast (~30 min)</th><th>Quality (~50-70 min)</th></tr>
      </thead>
      <tbody>
        <tr><td>Dataset</td><td>~500 examples</td><td>2500+ examples</td></tr>
        <tr><td>LoRA rank</td><td>8</td><td>32</td></tr>
        <tr><td>Max seq length</td><td>2048</td><td>4096</td></tr>
        <tr><td>Epochs</td><td>1</td><td>2</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section class="benchmark">
  <div class="wrapper">
    <h2>Benchmark Stats</h2>
    <p><strong>210 questions</strong> across 4 categories and 3 difficulty levels.</p>
    <ul class="stats">
      <li><strong>Categories</strong>: pwn (57), rev (50), crypto (50), web (53)</li>
      <li><strong>Difficulties</strong>: easy (72), medium (79), hard (59)</li>
      <li><strong>Task types</strong>: flag_extraction, code_generation, vulnerability_identification, patch_generation, exploit_trace</li>
      <li><strong>Statistical rigor</strong>: Wilson 95% CI, McNemar's test, contamination check, cheating detection</li>
    </ul>
  </div>
</section>

<section class="quickstart" id="quickstart">
  <div class="wrapper">
    <h2>Quick Start</h2>
    <pre><code># 1. Install dependencies
uv sync

# 2. Run full pipeline (build data + train)
./finetune.sh gemma4 --all

# 3. Evaluate
./finetune.sh gemma4 --eval</code></pre>
  </div>
</section>

<section class="stack">
  <div class="wrapper">
    <h2>Tech Stack</h2>
    <ul>
      <li><strong>Unsloth</strong> &mdash; QLoRA fine-tuning framework</li>
      <li><strong>PyTorch</strong> &mdash; deep learning backend</li>
      <li><strong>HuggingFace</strong> &mdash; datasets + model hub</li>
      <li><strong>Google Colab</strong> &mdash; free T4 GPU runtime</li>
      <li><strong>uv</strong> &mdash; Python package manager</li>
    </ul>
  </div>
</section>

<footer>
  <p>Apache 2.0 License &middot; <a href="https://github.com/yuzu-octopus/CTF-LLM">github.com/yuzu-octopus/CTF-LLM</a></p>
</footer>

# Plan 011: Add OG Image Asset (F6)

**Commit**: `6d7af02`  
**Status**: TODO  
**Effort**: S (~30 min design + asset commit)  
**Risk**: NONE (new asset, no code changes)

## Problem

Open Graph meta tags in `docs/index.html:14,19` point to `https://yuzu-octopus.github.io/CTF-LLM/og-image.png` which doesn't exist. Social previews (Twitter/X, LinkedIn, Discord, Slack) render as a bare URL or broken image.

The meta tags themselves shipped in an earlier commit; only the image asset is missing.

## Current State

```html
<!-- docs/index.html:14,19 -->
<meta property="og:image" content="https://yuzu-octopus.github.io/CTF-LLM/og-image.png">
<meta name="twitter:image" content="https://yuzu-octopus.github.io/CTF-LLM/og-image.png">
```

File doesn't exist:
```bash
$ ls docs/og-image.png
ls: docs/og-image.png: No such file or directory
```

## Fix

### Step 1: Create the OG image

Design a 1200×630 PNG banner with:
- **Background**: Dracula `bg` (#282a36)
- **Subject**: CTF-LLM flag logo (SVG converted) on the left, with the project name "CTF-LLM" in Dracula purple (#bd93f9) using JetBrains Mono
- **Tagline**: "Fine-tune LLMs for CTF Challenges" in Dracula foreground (#f8f8f2)
- **Decoration**: Subtle 2px purple underline accent, and small Dracula-comment (#6272a4) tech badges (Unsloth, QLoRA, T4)

Tools: Figma, Canva, or a simple Python script with Pillow.

### Step 2: Save to docs/og-image.png

Place the image at `docs/og-image.png` (alongside `docs/index.html`).

### Step 3: Optimize

Ensure the PNG is < 100 KB for fast loading. Use `pngquant` or TinyPNG.

### Step 4: Verify

```bash
# File exists
ls -lh docs/og-image.png
# Expected: og-image.png with 1200×630 dimensions, < 100 KB

# Meta tags already point to correct path
grep -n 'og:image' docs/index.html
# Expected: the existing tags (no change needed)

# After deploying: check with https://www.opengraph.xyz/
```

## Files to Create

- `docs/og-image.png` (1200×630 PNG, < 100 KB)

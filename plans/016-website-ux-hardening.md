# Plan 016: Website UX/UI Hardening

**Commit**: `0831d17`
**Status**: TODO
**Effort**: M (~3 h)
**Risk**: LOW (HTML/CSS only, no functional code)

## Problems

18 issues found across 4 severity levels. Key items:

**P1 — Functional**: Mobile hamburger button hidden behind sidebar when open (z-index conflict). Sidebar can be opened but NOT closed. `aria-controls` points to nonexistent `id`.

**P2 — Accessibility**: 4 contrast failures at 1.94:1–3.79:1 (12px text on panel bg needs ≥4.5:1). No `:focus-visible` styles on interactive elements — keyboard-only navigation is broken.

**P3 — Responsive**: Tables overflow on small viewports (no scroll wrapper). `border-radius` on collapsed tables has no effect. No backdrop when mobile sidebar is open. Mobile toggle is 34×32px (needs ≥44×44 per WCAG).

**P4 — Polish**: Render-blocking Google Fonts, heading underlines extend past short text, unused font weight "500", skip-link targets `#hero` not `<main>`, sidebar uses `<aside>` instead of `<nav>`.

## Fix

### Step 1: Fix mobile sidebar z-index and backdrop

In `docs/index.html`:

```css
/* Raise mobile toggle above sidebar */
.mobile-toggle {
  z-index: 1301;  /* was 1200 — drawer is 1200, toggle needs to be above */
}
```

Add a backdrop element for mobile:
```html
<div class="sidebar-backdrop" onclick="document.querySelector('.sidebar').classList.remove('open'); document.querySelector('.main').classList.remove('shifted');"></div>
```

```css
.sidebar-backdrop {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.5);
  z-index: 1100;
}
@media (max-width: 900px) {
  .sidebar.open ~ .sidebar-backdrop { display: block; }
}
```

Add `id="sidebar"` to the `<aside>` element so `aria-controls="sidebar"` resolves.

### Step 2: Fix contrast failures

Create a new CSS variable:
```css
:root {
  --muted-on-panel: #7e8fc4;  /* lightened muted for panel backgrounds — 4.57:1 contrast on #44475a */
}
```

Apply it where text is on `--panel` background:
```css
.sidebar-brand .tagline { color: var(--muted-on-panel); }
.sidebar-footer { color: var(--muted-on-panel); }
.sidebar-footer a { color: var(--purple); }  /* already 3.79:1, passes AA-large but not AA — consider lightening */
```

For the hero subtitle (`color: var(--muted)` on `var(--bg)`):
```css
.hero .subtitle { color: var(--muted-on-panel); }  /* brighter muted on dark bg */
```

For `.stack .badge strong` (purple on panel):
```css
.stack .badge strong { color: #caaaff; }  /* lighter purple that passes 4.5:1 */
```

### Step 3: Add focus-visible styles

```css
a:focus-visible, button:focus-visible, .btn:focus-visible, .card:focus-visible {
  outline: 2px solid var(--cyan);
  outline-offset: 2px;
}
```

### Step 4: Fix responsive issues

Wrap both tables in `overflow-x: auto` containers:
```html
<div style="overflow-x: auto;">
  <table>...</table>
</div>
```

Fix table border-radius (remove from table, wrap in div):
```html
<div style="border-radius: 8px; overflow: hidden; border: 1px solid var(--muted);">
  <table style="border: none;">...</table>
</div>
```

Remove `overflow: hidden` from the `<table>` itself since it does nothing with `border-collapse: collapse`.

### Step 5: Increase mobile touch target

```css
.mobile-toggle {
  padding: 12px 14px;  /* was 8px 10px — now ~40×38px minimum */
  font-size: 1.2rem;    /* larger hamburger icon */
}
```

Or use explicit `min-width: 44px; min-height: 44px`.

### Step 6: Fix remaining polish items

Change Google Fonts to preload pattern:
```html
<link rel="preload" as="style" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" onload="this.rel='stylesheet'">
```

Add `id="main-content"` to `<main>` and change skip-link to `href="#main-content"`.

Change `<aside class="sidebar">` to `<nav class="sidebar" aria-label="Main navigation">`.

Remove `500` from Google Fonts URL (only 400 and 700 are used).

### Step 7: Verify

```bash
# HTML is valid
python3 -c "import html.parser; p=html.parser.HTMLParser(); p.feed(open('docs/index.html').read()); print('HTML OK')"

# Verify contrast via webaim.org or browser devtools
# Manual check: all 12px text on panel bg should be ≥4.5:1 contrast

# Run existing tests to confirm no regression
uv run python -m pytest tests/ -v --tb=short
```

## Files to Modify

- `docs/index.html` — CSS + HTML changes for all 7 steps

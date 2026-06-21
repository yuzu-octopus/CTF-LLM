# Improvement Plans — CTF-LLM

## Previous /improve cycle (all shipped)

Generated from `/improve` audit against commit `5f04de4`. All 8 plans shipped in commit `9b71fcc`.

| # | Plan | Status |
|---|------|--------|
| 1 | Notebook critical fixes (N1-N6) | ✅ DONE (`9b71fcc`) |
| 2 | Eval security + correctness | ✅ DONE (`9b71fcc`) |
| 3 | Dead code cleanup | ✅ DONE (`9b71fcc`) |
| 4 | Model loading dedup | ✅ DONE (`9b71fcc`) |
| 5 | Test coverage | ✅ DONE (`9b71fcc`) |
| 6 | Docs stale values | ✅ DONE (`9b71fcc`) |
| 7 | finetune.sh fixes | ✅ DONE (`9b71fcc`) |
| 8 | Data quality hardening | ✅ DONE (`9b71fcc`) |

## Current /improve cycle (all shipped)

Generated from `/improve` audit against commit `6d7af02`. All 7 plans shipped in commit `44d12cb`.

### Execution Order

| # | Plan | Dependencies | Status |
|---|------|-------------|--------|
| 9 | Config drift fix (F3) | None | ✅ DONE (`44d12cb`) |
| 10 | Dead code cleanup (F5) | None | ✅ DONE (`44d12cb`) |
| 11 | Add OG image asset (F6) | None | ✅ DONE (`44d12cb`) |
| 12 | gen_eval_bench tests + docs (F1) | None | ✅ DONE (`44d12cb`) |
| 13 | CI pipeline (F2) | None | ✅ DONE (`44d12cb`) |
| 14 | Data pipeline test expansion (F4) | None | ✅ DONE (`44d12cb`) |
| 15 | eval.py orchestration tests (F7) | None | ✅ DONE (`44d12cb`) |

### Key outcomes

- **Config drift**: quality-mode comment headers added to all 5 config files
- **Dead code**: `extract_from_huggingface()` removed from `src/build_dataset.py`
- **OG image**: `docs/og-image.png` created (1200x630, Dracula + flag logo, JetBrains Mono)
- **gen_eval_bench**: 12 tests, `--output` CLI arg, preservation warning, doc refs in README + AGENTS
- **CI pipeline**: `.github/workflows/ci.yml` — ruff lint + pytest on push/PR
- **Data tests**: 10 tests across `build_dataset` (expanded) and `download_datasets` (new)
- **Eval tests**: 8 orchestration tests, `torch` import guard for GPU-free import

### Findings Index

| # | Finding | Plan | Status |
|---|---------|------|--------|
| F1 | gen_eval_bench.py (745 LOC) — 0 tests, 0 doc refs | 012 | ✅ |
| F2 | No CI pipeline | 013 | ✅ |
| F3 | Config drift between config.yaml, configs/*.yaml, notebook | 009 | ✅ |
| F4 | download_datasets.py untested; build_dataset.py under-tested | 014 | ✅ |
| F5 | Dead code: extract_from_huggingface() in build_dataset.py | 010 | ✅ |
| F6 | Missing og-image.png — OG tags return 404 | 011 | ✅ |
| F7 | eval.py orchestration functions untested | 015 | ✅ |

### Considered and rejected

- **More sidebar CSS tuning**: Already iterated 5+ rounds; remaining differences (icon style, line-height) are within acceptable variance. Diminishing returns.
- **Training pipeline tests (train.py)**: Requires GPU hardware ($250+/mo Colab Pro). Not testable in CI.

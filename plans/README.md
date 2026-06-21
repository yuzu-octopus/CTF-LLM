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

## Current /improve cycle

Generated from `/improve` audit against commit `6d7af02`. Flight plan: `fix_plan.md`.

### Execution Order

| # | Plan | Dependencies | Status |
|---|------|-------------|--------|
| 9 | Config drift fix (F3) | None | TODO |
| 10 | Dead code cleanup (F5) | None | TODO |
| 11 | Add OG image asset (F6) | None | TODO |
| 12 | gen_eval_bench tests + docs (F1) | None | TODO |
| 13 | CI pipeline (F2) | None | TODO |
| 14 | Data pipeline test expansion (F4) | None | TODO |
| 15 | eval.py orchestration tests (F7) | 015 depends on 013 (CI) for automation | TODO |

### Dependency notes

- Plans 9-11 are quick wins with no deps (~5-30 min each)
- Plan 12 is highest leverage but requires careful subtask preservation
- Plan 13 enables automated test running for all subsequent plans
- Plan 15 has a `from src.eval import torch` barrier that may need resolving first

### Findings Index

| # | Finding | Plan | Confidence |
|---|---------|------|------------|
| F1 | gen_eval_bench.py (745 LOC) — 0 tests, 0 doc refs | 012 | HIGH |
| F2 | No CI pipeline | 013 | HIGH |
| F3 | Config drift between config.yaml, configs/*.yaml, notebook | 009 | HIGH |
| F4 | download_datasets.py untested; build_dataset.py under-tested | 014 | HIGH |
| F5 | Dead code: extract_from_huggingface() in build_dataset.py | 010 | HIGH |
| F6 | Missing og-image.png — OG tags return 404 | 011 | MED |
| F7 | eval.py orchestration functions untested | 015 | MED |

### Considered and rejected

- **More sidebar CSS tuning**: Already iterated 5+ rounds; remaining differences (icon style, line-height) are within acceptable variance. Diminishing returns.
- **Training pipeline tests (train.py)**: Requires GPU hardware ($250+/mo Colab Pro). Not testable in CI.

# Improvement Plans — CTF-LLM

## Previous /improve cycles (all shipped)

| Cycle | Plans | Status |
|-------|-------|--------|
| Cycle 1 | 001-008 | ✅ DONE |
| Cycle 2 | 009-015 | ✅ DONE |
| Cycle 3 | 016-019 | TODO (stale — see notes) |

## Current /improve cycle — plans 020-029

Generated from `/improve deep` audit against commit `087ce44`.

### Execution Order

| # | Plan | Priority | Effort | Depends on | Status |
|---|------|----------|--------|------------|--------|
| 020 | Fix undefined var in process_data | P0 | S | none | DONE |
| 021 | Fix undefined LEARNING_RATE in notebook | P0 | S | none | DONE |
| 022 | Fix config drift lora_alpha | P0 | S | none | DONE |
| 023 | Fix temp dir race condition | P1 | S | none | DONE |
| 024 | Fix bare except in eval | P1 | S | none | DONE |
| 025 | Fix temperature=None in eval | P1 | S | none | DONE |
| 026 | Add tests for print_results | P2 | M | none | DONE |
| 027 | Add tests for grade_code sandbox | P2 | M | none | DONE |
| 028 | Parallelize doc scraping | P2 | S | none | REJECTED |
| 029 | Inline _model_name in eval | P2 | S | none | DONE |

### Notes on plans 016-019

Plans 016-019 were written before the recent streamlining passes (ponytail cuts, rsLoRA, config sync). Several items may already be addressed:
- 017 (pipeline fixes): B1 two-stage format mismatch still exists; B3 datasets version — check
- 018 (training quality): config drift partially addressed by rsLoRA changes
- 019 (dataset quality): depends on 017

### Dependency notes

- Plans 020-025 are independent bug fixes — can be parallelized
- Plans 026-027 add test coverage — should land after bug fixes
- Plans 028-029 are low-risk improvements — can be done anytime

### Findings considered and rejected

- `temperature=None` in generate(): acceptable for standard HF backends
- `HAS_TQDM` guards: already removed in ponytail pass
- `gen_eval_bench.py` size: data is data, not code bloat
- Triple merge duplication: 3 functions with different contexts, abstraction not worth it
- Inconsistent tqdm: cosmetic, no runtime impact

- **Parallelize doc scraping (028)**: Not worth doing — serial is fine for 12 docs, parallelization adds complexity for marginal gain.

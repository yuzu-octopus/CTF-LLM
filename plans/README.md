# Improvement Plans — CTF-LLM

Generated from `/improve` audit against commit `5f04de4`.

## Execution Order

| # | Plan | Dependencies | Status |
|---|------|-------------|--------|
| 1 | Notebook critical fixes (N1-N6) | None | TODO |
| 2 | Eval security + correctness (1, 2) | None | TODO |
| 3 | Dead code cleanup (3, 4, N5) | None | TODO |
| 4 | Model loading dedup (5) | None | TODO |
| 5 | Test coverage (6, 7) | None | TODO |
| 6 | Docs stale values (8, 9) | None | TODO |
| 7 | finetune.sh fixes (10, 11) | None | TODO |
| 8 | Data quality hardening (12, 13) | None | TODO |

## Findings Index

| # | Finding | Plan | Confidence |
|---|---------|------|------------|
| N1 | notebook Cell 19: `hf_examples`/`doc_examples` undefined | 001 | HIGH |
| N2 | notebook Cell 21: `use_rslora` missing for Gemma | 001 | HIGH |
| N3 | notebook hardcoded `/content` paths | 001 | HIGH |
| N4 | generator propagates broken Cell 19 | 001 | HIGH |
| N5 | generator dead `extract_writeup` extraction | 003 | HIGH |
| N6 | notebook Cell 21: `max_seq_length` missing for Gemma | 001 | MED |
| 1 | eval.py exec sandbox escape | 002 | HIGH |
| 2 | eval.py McNemar's p-value wrong | 002 | HIGH |
| 3 | train.py stage1_config dead code | 003 | HIGH |
| 4 | build_dataset.py --source merge no-op | 003 | HIGH |
| 5 | eval.py/train.py model loading duplication | 004 | HIGH |
| 6 | eval.py grading zero test coverage | 005 | HIGH |
| 7 | build_dataset.py extraction zero test coverage | 005 | HIGH |
| 8 | TRAINING.md wrong hyperparameters | 006 | HIGH |
| 9 | README.md stale config examples | 006 | HIGH |
| 10 | finetune.sh step numbering wrong | 007 | HIGH |
| 11 | finetune.sh --eval not in --all | 007 | HIGH |
| 12 | process_data.py silent exception swallowing | 008 | HIGH |
| 13 | config.yaml epochs=3 disagrees with AGENTS.md | 008 | HIGH |

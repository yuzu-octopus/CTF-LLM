# FIX_PLAN.md

All 9 evaluator-hardening fixes (E1–E9) verified resolved. Pipeline is colab-ready with statistics-defensible measurements.

## Run it

Fast (~30 min): `./finetune.sh qwen35-4b --all`
Quality (~50-70 min): `./finetune.sh gemma4 --all`
Evaluate: `./finetune.sh gemma4 --eval`

That's it.

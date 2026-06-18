# FIX_PLAN.md

Pipeline is complete and ready for Colab. All audited fixes (T1–T4 + N4/N6/N7) shipped; pytest 8/8; notebook regenerates self-consistently; runtime guards in place.

## Run it

```bash
./finetune.sh qwen35-4b --all    # fast mode, ~30 min on T4
./finetune.sh gemma4 --all       # quality mode, ~50-70 min on T4
```

Nothing left to do here.

"""
Unified Fine-tuning Script for CTF/Coding Models
Based on official Unsloth notebooks.
Usage:
  colab run --gpu T4 src/train.py --model gemma4
  colab run --gpu T4 src/train.py --model qwen35
"""
import argparse
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset


def load_config(model_key: str, config_dir: str = "configs") -> dict:
    """Load model config from YAML file"""
    config_path = Path(config_dir) / f"{model_key}.yaml"
    
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    # Fallback to main config.yaml
    main_config = Path("config.yaml")
    if main_config.exists():
        with open(main_config) as f:
            full_config = yaml.safe_load(f)
            return full_config.get("models", {}).get(model_key, {})
    
    raise FileNotFoundError(f"No config found for model: {model_key}")


def has_assistant(ex):
    msgs = ex.get('messages') or []
    return bool(msgs) and bool(msgs[-1].get('content'))


def train(model_key: str, data_file: str, output_dir: str, epochs: int = 3, lora_r: int = None, lora_alpha: int = None):
    config = load_config(model_key)
    model_config = config.get("model", config)
    training_config = config.get("training", {})
    
    # Override LoRA params if provided (used by two-stage training)
    if lora_r is not None:
        model_config['r'] = lora_r
    if lora_alpha is not None:
        model_config['lora_alpha'] = lora_alpha
    
    total_start = time.time()
    
    print(f"\n{'='*50}")
    print(f"  Training {model_key.upper()}")
    print(f"  Model: {model_config['name']}")
    print(f"  Max seq length: {model_config['max_seq_length']}")
    print(f"  LoRA rank: {model_config['r']}")
    print(f"{'='*50}")
    
    # Step 1/5: Load model
    print("\n  [1/5] Loading model...")
    step_start = time.time()
    
    # Use FastVisionModel for Gemma 4 models, FastLanguageModel for others
    if model_key.startswith("gemma"):
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_config["name"],
            load_in_4bit=model_config.get("load_in_4bit", True),
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = processor.tokenizer
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["name"],
            max_seq_length=model_config["max_seq_length"],
            load_in_4bit=model_config.get("load_in_4bit", True),
            dtype=None,
        )
    
    print(f"    ✓ Model loaded ({time.time() - step_start:.1f}s)")
    
    # Step 2/5: Set up chat template
    print("\n  [2/5] Setting up chat template...")
    from unsloth import get_chat_template
    
    if model_key.startswith("gemma"):
        processor = get_chat_template(processor, "gemma-4")
        tokenizer = processor.tokenizer
    else:
        tokenizer = get_chat_template(tokenizer, "chatml")

    print(f"    ✓ Chat template configured")

    # Step 3/5: Configure LoRA
    print("\n  [3/5] Configuring LoRA adapters...")
    if model_key.startswith("gemma"):
        from unsloth import FastVisionModel
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,  # CTF data is 100% text
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=model_config["r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config.get("lora_dropout", 0),
            bias=model_config.get("bias", "none"),
            random_state=training_config.get("seed", 3407),
            use_rslora=model_config.get("use_rslora", False),
            loftq_config=None,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
        )
    else:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_config["r"],
            target_modules=model_config["target_modules"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config.get("lora_dropout", 0),
            bias=model_config.get("bias", "none"),
            use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
            random_state=training_config.get("seed", 3407),
            max_seq_length=model_config["max_seq_length"],
            use_rslora=model_config.get("use_rslora", False),
        )
    
    print(f"    ✓ LoRA configured (r={model_config['r']})")
    
    # Step 4/5: Load and format dataset
    print("\n  [4/5] Loading dataset...")
    step_start = time.time()
    dataset = load_dataset("json", data_files={"train": data_file}, split="train")
    dataset = dataset.filter(has_assistant)
    print(f"    Dataset size: {len(dataset)} examples")

    # Length filter using chat-template token count
    actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    def length_ok(ex):
        text = actual_tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        tokens = actual_tokenizer.encode(text, add_special_tokens=False)
        return len(tokens) <= model_config["max_seq_length"]

    before_len = len(dataset)
    dataset = dataset.filter(length_ok, desc="Filter by length")
    print(f"    Length-filtered: {len(dataset)} examples (dropped {before_len - len(dataset)} long samples)")

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"    Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    print(f"    ✓ Dataset prepared ({time.time() - step_start:.1f}s)")
    
    # Step 5/5: Train
    print("\n  [5/5] Training...")
    from trl import SFTTrainer, SFTConfig
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            max_seq_length=model_config["max_seq_length"],
            per_device_train_batch_size=model_config.get("batch_size", 1),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            warmup_ratio=training_config.get("warmup_ratio", 0.05),
            num_train_epochs=epochs,
            learning_rate=float(training_config.get("learning_rate", 1e-4)),
            logging_steps=1,
            output_dir=output_dir,
            optim=training_config.get("optim", "adamw_8bit"),
            seed=training_config.get("seed", 3407),
            save_strategy=training_config.get("save_strategy", "steps"),
            save_steps=training_config.get("save_steps", 100),
            save_total_limit=training_config.get("save_total_limit", 2),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            assistant_only_loss=True,
            eval_strategy=training_config.get("eval_strategy", "steps"),
            eval_steps=training_config.get("eval_steps", 100),
            load_best_model_at_end=training_config.get("load_best_model_at_end", True),
            metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
            report_to="none",
            # NO packing — conflicts with assistant_only_loss in trl>=0.12
        ),
    )
    
    print(f"    Epochs: {epochs}, Batch: {model_config.get('batch_size', 1)}, Grad accum: {training_config.get('gradient_accumulation_steps', 4)}")
    train_start = time.time()
    trainer.train()
    print(f"    ✓ Training complete ({time.time() - train_start:.1f}s)")
    
    # Save outputs
    print("\n=== Saving models ===")
    
    print("  Saving LoRA adapter...")
    lora_path = f"{output_dir}/lora"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"    ✓ {lora_path}")
    
    if model_key.startswith("gemma"):
        print("  Skipping GGUF export (not supported on FastVisionModel)")
    else:
        print("  Exporting GGUF model...")
        gguf_path = f"{output_dir}/gguf"
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
        print(f"    ✓ {gguf_path}")

    if model_key.startswith("gemma"):
        print("  Skipping merged export (not supported on FastVisionModel)")
    else:
        print("  Exporting merged model...")
        merged_path = f"{output_dir}/merged"
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        print(f"    ✓ {merged_path}")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"  ✓ Training complete!")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Output: {output_dir}/")
    print(f"{'='*50}")
    
    return model, tokenizer


def train_two_stage(model_key: str, data_file: str, output_dir: str):
    """Two-stage SFT: broad r=8 first, then sharp r=32 on curated subset."""
    from pathlib import Path
    import json as _json
    
    config = load_config(model_key)
    model_config = config.get("model", config)
    training_config = config.get("training", {})
    
    # === STAGE 1: Broad foundation (r=8, 1 epoch, full dataset) ===
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Broad Foundation")
    print(f"  LoRA r=8, alpha=16, 1 epoch, full dataset")
    print(f"{'='*60}")
    
    stage1_config = {
        "model": dict(model_config, r=8, lora_alpha=16),
        "training": dict(training_config, num_train_epochs=1),
    }
    
    stage1_dir = f"{output_dir}/stage1"
    stage1_model, stage1_tokenizer = train(model_key, data_file, stage1_dir, epochs=1, lora_r=8, lora_alpha=16)
    
    # === STAGE 2: Sharp patterns (r=32, 2 epochs, curated subset) ===
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Sharp Patterns")
    print(f"  LoRA r=32, alpha=64, 2 epochs, curated subset")
    print(f"{'='*60}")
    
    # Build curated subset: synthetic_rev_pwn + ctfdojo + top writeups
    curated_file = f"{output_dir}/curated.jsonl"
    _build_curated_subset(curated_file)
    
    stage2_dir = f"{output_dir}/stage2"
    stage2_model, stage2_tokenizer = train(model_key, curated_file, stage2_dir, epochs=2, lora_r=32, lora_alpha=64)
    
    return stage2_model, stage2_tokenizer


def _build_curated_subset(output_path: str):
    """Build curated subset from synthetic + ctfdojo + top writeups."""
    import json as _json
    from pathlib import Path
    
    curated = []
    
    # 1. Add synthetic rev/pwn examples
    try:
        from src.synthetic_rev_pwn import SYNTHETIC_EXAMPLES
        for ex in SYNTHETIC_EXAMPLES:
            curated.append({
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex["output"],
                "category": ex.get("category", "rev/pwn"),
                "source": "synthetic",
            })
        print(f"  Added {len(SYNTHETIC_EXAMPLES)} synthetic rev/pwn examples")
    except ImportError:
        print("  Warning: synthetic_rev_pwn not found, skipping")
    
    # 2. Add CTF-Dojo examples (if available)
    ctfdojo_path = Path("data/raw/ctfdojo.jsonl")
    if ctfdojo_path.exists():
        with open(ctfdojo_path) as f:
            for line in f:
                ex = _json.loads(line)
                curated.append(ex)
        print(f"  Added {len(curated)} CTF-Dojo examples")
    else:
        print("  Warning: ctfdojo.jsonl not found, skipping")
    
    # 3. Add top 200 writeups from writeups.jsonl
    writeups_path = Path("data/raw/writeups.jsonl")
    if writeups_path.exists():
        count = 0
        with open(writeups_path) as f:
            for line in f:
                if count >= 200:
                    break
                ex = _json.loads(line)
                # Only include writeups with code blocks (higher quality)
                if "```" in ex.get("output", ""):
                    curated.append(ex)
                    count += 1
        print(f"  Added {count} curated writeups (with code blocks)")
    
    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for ex in curated:
            f.write(_json.dumps(ex) + "\n")
    print(f"  Total curated subset: {len(curated)} examples saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified CTF/Coding model fine-tuning")
    parser.add_argument("--model", choices=["gemma4", "gemma4-12b", "qwen35", "qwen35-4b"], required=True,
                       help="Model to fine-tune")
    parser.add_argument("--data", default="data/merged/train.jsonl",
                       help="Training data file (JSONL)")
    parser.add_argument("--output", default=None,
                       help="Output directory (default: outputs/{model}-ctf)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--two-stage", action="store_true",
                       help="Enable two-stage training: broad r=8 first, then sharp r=32 on curated subset")
    args = parser.parse_args()
    
    output_dir = args.output or f"outputs/{args.model}-ctf"
    
    print(f"\n{'='*50}")
    print(f"  CTF/Coding Fine-tuning Pipeline")
    print(f"  Model: {args.model}, Epochs: {args.epochs}")
    print(f"  Data: {args.data}")
    print(f"  Two-stage: {args.two_stage}")
    print(f"{'='*50}")
    
    if args.two_stage:
        train_two_stage(args.model, args.data, output_dir)
    else:
        train(args.model, args.data, output_dir, args.epochs)


if __name__ == "__main__":
    main()

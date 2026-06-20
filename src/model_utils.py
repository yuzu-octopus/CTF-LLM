"""Shared model loading utilities for train.py and eval.py."""
from pathlib import Path


def _model_name(model_key: str) -> str:
    """Map model key to HuggingFace model name."""
    from src.train import load_config
    config = load_config(model_key)
    return config.get("model", config).get("name", model_key)


def load_model_for_training(model_key: str, config: dict):
    """Load model for training with LoRA config."""
    from unsloth import FastLanguageModel, FastVisionModel

    model_config = config.get("model", config)
    model_name = _model_name(model_key)

    if model_key.startswith("gemma"):
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=model_config.get("load_in_4bit", True),
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = processor.tokenizer
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=model_config["max_seq_length"],
            load_in_4bit=model_config.get("load_in_4bit", True),
            dtype=None,
        )
    return model, tokenizer


def load_model_for_inference(model_key: str, adapter_path: str = None):
    """Load model for inference with optional LoRA adapter."""
    from unsloth import FastLanguageModel, FastVisionModel, get_chat_template

    model_name = _model_name(model_key)

    if model_key.startswith("gemma"):
        from unsloth import FastVisionModel as FVM
        model, processor = FVM.from_pretrained(
            model_name=model_name, load_in_4bit=True,
        )
        processor = get_chat_template(processor, "gemma-4")
        tokenizer = processor.tokenizer
        model = FVM.get_for_inference(model)
    else:
        from unsloth import FastLanguageModel as FLM
        model, tokenizer = FLM.from_pretrained(
            model_name=model_name, load_in_4bit=True, dtype=None,
        )
        tokenizer = get_chat_template(tokenizer, "chatml")
        model = FLM.get_for_inference(model)

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded LoRA adapter from {adapter_path}")

    return model, tokenizer

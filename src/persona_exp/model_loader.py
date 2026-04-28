from __future__ import annotations

import torch

from persona_exp.config import ModelConfig


def torch_dtype(dtype: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_and_tokenizer(model_cfg: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(model_cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        torch_dtype=torch_dtype(model_cfg.dtype) if device != "cpu" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    if hasattr(model.config, "n_layer"):
        return int(model.config.n_layer)
    raise ValueError("Could not determine model layer count from config")


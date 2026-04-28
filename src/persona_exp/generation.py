from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm


def generate_response(model, tokenizer, prompt_text: str, gen_cfg: dict[str, Any], device: str) -> dict[str, Any]:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 64))
    do_sample = bool(gen_cfg.get("do_sample", False))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(gen_cfg.get("temperature", 1.0))
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    response_ids = output[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    response_token_ids = [int(x) for x in response_ids.detach().cpu().tolist()]
    return {
        "response_text": response_text,
        "response_token_ids": response_token_ids,
        "num_prompt_tokens": int(prompt_len),
        "num_response_tokens": int(response_ids.shape[0]),
        "generation_config": dict(gen_cfg),
    }


def generate_responses(model, tokenizer, records: list[dict[str, Any]], gen_cfg, device: str):
    out = []
    for record in tqdm(records, desc="generating"):
        gen = generate_response(model, tokenizer, record["prompt_text"], gen_cfg, device)
        out.append({**record, **gen})
    return out

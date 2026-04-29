from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm


def generation_kwargs(tokenizer, gen_cfg: dict[str, Any]) -> dict[str, Any]:
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 64))
    do_sample = bool(gen_cfg.get("do_sample", False))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(gen_cfg.get("temperature", 1.0))
    return kwargs


def trim_generated_ids(tokenizer, response_ids: torch.Tensor) -> torch.Tensor:
    stop_ids = {x for x in [tokenizer.eos_token_id, tokenizer.pad_token_id] if x is not None}
    if not stop_ids:
        return response_ids
    ids = response_ids.detach().cpu().tolist()
    stop = len(ids)
    for idx, token_id in enumerate(ids):
        if int(token_id) in stop_ids:
            stop = 1 if idx == 0 else idx
            break
    return response_ids[:stop]


def generate_response(model, tokenizer, prompt_text: str, gen_cfg: dict[str, Any], device: str) -> dict[str, Any]:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    kwargs = generation_kwargs(tokenizer, gen_cfg)
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    response_ids = trim_generated_ids(tokenizer, output[0, prompt_len:])
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
    batch_size = int(gen_cfg.get("batch_size", 1))
    if batch_size <= 1:
        return generate_responses_unbatched(model, tokenizer, records, gen_cfg, device)

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    out = []
    kwargs = generation_kwargs(tokenizer, gen_cfg)
    try:
        for start in tqdm(range(0, len(records), batch_size), desc=f"generating batch_size={batch_size}"):
            batch = records[start : start + batch_size]
            prompt_texts = [record["prompt_text"] for record in batch]
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
            prompt_token_counts = inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
            input_width = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output = model.generate(**inputs, **kwargs)
            for idx, record in enumerate(batch):
                response_ids = trim_generated_ids(tokenizer, output[idx, input_width:])
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                response_token_ids = [int(x) for x in response_ids.detach().cpu().tolist()]
                out.append(
                    {
                        **record,
                        "response_text": response_text,
                        "response_token_ids": response_token_ids,
                        "num_prompt_tokens": int(prompt_token_counts[idx]),
                        "num_response_tokens": int(response_ids.shape[0]),
                        "generation_config": dict(gen_cfg),
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
    return out


def generate_responses_unbatched(model, tokenizer, records: list[dict[str, Any]], gen_cfg, device: str):
    out = []
    for record in tqdm(records, desc="generating"):
        gen = generate_response(model, tokenizer, record["prompt_text"], gen_cfg, device)
        out.append({**record, **gen})
    return out

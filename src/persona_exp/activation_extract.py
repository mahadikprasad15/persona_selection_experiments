from __future__ import annotations

from typing import Any

import torch

from persona_exp.formatting import (
    find_assistant_marker_token_index,
    find_pre_assistant_final_index,
    get_response_token_slice_with_validation,
)
from persona_exp.pooling import pool_first_k_generated, pool_response_mean, select_token


def _full_text(prompt_text: str, response_text: str) -> str:
    return prompt_text + response_text


def extract_hidden_states_teacher_forced(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    layers: list[Any],
    sites: list[str],
    device: str,
    activation_dtype: str = "float16",
    response_token_ids: list[int] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    used_synthetic_eos_for_empty_response = False
    if response_token_ids is not None:
        if len(response_token_ids) == 0:
            fallback_token_id = tokenizer.eos_token_id
            if fallback_token_id is None:
                fallback_token_id = tokenizer.pad_token_id
            if fallback_token_id is None:
                raise ValueError("Empty response token ids and tokenizer has no eos/pad fallback token")
            response_token_ids = [int(fallback_token_id)]
            used_synthetic_eos_for_empty_response = True
        inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = int(inputs["input_ids"].shape[1])
        response_ids = torch.tensor([response_token_ids], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], response_ids], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        response_slice = slice(prompt_len, prompt_len + len(response_token_ids))
        response_token_ids_match = True
    else:
        text = _full_text(prompt_text, response_text)
        inputs = tokenizer(text, return_tensors="pt")
        response_slice, response_token_ids_match = get_response_token_slice_with_validation(
            tokenizer, prompt_text, response_text, response_token_ids
        )
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True, use_cache=False)

    assistant_idx = find_assistant_marker_token_index(tokenizer, prompt_text)
    pre_assistant_idx = find_pre_assistant_final_index(tokenizer, prompt_text)
    dtype = torch.float16 if activation_dtype == "float16" else torch.float32
    tensors = {}
    for layer in layers:
        hidden = output.hidden_states[layer.layer_idx + 1][0].detach()
        for site in sites:
            if site == "role_response_mean":
                pooled = pool_response_mean(hidden, response_slice)
            elif site == "assistant_marker_final_token":
                pooled = select_token(hidden, assistant_idx)
            elif site == "pre_assistant_final_token":
                pooled = select_token(hidden, pre_assistant_idx)
            elif site == "gen_first_token":
                pooled = pool_first_k_generated(hidden, response_slice, 1)
            elif site == "gen_mean_5":
                pooled = pool_first_k_generated(hidden, response_slice, 5)
            elif site == "gen_mean_20":
                pooled = pool_first_k_generated(hidden, response_slice, 20)
            else:
                raise ValueError(f"Unknown site {site}")
            tensors[f"{layer.layer_tag}/{site}"] = pooled.to("cpu", dtype=dtype)
    meta = {
        "num_tokens": int(inputs["input_ids"].shape[1]),
        "response_start": int(response_slice.start or 0),
        "response_stop": int(response_slice.stop or 0),
        "response_token_ids_match": bool(response_token_ids_match),
        "used_synthetic_eos_for_empty_response": bool(used_synthetic_eos_for_empty_response),
        "assistant_marker_idx": int(assistant_idx),
        "pre_assistant_idx": int(pre_assistant_idx),
    }
    return tensors, meta

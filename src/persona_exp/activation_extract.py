from __future__ import annotations

from typing import Any

import torch

from persona_exp.formatting import (
    find_assistant_marker_token_index,
    find_pre_assistant_final_index,
    get_response_token_slice,
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
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    text = _full_text(prompt_text, response_text)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True, use_cache=False)

    response_slice = get_response_token_slice(tokenizer, prompt_text, response_text)
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
        "assistant_marker_idx": int(assistant_idx),
        "pre_assistant_idx": int(pre_assistant_idx),
    }
    return tensors, meta


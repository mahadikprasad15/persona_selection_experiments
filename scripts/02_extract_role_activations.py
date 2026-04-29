#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from persona_exp.activation_extract import extract_hidden_states_teacher_forced
from persona_exp.config import get_model, load_config, resolve_layer_indices, run_dir
from persona_exp.io import clean_tmp_files, mark_completed, write_status
from persona_exp.model_loader import get_num_layers, load_model_and_tokenizer
from persona_exp.utils import add_common_args, read_jsonl


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.model:
        raise ValueError("--model is required")
    model_cfg = get_model(cfg, args.model)
    rd = run_dir(cfg)
    out_root = rd / "role_activations" / f"model={model_cfg.name}" / "site=role_response_mean"
    if (out_root / "_SUCCESS").exists() and not args.overwrite:
        print(f"role activations already complete: {out_root}")
        return
    clean_tmp_files(out_root)
    write_status(rd, "running", "role_activations", {"model": model_cfg.name})
    model, tokenizer, device = load_model_and_tokenizer(model_cfg)
    layers = resolve_layer_indices(cfg["extraction"]["layer_fracs"], get_num_layers(model))
    rollouts = []
    for path in sorted((rd / "role_rollouts" / f"model={model_cfg.name}").glob("part-*.jsonl")):
        rollouts.extend(read_jsonl(path))
    if not rollouts:
        raise FileNotFoundError("No role rollouts found")

    by_layer = {layer.layer_tag: {"vectors": [], "meta": []} for layer in layers}
    for record_idx, record in enumerate(tqdm(rollouts, desc="extracting role activations")):
        try:
            tensors, meta = extract_hidden_states_teacher_forced(
                model,
                tokenizer,
                record["prompt_text"],
                record["response_text"],
                layers,
                ["role_response_mean"],
                device,
                cfg["extraction"]["activation_dtype"],
                record.get("response_token_ids"),
            )
        except ValueError as exc:
            keys = ["role_id", "role_name", "question_id", "template_id", "instruction_id"]
            context = {key: record.get(key) for key in keys if key in record}
            context["record_idx"] = record_idx
            context["num_response_tokens"] = record.get("num_response_tokens")
            raise ValueError(f"Failed to extract role activations for {context}") from exc
        for layer in layers:
            by_layer[layer.layer_tag]["vectors"].append(tensors[f"{layer.layer_tag}/role_response_mean"])
            by_layer[layer.layer_tag]["meta"].append({**record, **meta, **layer.__dict__, "site": "role_response_mean"})

    for layer in layers:
        layer_dir = out_root / f"layer={layer.layer_tag}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        mat = torch.stack(by_layer[layer.layer_tag]["vectors"])
        save_file({"activations": mat}, str(layer_dir / "shard-00000.safetensors"))
        pd.DataFrame(by_layer[layer.layer_tag]["meta"]).to_parquet(layer_dir / "shard-00000.metadata.parquet", index=False)
    mark_completed(out_root, {"num_records": len(rollouts)})
    write_status(rd, "completed", "role_activations", {"model": model_cfg.name, "num_records": len(rollouts)})


if __name__ == "__main__":
    main()

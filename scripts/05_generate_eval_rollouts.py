#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from persona_exp.config import get_model, load_config, load_eval_prompts, run_dir
from persona_exp.formatting import format_eval_prompt
from persona_exp.generation import generate_responses
from persona_exp.io import write_jsonl_shards, write_status
from persona_exp.model_loader import load_model_and_tokenizer
from persona_exp.utils import add_common_args, set_seed


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.model:
        raise ValueError("--model is required")
    set_seed(int(cfg["run"]["seed"]))
    model_cfg = get_model(cfg, args.model)
    rd = run_dir(cfg)
    out_root = rd / "eval_rollouts" / f"model={model_cfg.name}"
    if (out_root / "_SUCCESS").exists() and not args.overwrite:
        print(f"eval rollouts already complete: {out_root}")
        return
    write_status(rd, "running", "eval_rollouts", {"model": model_cfg.name})
    model, tokenizer, device = load_model_and_tokenizer(model_cfg)
    records = []
    for prompt in load_eval_prompts(cfg):
        records.append(
            {
                "prompt_id": prompt["prompt_id"],
                "category": prompt["category"],
                "subcategory": prompt.get("subcategory", "unknown"),
                "source": prompt.get("source", "unknown"),
                "source_id": str(prompt.get("source_id", "")),
                "source_metadata_json": json.dumps(prompt.get("metadata", {}), sort_keys=True),
                "model_name": model_cfg.name,
                "prompt_text": format_eval_prompt(prompt, cfg["formatting"]["template"]),
            }
        )
    generated = generate_responses(model, tokenizer, records, cfg["generation"]["eval"], device)
    by_cat = {}
    for rec in generated:
        by_cat.setdefault(rec["category"], []).append(rec)
    shard_size = int(cfg["storage"]["shard_size"]["eval_rollouts"])
    for category, rows in by_cat.items():
        write_jsonl_shards(out_root / f"category={category}", rows, shard_size, overwrite=args.overwrite)
    from persona_exp.io import mark_completed

    mark_completed(out_root, {"num_records": len(generated)})
    write_status(rd, "completed", "eval_rollouts", {"model": model_cfg.name, "num_records": len(generated)})


if __name__ == "__main__":
    main()

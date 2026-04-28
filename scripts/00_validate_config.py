#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys

from persona_exp.config import (
    get_models,
    load_config,
    resolve_layer_indices,
    save_resolved_config,
    validate_data_files,
    write_initial_run_metadata,
)
from persona_exp.model_loader import get_num_layers, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    counts = validate_data_files(cfg)
    resolved = {"data_counts": counts, "models": {}}
    if not args.skip_model_load:
        for model_cfg in get_models(cfg):
            model, tokenizer, _device = load_model_and_tokenizer(model_cfg)
            num_layers = get_num_layers(model)
            sample = tokenizer("User: hello\nAssistant:", add_special_tokens=False)
            resolved["models"][model_cfg.name] = {
                "num_layers": num_layers,
                "layers": [x.__dict__ for x in resolve_layer_indices(cfg["extraction"]["layer_fracs"], num_layers)],
                "sample_prompt_tokens": len(sample["input_ids"]),
            }
            del model
    write_initial_run_metadata(cfg, status="validated")
    save_resolved_config(cfg, resolved)
    print(f"validated {args.config}: {counts}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise


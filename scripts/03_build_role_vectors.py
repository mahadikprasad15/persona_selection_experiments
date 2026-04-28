#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd
from safetensors.numpy import load_file, save_file

from persona_exp.config import get_model, load_config, run_dir
from persona_exp.io import mark_completed, write_status
from persona_exp.utils import add_common_args
from persona_exp.vector_building import build_all_vector_types


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.model:
        raise ValueError("--model is required")
    model_cfg = get_model(cfg, args.model)
    rd = run_dir(cfg)
    in_root = rd / "role_activations" / f"model={model_cfg.name}" / "site=role_response_mean"
    out_root = rd / "role_vectors" / f"model={model_cfg.name}"
    if (out_root / "_SUCCESS").exists() and not args.overwrite:
        print(f"role vectors already complete: {out_root}")
        return
    write_status(rd, "running", "role_vectors", {"model": model_cfg.name})
    out_root.mkdir(parents=True, exist_ok=True)
    for layer_dir in sorted(in_root.glob("layer=*")):
        layer_tag = layer_dir.name.split("=", 1)[1]
        vectors = load_file(str(layer_dir / "shard-00000.safetensors"))["activations"]
        meta = pd.read_parquet(layer_dir / "shard-00000.metadata.parquet").reset_index(drop=True)
        layer_out = out_root / f"layer={layer_tag}"
        layer_out.mkdir(parents=True, exist_ok=True)
        for vector_type, (role_meta, role_vectors) in build_all_vector_types(meta, vectors).items():
            save_file({"role_vectors": role_vectors}, str(layer_out / f"role_vectors_{vector_type}.safetensors"))
            role_meta.to_parquet(layer_out / f"role_vector_metadata_{vector_type}.parquet", index=False)
    mark_completed(out_root)
    write_status(rd, "completed", "role_vectors", {"model": model_cfg.name})


if __name__ == "__main__":
    main()


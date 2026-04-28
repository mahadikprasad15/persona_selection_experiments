#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd
from safetensors.numpy import load_file

from persona_exp.alignment import compute_pc_alignment, compute_same_role_cosines
from persona_exp.config import get_models, load_config, run_dir
from persona_exp.io import mark_completed, write_status


def load_vectors(rd, model_name, layer_tag, vector_type):
    root = rd / "role_vectors" / f"model={model_name}" / f"layer={layer_tag}"
    vec = load_file(str(root / f"role_vectors_{vector_type}.safetensors"))["role_vectors"]
    meta = pd.read_parquet(root / f"role_vector_metadata_{vector_type}.parquet")
    return meta, vec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vector-type", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rd = run_dir(cfg)
    vector_type = args.vector_type or cfg["role_vectors"]["default_vector_type"]
    write_status(rd, "running", "alignment", {"vector_type": vector_type})
    models = get_models(cfg)
    layer_dirs = sorted((rd / "role_vectors" / f"model={models[0].name}").glob("layer=*"))
    for a, b in itertools.combinations(models, 2):
        pair_dir = rd / "alignment" / f"{a.name}_vs_{b.name}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        for layer_dir in layer_dirs:
            layer_tag = layer_dir.name.split("=", 1)[1]
            meta_a, vec_a = load_vectors(rd, a.name, layer_tag, vector_type)
            meta_b, vec_b = load_vectors(rd, b.name, layer_tag, vector_type)
            cos = compute_same_role_cosines(meta_a, vec_a, meta_b, vec_b)
            cos.to_parquet(pair_dir / f"same_role_cosines_{layer_tag}.parquet", index=False)
            pc = compute_pc_alignment(vec_a, vec_b, n_components=3)
            with (pair_dir / f"pc_alignment_{layer_tag}.json").open("w", encoding="utf-8") as f:
                json.dump(pc, f, indent=2, sort_keys=True)
    mark_completed(rd / "alignment")
    write_status(rd, "completed", "alignment", {"vector_type": vector_type})


if __name__ == "__main__":
    main()


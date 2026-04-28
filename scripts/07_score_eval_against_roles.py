#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd
from safetensors.numpy import load_file

from persona_exp.config import get_model, load_config, run_dir
from persona_exp.io import mark_completed, write_status
from persona_exp.scoring import score_dot, softmax_scores
from persona_exp.utils import add_common_args


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--vector-type", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.model:
        raise ValueError("--model is required")
    model_cfg = get_model(cfg, args.model)
    rd = run_dir(cfg)
    vector_type = args.vector_type or cfg["role_vectors"]["default_vector_type"]
    out_root = rd / "scores" / f"model={model_cfg.name}" / f"vector_set={vector_type}"
    if (out_root / "_SUCCESS").exists() and not args.overwrite:
        print(f"scores already complete: {out_root}")
        return
    write_status(rd, "running", "scores", {"model": model_cfg.name, "vector_type": vector_type})
    for site_dir in sorted((rd / "eval_activations" / f"model={model_cfg.name}").glob("site=*")):
        site = site_dir.name.split("=", 1)[1]
        for layer_dir in sorted(site_dir.glob("layer=*")):
            layer_tag = layer_dir.name.split("=", 1)[1]
            h = load_file(str(layer_dir / "shard-00000.safetensors"))["activations"]
            h_meta = pd.read_parquet(layer_dir / "shard-00000.metadata.parquet")
            v_root = rd / "role_vectors" / f"model={model_cfg.name}" / f"layer={layer_tag}"
            v = load_file(str(v_root / f"role_vectors_{vector_type}.safetensors"))["role_vectors"]
            v_meta = pd.read_parquet(v_root / f"role_vector_metadata_{vector_type}.parquet")
            scores = score_dot(h, v)
            probs = softmax_scores(scores, float(cfg["scoring"]["softmax"]["temperature"]))
            rows = []
            for i, sample in h_meta.reset_index(drop=True).iterrows():
                for j, role in v_meta.reset_index(drop=True).iterrows():
                    rows.append(
                        {
                            "model_name": model_cfg.name,
                            "checkpoint_stage": model_cfg.stage,
                            "layer_tag": sample["layer_tag"],
                            "layer_idx": int(sample["layer_idx"]),
                            "site": site,
                            "prompt_id": sample["prompt_id"],
                            "prompt_category": sample["category"],
                            "prompt_subcategory": sample.get("subcategory", "unknown"),
                            "prompt_source": sample.get("source", "unknown"),
                            "prompt_source_id": sample.get("source_id", ""),
                            "prompt_source_metadata_json": sample.get("source_metadata_json", "{}"),
                            "role_id": role["role_id"],
                            "role_cluster": role["role_cluster"],
                            "score_dot": float(scores[i, j]),
                            "score_softmax_T1": float(probs[i, j]),
                        }
                    )
            score_dir = out_root / f"site={site}" / f"layer={layer_tag}"
            score_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(score_dir / "scores.parquet", index=False)
    mark_completed(out_root)
    write_status(rd, "completed", "scores", {"model": model_cfg.name, "vector_type": vector_type})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd

from persona_exp.aggregation import aggregate_cluster_mass, aggregate_mean_scores, aggregate_mean_softmax, compute_model_deltas
from persona_exp.config import load_config, run_dir
from persona_exp.io import mark_completed, write_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vector-type", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rd = run_dir(cfg)
    vector_type = args.vector_type or cfg["role_vectors"]["default_vector_type"]
    write_status(rd, "running", "aggregates", {"vector_type": vector_type})
    paths = sorted((rd / "scores").glob(f"model=*/vector_set={vector_type}/site=*/layer=*/scores.parquet"))
    if not paths:
        raise FileNotFoundError("No score parquet files found")
    scores = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out = rd / "aggregates"
    out.mkdir(parents=True, exist_ok=True)
    mean_scores = aggregate_mean_scores(scores)
    mean_softmax = aggregate_mean_softmax(scores)
    cluster_mass = aggregate_cluster_mass(scores)
    mean_scores.to_parquet(out / "mean_scores.parquet", index=False)
    mean_softmax.to_parquet(out / "mean_softmax.parquet", index=False)
    cluster_mass.to_parquet(out / "cluster_mass.parquet", index=False)
    compute_model_deltas(cluster_mass, "cluster_mass").to_parquet(out / "model_deltas.parquet", index=False)
    mark_completed(out, {"num_score_rows": len(scores)})
    write_status(rd, "completed", "aggregates", {"vector_type": vector_type})


if __name__ == "__main__":
    main()


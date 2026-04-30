#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors.numpy import load_file

from persona_exp.alignment import compute_pc_alignment, compute_role_pair_alignment, compute_same_role_cosines
from persona_exp.config import get_models, load_config, run_dir
from persona_exp.io import mark_completed, write_status


def load_vectors(rd, model_name, layer_tag, vector_type):
    root = rd / "role_vectors" / f"model={model_name}" / f"layer={layer_tag}"
    vec = load_file(str(root / f"role_vectors_{vector_type}.safetensors"))["role_vectors"]
    meta = pd.read_parquet(root / f"role_vector_metadata_{vector_type}.parquet")
    return meta, vec


def save_role_pair_heatmap(pair_alignment: pd.DataFrame, path, title: str, value_col: str = "cosine") -> None:
    if pair_alignment.empty:
        return
    pivot = pair_alignment.pivot_table(
        index="role_id_a",
        columns="role_id_b",
        values=value_col,
        aggfunc="mean",
    )
    roles = sorted(set(pivot.index) & set(pivot.columns))
    if roles:
        pivot = pivot.reindex(index=roles, columns=roles)
    vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig_size = max(8, len(pivot.index) * 0.24)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(pivot.fillna(0).values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=75, ha="right", fontsize=6)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
    plt.xlabel("model B role vector")
    plt.ylabel("model A role vector")
    plt.colorbar(label=value_col)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_same_role_cosine_bar(cosines: pd.DataFrame, path, title: str) -> None:
    if cosines.empty:
        return
    plot_df = cosines.sort_values("cosine", ascending=False).reset_index(drop=True)
    colors = ["#b45f5f" if value < 0 else "#4c78a8" for value in plot_df["cosine"]]
    plt.figure(figsize=(max(10, len(plot_df) * 0.28), 5.5))
    plt.bar(range(len(plot_df)), plot_df["cosine"], color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel("same-role cosine")
    plt.xticks(range(len(plot_df)), plot_df["role_id"], rotation=75, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


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
        figure_dir = pair_dir / "figures"
        pair_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        for layer_dir in layer_dirs:
            layer_tag = layer_dir.name.split("=", 1)[1]
            meta_a, vec_a = load_vectors(rd, a.name, layer_tag, vector_type)
            meta_b, vec_b = load_vectors(rd, b.name, layer_tag, vector_type)
            cos = compute_same_role_cosines(meta_a, vec_a, meta_b, vec_b)
            cos.to_parquet(pair_dir / f"same_role_cosines_{layer_tag}.parquet", index=False)
            save_same_role_cosine_bar(
                cos,
                figure_dir / f"same_role_cosines_{layer_tag}.png",
                f"Same-role vector cosines: {a.name} vs {b.name}, layer={layer_tag}",
            )
            pair_alignment = compute_role_pair_alignment(meta_a, vec_a, meta_b, vec_b)
            pair_alignment.to_parquet(pair_dir / f"role_pair_alignment_{layer_tag}.parquet", index=False)
            save_role_pair_heatmap(
                pair_alignment,
                figure_dir / f"role_pair_cosine_heatmap_{layer_tag}.png",
                f"Role-vector cosine matrix: {a.name} vs {b.name}, layer={layer_tag}",
                value_col="cosine",
            )
            save_role_pair_heatmap(
                pair_alignment,
                figure_dir / f"role_pair_dot_heatmap_{layer_tag}.png",
                f"Role-vector dot matrix: {a.name} vs {b.name}, layer={layer_tag}",
                value_col="dot",
            )
            pc = compute_pc_alignment(vec_a, vec_b, n_components=3)
            with (pair_dir / f"pc_alignment_{layer_tag}.json").open("w", encoding="utf-8") as f:
                json.dump(pc, f, indent=2, sort_keys=True)
    mark_completed(rd / "alignment")
    write_status(rd, "completed", "alignment", {"vector_type": vector_type})


if __name__ == "__main__":
    main()

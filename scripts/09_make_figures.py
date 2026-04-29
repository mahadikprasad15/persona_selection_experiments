#!/usr/bin/env python
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from persona_exp.config import load_config, run_dir
from persona_exp.io import mark_completed, write_status


def _save_heatmap(
    pivot: pd.DataFrame,
    path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    if pivot.empty:
        return
    values = pivot.fillna(0).values
    kwargs = {"aspect": "auto", "cmap": cmap}
    if center_zero:
        vmax = abs(values).max()
        if vmax > 0:
            kwargs.update({"vmin": -vmax, "vmax": vmax})
    plt.figure(figsize=(max(7, len(pivot.columns) * 0.75), max(4, len(pivot.index) * 0.55)))
    plt.imshow(values, **kwargs)
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=colorbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_category_cluster_heatmaps(
    df: pd.DataFrame,
    figures,
    value_col: str,
    file_prefix: str,
    colorbar_label: str,
    title_prefix: str,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> int:
    group_cols = [c for c in ["model_name", "layer_tag", "site", "delta"] if c in df.columns]
    count = 0
    for group_values, group in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        parts = dict(zip(group_cols, group_values))
        pivot = group.pivot_table(
            index="prompt_category",
            columns="role_cluster",
            values=value_col,
            aggfunc="mean",
        )
        label = "_".join(str(parts[c]) for c in group_cols)
        title_bits = ", ".join(f"{c}={parts[c]}" for c in group_cols)
        _save_heatmap(
            pivot,
            figures / f"{file_prefix}_{label}.png",
            f"{title_prefix}: {title_bits}",
            colorbar_label,
            cmap=cmap,
            center_zero=center_zero,
        )
        count += 1
    return count


def _plot_category_role_heatmaps(
    df: pd.DataFrame,
    figures,
    value_col: str,
    file_prefix: str,
    colorbar_label: str,
    title_prefix: str,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> int:
    group_cols = [c for c in ["model_name", "layer_tag", "site"] if c in df.columns]
    count = 0
    for group_values, group in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        parts = dict(zip(group_cols, group_values))
        pivot = group.pivot_table(
            index="prompt_category",
            columns="role_id",
            values=value_col,
            aggfunc="mean",
        )
        label = "_".join(str(parts[c]) for c in group_cols)
        title_bits = ", ".join(f"{c}={parts[c]}" for c in group_cols)
        _save_heatmap(
            pivot,
            figures / f"{file_prefix}_{label}.png",
            f"{title_prefix}: {title_bits}",
            colorbar_label,
            cmap=cmap,
            center_zero=center_zero,
        )
        count += 1
    return count


def _save_role_bar(
    df: pd.DataFrame,
    path,
    title: str,
    value_col: str,
    ylabel: str,
    center_zero: bool = False,
) -> None:
    if df.empty:
        return
    plot_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    colors = "#4c78a8"
    if center_zero:
        colors = ["#b45f5f" if value < 0 else "#4c78a8" for value in plot_df[value_col]]
    plt.figure(figsize=(max(11, len(plot_df) * 0.28), 5.5))
    plt.bar(range(len(plot_df)), plot_df[value_col], color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(range(len(plot_df)), plot_df["role_id"], rotation=75, ha="right", fontsize=7)
    if center_zero:
        plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_role_bars(
    df: pd.DataFrame,
    figures,
    value_col: str,
    file_prefix: str,
    ylabel: str,
    title_prefix: str,
    center_zero: bool = False,
) -> int:
    group_cols = [c for c in ["model_name", "layer_tag", "site"] if c in df.columns]
    count = 0
    for group_values, group in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        parts = dict(zip(group_cols, group_values))
        label = "_".join(str(parts[c]) for c in group_cols)
        title_bits = ", ".join(f"{c}={parts[c]}" for c in group_cols)

        all_eval = group.groupby("role_id", as_index=False)[value_col].mean()
        _save_role_bar(
            all_eval,
            figures / f"{file_prefix}_all_eval_{label}.png",
            f"{title_prefix}, all eval categories: {title_bits}",
            value_col,
            ylabel,
            center_zero=center_zero,
        )
        count += 1

        for category, category_group in group.groupby("prompt_category"):
            category_df = category_group.groupby("role_id", as_index=False)[value_col].mean()
            _save_role_bar(
                category_df,
                figures / f"{file_prefix}_category={category}_{label}.png",
                f"{title_prefix}, category={category}: {title_bits}",
                value_col,
                ylabel,
                center_zero=center_zero,
            )
            count += 1
    return count


def _plot_sum_role_bars(
    df: pd.DataFrame,
    figures,
    value_col: str,
    file_prefix: str,
    ylabel: str,
    title_prefix: str,
) -> int:
    group_cols = [c for c in ["model_name", "layer_tag", "site"] if c in df.columns]
    count = 0
    for group_values, group in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        parts = dict(zip(group_cols, group_values))
        label = "_".join(str(parts[c]) for c in group_cols)
        title_bits = ", ".join(f"{c}={parts[c]}" for c in group_cols)

        for category, category_group in group.groupby("prompt_category"):
            category_df = category_group.groupby("role_id", as_index=False)[value_col].sum()
            category_label = "all_eval" if category == "all_eval" else f"category={category}"
            title_category = "all eval samples" if category == "all_eval" else f"category={category}"
            _save_role_bar(
                category_df,
                figures / f"{file_prefix}_{category_label}_{label}.png",
                f"{title_prefix}, {title_category}: {title_bits}",
                value_col,
                ylabel,
                center_zero=True,
            )
            count += 1
    return count


def _plot_role_delta_bars(
    df: pd.DataFrame,
    figures,
    value_col: str,
    file_prefix: str,
    ylabel: str,
    title_prefix: str,
) -> int:
    index_cols = [
        c
        for c in ["layer_tag", "site", "prompt_category", "role_id"]
        if c in df.columns
    ]
    pivot = (
        df.pivot_table(
            index=index_cols,
            columns="checkpoint_stage",
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "base" not in pivot.columns or "sft" not in pivot.columns:
        return 0
    pivot["sft_minus_base"] = pivot["sft"] - pivot["base"]
    count = 0
    for (layer, site), group in pivot.groupby(["layer_tag", "site"]):
        all_eval = group.groupby("role_id", as_index=False)["sft_minus_base"].mean()
        _save_role_bar(
            all_eval,
            figures / f"{file_prefix}_all_eval_{layer}_{site}_sft_minus_base.png",
            f"{title_prefix}, all eval categories: layer={layer}, site={site}",
            "sft_minus_base",
            ylabel,
            center_zero=True,
        )
        count += 1

        for category, category_group in group.groupby("prompt_category"):
            category_df = category_group.groupby("role_id", as_index=False)["sft_minus_base"].mean()
            _save_role_bar(
                category_df,
                figures / f"{file_prefix}_category={category}_{layer}_{site}_sft_minus_base.png",
                f"{title_prefix}, category={category}: layer={layer}, site={site}",
                "sft_minus_base",
                ylabel,
                center_zero=True,
            )
            count += 1
    return count


def _plot_sum_role_delta_bars(
    df: pd.DataFrame,
    figures,
    value_col: str,
    file_prefix: str,
    ylabel: str,
    title_prefix: str,
) -> int:
    index_cols = [
        c
        for c in ["layer_tag", "site", "prompt_category", "role_id"]
        if c in df.columns
    ]
    pivot = (
        df.pivot_table(
            index=index_cols,
            columns="checkpoint_stage",
            values=value_col,
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "base" not in pivot.columns or "sft" not in pivot.columns:
        return 0
    pivot["sft_minus_base"] = pivot["sft"] - pivot["base"]
    count = 0
    for (layer, site), group in pivot.groupby(["layer_tag", "site"]):
        for category, category_group in group.groupby("prompt_category"):
            category_df = category_group.groupby("role_id", as_index=False)["sft_minus_base"].sum()
            category_label = "all_eval" if category == "all_eval" else f"category={category}"
            title_category = "all eval samples" if category == "all_eval" else f"category={category}"
            _save_role_bar(
                category_df,
                figures / f"{file_prefix}_{category_label}_{layer}_{site}_sft_minus_base.png",
                f"{title_prefix}, {title_category}: layer={layer}, site={site}",
                "sft_minus_base",
                ylabel,
                center_zero=True,
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rd = run_dir(cfg)
    write_status(rd, "running", "figures")
    figures = rd / "figures"
    reports = rd / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    figure_counts = {}

    cluster_path = rd / "aggregates" / "cluster_mass.parquet"
    if cluster_path.exists():
        df = pd.read_parquet(cluster_path)
        figure_counts["cluster_mass"] = _plot_category_cluster_heatmaps(
            df,
            figures,
            "cluster_mass",
            "cluster_mass",
            "cluster mass",
            "Cluster mass",
        )

    mean_scores_path = rd / "aggregates" / "mean_scores.parquet"
    if mean_scores_path.exists():
        df = pd.read_parquet(mean_scores_path)
        figure_counts["mean_scores_by_cluster"] = _plot_category_cluster_heatmaps(
            df,
            figures,
            "score_dot",
            "mean_score_by_cluster",
            "mean dot score",
            "Mean dot score by role cluster",
            cmap="coolwarm",
            center_zero=True,
        )
        figure_counts["mean_scores_by_role"] = _plot_category_role_heatmaps(
            df,
            figures,
            "score_dot",
            "mean_score_by_role",
            "mean dot score",
            "Mean dot score by role",
            cmap="coolwarm",
            center_zero=True,
        )
        figure_counts["mean_scores_role_bars"] = _plot_role_bars(
            df,
            figures,
            "score_dot",
            "role_bar_mean_score",
            "mean dot score",
            "Mean dot score by role",
            center_zero=True,
        )
        figure_counts["mean_score_role_delta_bars"] = _plot_role_delta_bars(
            df,
            figures,
            "score_dot",
            "role_delta_bar_mean_score",
            "SFT - base mean dot score",
            "SFT - base mean dot score by role",
        )

    sum_scores_path = rd / "aggregates" / "sum_scores.parquet"
    if sum_scores_path.exists():
        df = pd.read_parquet(sum_scores_path)
        figure_counts["sum_scores_role_bars"] = _plot_sum_role_bars(
            df,
            figures,
            "score_dot",
            "role_bar_sum_score",
            "summed dot score",
            "Summed dot score by role",
        )
        figure_counts["sum_score_role_delta_bars"] = _plot_sum_role_delta_bars(
            df,
            figures,
            "score_dot",
            "role_delta_bar_sum_score",
            "SFT - base summed dot score",
            "SFT - base summed dot score by role",
        )

    mean_softmax_path = rd / "aggregates" / "mean_softmax.parquet"
    if mean_softmax_path.exists():
        df = pd.read_parquet(mean_softmax_path)
        figure_counts["mean_softmax_by_cluster"] = _plot_category_cluster_heatmaps(
            df,
            figures,
            "score_softmax_T1",
            "mean_softmax_by_cluster",
            "mean role softmax",
            "Mean role softmax by role cluster",
        )
        figure_counts["mean_softmax_by_role"] = _plot_category_role_heatmaps(
            df,
            figures,
            "score_softmax_T1",
            "mean_softmax_by_role",
            "mean role softmax",
            "Mean role softmax by role",
        )
        figure_counts["mean_softmax_role_bars"] = _plot_role_bars(
            df,
            figures,
            "score_softmax_T1",
            "role_bar_mean_softmax",
            "mean role softmax",
            "Mean role softmax by role",
        )
        figure_counts["mean_softmax_role_delta_bars"] = _plot_role_delta_bars(
            df,
            figures,
            "score_softmax_T1",
            "role_delta_bar_mean_softmax",
            "SFT - base mean role softmax",
            "SFT - base mean role softmax by role",
        )

    deltas_path = rd / "aggregates" / "model_deltas.parquet"
    if deltas_path.exists():
        df = pd.read_parquet(deltas_path)
        figure_counts["model_deltas"] = _plot_category_cluster_heatmaps(
            df,
            figures,
            "cluster_mass",
            "model_delta_cluster_mass",
            "cluster mass delta",
            "Model delta in cluster mass",
            cmap="coolwarm",
            center_zero=True,
        )

    counts_text = "\n".join(f"- {name}: {count}" for name, count in sorted(figure_counts.items()))
    summary = reports / "summary.md"
    summary.write_text(
        "# Persona Selection Experiment Summary\n\n"
        "Generated by `scripts/09_make_figures.py`.\n\n"
        "Figure counts:\n"
        f"{counts_text}\n\n"
        "Interpret scores as prototype-induced persona-evidence profiles, not literal internal persona probabilities.\n",
        encoding="utf-8",
    )
    mark_completed(figures)
    write_status(rd, "completed", "figures")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from persona_exp.config import load_config, run_dir
from persona_exp.io import mark_completed, write_status


DEFAULT_KEY_ROLES = [
    "assistant",
    "tutor",
    "analyst",
    "moderator",
    "narcissist",
    "criminal",
    "ghost",
    "spy",
    "philosopher",
]
CATEGORY_ORDER = [
    "all_eval",
    "neutral",
    "harmful",
    "jailbreak",
    "deception",
    "sycophancy",
    "vulnerable_user",
]


def _load_scores(rd: Path) -> pd.DataFrame:
    paths = sorted((rd / "scores").glob("model=*/vector_set=*/site=*/layer=*/scores.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _cleanup_stale_sum_figures(figures: Path) -> int:
    patterns = [
        "role_bar_sum_score*.png",
        "role_delta_bar_sum_score*.png",
    ]
    count = 0
    for pattern in patterns:
        for path in figures.glob(pattern):
            path.unlink()
            count += 1
    return count


def _ordered_categories(columns) -> list[str]:
    existing = list(columns)
    ordered = [c for c in CATEGORY_ORDER if c in existing]
    ordered.extend(sorted(c for c in existing if c not in ordered))
    return ordered


def _bootstrap_ci(values: pd.Series, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _mean_ci_by_role(scores: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    keys = ["model_name", "checkpoint_stage", "layer_tag", "site", "prompt_category", "role_id"]
    rows = []
    for key_values, group in scores.groupby(keys):
        rec = dict(zip(keys, key_values))
        rec["mean"] = group["score_dot"].mean()
        rec["n"] = group["prompt_id"].nunique()
        rec["ci_low"], rec["ci_high"] = _bootstrap_ci(group["score_dot"], n_boot=n_boot)
        rows.append(rec)

    all_keys = ["model_name", "checkpoint_stage", "layer_tag", "site", "role_id"]
    for key_values, group in scores.groupby(all_keys):
        rec = dict(zip(all_keys, key_values))
        rec["prompt_category"] = "all_eval"
        rec["mean"] = group["score_dot"].mean()
        rec["n"] = group["prompt_id"].nunique()
        rec["ci_low"], rec["ci_high"] = _bootstrap_ci(group["score_dot"], n_boot=n_boot)
        rows.append(rec)
    return pd.DataFrame(rows)


def _paired_delta_by_prompt(scores: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "layer_tag",
        "layer_idx",
        "site",
        "prompt_category",
        "prompt_id",
        "role_id",
        "role_cluster",
    ]
    optional_cols = [c for c in ["prompt_subcategory", "prompt_source"] if c in scores.columns]
    index_cols.extend(optional_cols)
    pivot = (
        scores.pivot_table(
            index=index_cols,
            columns="checkpoint_stage",
            values="score_dot",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "base" not in pivot.columns or "sft" not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["base", "sft"]).copy()
    pivot["sft_minus_base"] = pivot["sft"] - pivot["base"]
    return pivot


def _summarize_paired_delta(delta: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    keys = ["layer_tag", "site", "prompt_category", "role_id"]
    rows = []
    for key_values, group in delta.groupby(keys):
        rec = dict(zip(keys, key_values))
        rec["mean_delta"] = group["sft_minus_base"].mean()
        rec["n"] = group["prompt_id"].nunique()
        rec["ci_low"], rec["ci_high"] = _bootstrap_ci(group["sft_minus_base"], n_boot=n_boot)
        rows.append(rec)

    all_keys = ["layer_tag", "site", "role_id"]
    for key_values, group in delta.groupby(all_keys):
        rec = dict(zip(all_keys, key_values))
        rec["prompt_category"] = "all_eval"
        rec["mean_delta"] = group["sft_minus_base"].mean()
        rec["n"] = group["prompt_id"].nunique()
        rec["ci_low"], rec["ci_high"] = _bootstrap_ci(group["sft_minus_base"], n_boot=n_boot)
        rows.append(rec)
    return pd.DataFrame(rows)


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


def _save_role_bar_with_ci(
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
    yerr = None
    if {"ci_low", "ci_high"} <= set(plot_df.columns):
        lower = (plot_df[value_col] - plot_df["ci_low"]).clip(lower=0)
        upper = (plot_df["ci_high"] - plot_df[value_col]).clip(lower=0)
        yerr = np.vstack([lower.to_numpy(), upper.to_numpy()])
    plt.figure(figsize=(max(11, len(plot_df) * 0.28), 5.8))
    plt.bar(range(len(plot_df)), plot_df[value_col], yerr=yerr, color=colors, capsize=2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(range(len(plot_df)), plot_df["role_id"], rotation=75, ha="right", fontsize=7)
    if center_zero:
        plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_mean_ci_role_bars(mean_ci: pd.DataFrame, figures) -> int:
    count = 0
    group_cols = ["model_name", "checkpoint_stage", "layer_tag", "site", "prompt_category"]
    for group_values, group in mean_ci.groupby(group_cols):
        parts = dict(zip(group_cols, group_values))
        category = parts["prompt_category"]
        category_label = "all_eval" if category == "all_eval" else f"category={category}"
        label = "_".join(str(parts[c]) for c in ["model_name", "layer_tag", "site"])
        _save_role_bar_with_ci(
            group,
            figures / f"role_bar_mean_score_ci_{category_label}_{label}.png",
            "Mean dot score by role with bootstrap CI: "
            f"category={category}, model={parts['model_name']}, layer={parts['layer_tag']}, site={parts['site']}",
            "mean",
            "mean dot score",
            center_zero=True,
        )
        count += 1
    return count


def _save_delta_heatmap(
    df: pd.DataFrame,
    path,
    title: str,
    value_col: str = "mean_delta",
) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index="role_id", columns="prompt_category", values=value_col, aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot[_ordered_categories(pivot.columns)]
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    plt.figure(figsize=(max(7, len(pivot.columns) * 1.05), max(8, len(pivot.index) * 0.30)))
    plt.imshow(pivot.fillna(0).values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    plt.colorbar(label="mean paired SFT - base dot score")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_paired_delta_heatmaps(delta_summary: pd.DataFrame, figures) -> int:
    count = 0
    for (layer, site), group in delta_summary.groupby(["layer_tag", "site"]):
        _save_delta_heatmap(
            group,
            figures / f"paired_delta_heatmap_mean_score_{layer}_{site}_sft_minus_base.png",
            f"Paired SFT - base mean dot score by role/category: layer={layer}, site={site}",
        )
        count += 1
    return count


def _plot_paired_delta_bars(delta_summary: pd.DataFrame, figures) -> int:
    count = 0
    for (layer, site, category), group in delta_summary.groupby(["layer_tag", "site", "prompt_category"]):
        category_label = "all_eval" if category == "all_eval" else f"category={category}"
        _save_role_bar_with_ci(
            group,
            figures / f"paired_delta_bar_mean_score_{category_label}_{layer}_{site}_sft_minus_base.png",
            f"Paired SFT - base mean dot score by role: category={category}, layer={layer}, site={site}",
            "mean_delta",
            "paired SFT - base mean dot score",
            center_zero=True,
        )
        count += 1
    return count


def _plot_base_sft_scatter(scores: pd.DataFrame, figures) -> int:
    keys = ["layer_tag", "site", "prompt_category", "role_id", "checkpoint_stage"]
    mean = scores.groupby(keys, as_index=False)["score_dot"].mean()
    all_eval = scores.groupby(["layer_tag", "site", "role_id", "checkpoint_stage"], as_index=False)[
        "score_dot"
    ].mean()
    all_eval["prompt_category"] = "all_eval"
    mean = pd.concat([mean, all_eval], ignore_index=True)
    pivot = (
        mean.pivot_table(
            index=["layer_tag", "site", "prompt_category", "role_id"],
            columns="checkpoint_stage",
            values="score_dot",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "base" not in pivot.columns or "sft" not in pivot.columns:
        return 0
    count = 0
    for (layer, site, category), group in pivot.dropna(subset=["base", "sft"]).groupby(
        ["layer_tag", "site", "prompt_category"]
    ):
        lim = float(np.nanmax(np.abs(group[["base", "sft"]].to_numpy())))
        if not np.isfinite(lim) or lim == 0:
            lim = 1.0
        plt.figure(figsize=(6.2, 6.0))
        plt.scatter(group["base"], group["sft"], s=22, color="#4c78a8", alpha=0.85)
        plt.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.8)
        for _, row in group.iterrows():
            if abs(row["sft"] - row["base"]) >= group.eval("abs(sft - base)").quantile(0.85):
                plt.text(row["base"], row["sft"], row["role_id"], fontsize=7)
        plt.xlim(-lim * 1.08, lim * 1.08)
        plt.ylim(-lim * 1.08, lim * 1.08)
        plt.xlabel("base mean dot score")
        plt.ylabel("SFT mean dot score")
        plt.title(f"Base vs SFT role scores: category={category}, layer={layer}, site={site}")
        plt.tight_layout()
        category_label = "all_eval" if category == "all_eval" else f"category={category}"
        plt.savefig(figures / f"base_sft_scatter_mean_score_{category_label}_{layer}_{site}.png", dpi=180)
        plt.close()
        count += 1
    return count


def _write_rank_correlations(scores: pd.DataFrame, reports) -> int:
    keys = ["layer_tag", "site", "prompt_category", "role_id", "checkpoint_stage"]
    mean = scores.groupby(keys, as_index=False)["score_dot"].mean()
    all_eval = scores.groupby(["layer_tag", "site", "role_id", "checkpoint_stage"], as_index=False)[
        "score_dot"
    ].mean()
    all_eval["prompt_category"] = "all_eval"
    mean = pd.concat([mean, all_eval], ignore_index=True)
    rows = []
    for (layer, site, category), group in mean.groupby(["layer_tag", "site", "prompt_category"]):
        pivot = group.pivot_table(index="role_id", columns="checkpoint_stage", values="score_dot")
        if {"base", "sft"} <= set(pivot.columns):
            rows.append(
                {
                    "comparison": "base_vs_sft",
                    "layer_tag": layer,
                    "site": site,
                    "prompt_category": category,
                    "spearman": pivot["base"].corr(pivot["sft"], method="spearman"),
                    "num_roles": len(pivot.dropna(subset=["base", "sft"])),
                }
            )
    for (layer, category, stage), group in mean.groupby(["layer_tag", "prompt_category", "checkpoint_stage"]):
        pivot = group.pivot_table(index="role_id", columns="site", values="score_dot")
        if {"assistant_marker_final_token", "gen_mean_20"} <= set(pivot.columns):
            rows.append(
                {
                    "comparison": "assistant_marker_vs_gen_mean",
                    "layer_tag": layer,
                    "site": "both",
                    "checkpoint_stage": stage,
                    "prompt_category": category,
                    "spearman": pivot["assistant_marker_final_token"].corr(
                        pivot["gen_mean_20"], method="spearman"
                    ),
                    "num_roles": len(pivot.dropna(subset=["assistant_marker_final_token", "gen_mean_20"])),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return 0
    out.to_csv(reports / "rank_correlations.csv", index=False)
    return len(out)


def _plot_key_role_distributions(delta: pd.DataFrame, figures, key_roles: list[str] | None = None) -> int:
    roles = key_roles or DEFAULT_KEY_ROLES
    all_eval = delta.copy()
    all_eval["prompt_category"] = "all_eval"
    data = pd.concat([delta, all_eval], ignore_index=True)
    data = data[data["role_id"].isin(roles)]
    if data.empty:
        return 0
    count = 0
    for (layer, site, category), group in data.groupby(["layer_tag", "site", "prompt_category"]):
        role_order = [role for role in roles if role in set(group["role_id"])]
        values = [group.loc[group["role_id"] == role, "sft_minus_base"].dropna().to_numpy() for role in role_order]
        if not values:
            continue
        plt.figure(figsize=(max(8, len(role_order) * 0.65), 5.4))
        plt.violinplot(values, showmeans=True, showextrema=True)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(range(1, len(role_order) + 1), role_order, rotation=45, ha="right")
        plt.ylabel("per-prompt paired SFT - base dot score")
        plt.title(f"Key role paired delta distributions: category={category}, layer={layer}, site={site}")
        plt.tight_layout()
        category_label = "all_eval" if category == "all_eval" else f"category={category}"
        plt.savefig(figures / f"key_role_delta_distribution_{category_label}_{layer}_{site}.png", dpi=180)
        plt.close()
        count += 1
    return count


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rd = run_dir(cfg)
    write_status(rd, "running", "figures")
    figures = rd / "figures"
    reports = rd / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    figure_counts = {}
    figure_counts["removed_stale_sum_figures"] = _cleanup_stale_sum_figures(figures)

    scores = _load_scores(rd)
    if not scores.empty:
        mean_ci = _mean_ci_by_role(scores, n_boot=args.bootstrap_samples)
        mean_ci.to_parquet(reports / "mean_score_role_ci.parquet", index=False)
        figure_counts["mean_score_ci_role_bars"] = _plot_mean_ci_role_bars(mean_ci, figures)

        paired_delta = _paired_delta_by_prompt(scores)
        if not paired_delta.empty:
            paired_delta.to_parquet(reports / "paired_sft_base_score_deltas.parquet", index=False)
            delta_summary = _summarize_paired_delta(paired_delta, n_boot=args.bootstrap_samples)
            delta_summary.to_parquet(reports / "paired_sft_base_score_delta_summary.parquet", index=False)
            figure_counts["paired_delta_heatmaps"] = _plot_paired_delta_heatmaps(delta_summary, figures)
            figure_counts["paired_delta_bars"] = _plot_paired_delta_bars(delta_summary, figures)
            figure_counts["key_role_delta_distributions"] = _plot_key_role_distributions(
                paired_delta,
                figures,
            )

        figure_counts["base_sft_scatter"] = _plot_base_sft_scatter(scores, figures)
        figure_counts["rank_correlation_rows"] = _write_rank_correlations(scores, reports)

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
        "This report intentionally omits summed dot-score role bars. Prefer per-sample means, "
        "paired prompt-level SFT-base deltas, confidence intervals, and rank/stability diagnostics.\n\n"
        "Interpret scores as prototype-induced persona-evidence profiles, not literal internal persona probabilities.\n",
        encoding="utf-8",
    )
    mark_completed(figures)
    write_status(rd, "completed", "figures")


if __name__ == "__main__":
    main()

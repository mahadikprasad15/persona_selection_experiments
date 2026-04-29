from __future__ import annotations

import pandas as pd


def _group_keys(scores: pd.DataFrame, include_role: bool = True) -> list[str]:
    keys = ["model_name", "checkpoint_stage", "layer_tag", "layer_idx", "site", "prompt_category"]
    for optional in ["prompt_subcategory", "prompt_source"]:
        if optional in scores.columns:
            keys.append(optional)
    if include_role:
        keys.extend(["role_id", "role_cluster"])
    else:
        keys.append("role_cluster")
    return keys


def aggregate_mean_scores(scores: pd.DataFrame) -> pd.DataFrame:
    keys = _group_keys(scores, include_role=True)
    return scores.groupby(keys, as_index=False)["score_dot"].mean()


def aggregate_mean_softmax(scores: pd.DataFrame) -> pd.DataFrame:
    keys = _group_keys(scores, include_role=True)
    return scores.groupby(keys, as_index=False)["score_softmax_T1"].mean()


def _sum_group(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return scores.groupby(keys, as_index=False).agg(
        score_dot=("score_dot", "sum"),
        num_samples=("prompt_id", "nunique"),
    )


def aggregate_sum_scores(scores: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_name",
        "checkpoint_stage",
        "layer_tag",
        "layer_idx",
        "site",
        "role_id",
        "role_cluster",
    ]
    category_keys = [*keys, "prompt_category"]
    category_sums = _sum_group(scores, category_keys)
    all_eval = _sum_group(scores, keys)
    all_eval["prompt_category"] = "all_eval"
    return pd.concat([all_eval, category_sums], ignore_index=True)[[*category_keys, "score_dot", "num_samples"]]


def aggregate_cluster_mass(scores: pd.DataFrame) -> pd.DataFrame:
    keys = _group_keys(scores, include_role=False)
    return scores.groupby(keys, as_index=False)["score_softmax_T1"].sum().rename(columns={"score_softmax_T1": "cluster_mass"})


def compute_model_deltas(agg: pd.DataFrame, value_col: str) -> pd.DataFrame:
    keys = [c for c in agg.columns if c not in {"model_name", "checkpoint_stage", value_col}]
    pivot = agg.pivot_table(index=keys, columns="checkpoint_stage", values=value_col, aggfunc="mean").reset_index()
    rows = []
    for _, row in pivot.iterrows():
        base = row.get("base")
        sft = row.get("sft")
        instruct = row.get("instruct")
        for name, value in [
            ("sft_minus_base", None if pd.isna(sft) or pd.isna(base) else sft - base),
            ("instruct_minus_base", None if pd.isna(instruct) or pd.isna(base) else instruct - base),
            ("instruct_minus_sft", None if pd.isna(instruct) or pd.isna(sft) else instruct - sft),
        ]:
            if value is not None:
                rec = {k: row[k] for k in keys}
                rec["delta"] = name
                rec[value_col] = value
                rows.append(rec)
    return pd.DataFrame(rows)

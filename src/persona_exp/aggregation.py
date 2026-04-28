from __future__ import annotations

import pandas as pd


def aggregate_mean_scores(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "checkpoint_stage", "layer_tag", "layer_idx", "site", "prompt_category", "role_id", "role_cluster"]
    return scores.groupby(keys, as_index=False)["score_dot"].mean()


def aggregate_mean_softmax(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "checkpoint_stage", "layer_tag", "layer_idx", "site", "prompt_category", "role_id", "role_cluster"]
    return scores.groupby(keys, as_index=False)["score_softmax_T1"].mean()


def aggregate_cluster_mass(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "checkpoint_stage", "layer_tag", "layer_idx", "site", "prompt_category", "role_cluster"]
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


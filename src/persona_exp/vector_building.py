from __future__ import annotations

import numpy as np
import pandas as pd


def residualize_by_question_template(df: pd.DataFrame, vectors: np.ndarray) -> np.ndarray:
    resid = vectors.astype(np.float32).copy()
    template_col = "instruction_id" if "instruction_id" in df.columns else "template_id"
    keys = list(zip(df["question_id"], df[template_col]))
    for key in sorted(set(keys)):
        idx = np.array([i for i, k in enumerate(keys) if k == key])
        resid[idx] -= resid[idx].mean(axis=0, keepdims=True)
    return resid


def build_role_vectors(df: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    out = []
    for role_id, group in df.groupby("role_id", sort=True):
        idx = group.index.to_numpy()
        out.append(vectors[idx].mean(axis=0))
        first = group.iloc[0]
        rows.append(
            {
                "role_id": role_id,
                "role_name": first["role_name"],
                "role_cluster": first["cluster"],
                "num_examples": int(len(group)),
            }
        )
    return pd.DataFrame(rows), np.vstack(out).astype(np.float32)


def center_role_vectors(vectors: np.ndarray) -> np.ndarray:
    return vectors - vectors.mean(axis=0, keepdims=True)


def build_all_vector_types(df: pd.DataFrame, vectors: np.ndarray):
    raw_meta, raw = build_role_vectors(df, vectors)
    resid_examples = residualize_by_question_template(df, vectors)
    resid_meta, resid = build_role_vectors(df, resid_examples)
    centered = center_role_vectors(resid)
    return {
        "raw": (raw_meta, raw),
        "question_residualized": (resid_meta, resid),
        "question_residualized_centered": (resid_meta, centered),
    }

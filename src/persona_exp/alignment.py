from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from persona_exp.scoring import score_cosine


def compute_same_role_cosines(meta_a: pd.DataFrame, vec_a: np.ndarray, meta_b: pd.DataFrame, vec_b: np.ndarray):
    rows = []
    by_b = {row.role_id: i for i, row in meta_b.reset_index(drop=True).iterrows()}
    for i, row in meta_a.reset_index(drop=True).iterrows():
        j = by_b.get(row.role_id)
        if j is None:
            continue
        cos = score_cosine(vec_a[i : i + 1], vec_b[j : j + 1])[0, 0]
        rows.append({"role_id": row.role_id, "role_cluster": row.role_cluster, "cosine": float(cos)})
    return pd.DataFrame(rows)


def compute_role_pair_alignment(meta_a: pd.DataFrame, vec_a: np.ndarray, meta_b: pd.DataFrame, vec_b: np.ndarray):
    scores = vec_a.astype(np.float32) @ vec_b.astype(np.float32).T
    a_norm = np.maximum(np.linalg.norm(vec_a, axis=1, keepdims=True), 1e-8)
    b_norm = np.maximum(np.linalg.norm(vec_b, axis=1, keepdims=True), 1e-8)
    cosines = (vec_a / a_norm).astype(np.float32) @ (vec_b / b_norm).astype(np.float32).T
    rows = []
    meta_a = meta_a.reset_index(drop=True)
    meta_b = meta_b.reset_index(drop=True)
    for i, role_a in meta_a.iterrows():
        for j, role_b in meta_b.iterrows():
            rows.append(
                {
                    "role_id_a": role_a.role_id,
                    "role_cluster_a": role_a.role_cluster,
                    "role_id_b": role_b.role_id,
                    "role_cluster_b": role_b.role_cluster,
                    "same_role": bool(role_a.role_id == role_b.role_id),
                    "dot": float(scores[i, j]),
                    "cosine": float(cosines[i, j]),
                }
            )
    return pd.DataFrame(rows)


def compute_pc_alignment(vec_a: np.ndarray, vec_b: np.ndarray, n_components: int = 3) -> dict:
    n = min(n_components, vec_a.shape[0], vec_b.shape[0])
    pca_a = PCA(n_components=n).fit(vec_a)
    pca_b = PCA(n_components=n).fit(vec_b)
    rows = {}
    for i in range(n):
        a = pca_a.components_[i]
        b = pca_b.components_[i]
        rows[f"pc{i + 1}_cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return rows

from __future__ import annotations

import numpy as np


def score_dot(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    return h.astype(np.float32) @ v.astype(np.float32).T


def score_cosine(h: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), eps)
    vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), eps)
    return hn.astype(np.float32) @ vn.astype(np.float32).T


def softmax_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


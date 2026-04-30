from __future__ import annotations

import numpy as np


def score_dot(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    return h.astype(np.float32) @ v.astype(np.float32).T


def score_cosine(h: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), eps)
    vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), eps)
    return hn.astype(np.float32) @ vn.astype(np.float32).T


def score_whitened_dot(h: np.ndarray, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    h32 = h.astype(np.float32)
    v32 = v.astype(np.float32)
    fit = np.concatenate([h32, v32], axis=0)
    mean = fit.mean(axis=0, keepdims=True)
    centered = fit - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    denom = np.sqrt(np.maximum((singular_values**2) / max(len(fit) - 1, 1), eps))
    h_white = (h32 - mean) @ vt.T / denom
    v_white = (v32 - mean) @ vt.T / denom
    return h_white.astype(np.float32) @ v_white.astype(np.float32).T


def softmax_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

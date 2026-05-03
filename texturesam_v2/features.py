from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-8


def _zscore_channels(feat: np.ndarray) -> np.ndarray:
    mu = feat.mean(axis=(0, 1), keepdims=True)
    sd = feat.std(axis=(0, 1), keepdims=True)
    return (feat - mu) / (sd + EPS)


def compute_texture_feature_map(image_rgb: np.ndarray) -> np.ndarray:
    """
    Build a compact, training-free texture feature map.

    Channels:
    - LAB color (3)
    - Sobel x/y magnitude (3)
    - Laplacian (1)
    - Gabor bank (8)
    """
    img_u8 = image_rgb.astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(np.maximum(sobel_x * sobel_x + sobel_y * sobel_y, 0.0))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    responses = [
        sobel_x[..., None],
        sobel_y[..., None],
        sobel_mag[..., None],
        lap[..., None],
    ]

    thetas = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    lambdas = [4.0, 8.0]
    for theta in thetas:
        for lambd in lambdas:
            k = cv2.getGaborKernel((9, 9), sigma=2.5, theta=theta, lambd=lambd, gamma=0.7, psi=0.0, ktype=cv2.CV_32F)
            g = cv2.filter2D(gray, cv2.CV_32F, k)
            responses.append(g[..., None])

    texture = np.concatenate(responses, axis=2)
    feat = np.concatenate([lab, texture], axis=2)
    return _zscore_channels(feat).astype(np.float32)


def region_descriptor(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not np.any(m):
        return np.zeros(feature_map.shape[2] * 4, dtype=np.float32)

    pix = feature_map[m]
    mean = pix.mean(axis=0)
    std = pix.std(axis=0)
    q25 = np.quantile(pix, 0.25, axis=0)
    q75 = np.quantile(pix, 0.75, axis=0)
    return np.concatenate([mean, std, q25, q75], axis=0).astype(np.float32)


def region_variance(feature_map: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        return 1.0
    pix = feature_map[m]
    return float(np.var(pix, axis=0).mean())


def mean_feature(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not np.any(m):
        return np.zeros(feature_map.shape[2], dtype=np.float32)
    return feature_map[m].mean(axis=0).astype(np.float32)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    num = float(np.dot(x, y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y) + EPS)
    return num / den

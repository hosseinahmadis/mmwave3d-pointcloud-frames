"""
radar_utils.py
==============
Minimal utility functions for mmWave radar point-cloud exploration.

Public API
----------
  fps_sample(points, n, seed=0)  ->  np.ndarray  (n, 5)
  pad_or_crop(points, n)         ->  np.ndarray  (n, 5)
"""

import numpy as np


def fps_sample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling — reduce a point cloud to exactly *n* points.

    Strategy
    --------
    - N > n : FPS on xyz (cols 0–2) picks n geometrically spread points;
              all 5 feature columns are preserved.
    - N < n : zero-pad to n rows.
    - N == n: return as-is.

    Parameters
    ----------
    points : (N, 5)  float32 array  [x, y, z, intensity, speed]
    n      : target point count
    seed   : starting-point index (0 = deterministic)

    Returns
    -------
    (n, 5)  float32
    """
    cur = points.shape[0]

    if cur == n:
        return points

    if cur < n:
        pad = np.zeros((n - cur, points.shape[1]), dtype=np.float32)
        return np.vstack([points, pad])

    # FPS on xyz
    xyz = points[:, :3].astype(np.float32)
    selected = np.zeros(n, dtype=np.int64)
    selected[0] = seed % cur
    dist = np.full(cur, np.inf, dtype=np.float32)

    for k in range(1, n):
        last_xyz = xyz[selected[k - 1]]
        d = np.sum((xyz - last_xyz) ** 2, axis=1)
        dist = np.minimum(dist, d)
        selected[k] = int(np.argmax(dist))

    return points[selected]


def pad_or_crop(points: np.ndarray, n: int) -> np.ndarray:
    """
    Ensure output is exactly (n, 5).

    - N > n : keep the n points with the highest intensity (col 3).
    - N < n : zero-pad.
    - N == n: return as-is.

    Parameters
    ----------
    points : (N, 5)  float32 array  [x, y, z, intensity, speed]
    n      : target point count

    Returns
    -------
    (n, 5)  float32
    """
    cur = points.shape[0]

    if cur == n:
        return points

    if cur > n:
        order = np.argsort(points[:, 3])[::-1]   # descending intensity
        return points[order[:n]]

    pad = np.zeros((n - cur, points.shape[1]), dtype=np.float32)
    return np.vstack([points, pad])

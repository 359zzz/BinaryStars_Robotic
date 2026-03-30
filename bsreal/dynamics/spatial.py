"""6D spatial algebra utilities (Featherstone [omega, v] convention)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _skew(v: FloatArray) -> FloatArray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def spatial_inertia(mass: float, inertia: np.ndarray, com: np.ndarray) -> FloatArray:
    cx = _skew(com)
    I3 = np.eye(3, dtype=float)
    top_left = inertia + mass * (cx @ cx.T)
    top_right = mass * cx
    bottom_left = mass * cx.T
    bottom_right = mass * I3
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def spatial_transform_inverse(
    rotation: np.ndarray, translation: np.ndarray
) -> FloatArray:
    R = np.asarray(rotation, dtype=float)
    p = np.asarray(translation, dtype=float)
    px = _skew(p)
    Z = np.zeros((3, 3), dtype=float)
    return np.block([[R, Z], [-R @ px, R]])

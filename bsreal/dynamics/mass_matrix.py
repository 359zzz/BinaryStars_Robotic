"""Composite Rigid Body Algorithm (CRBA) for mass matrix computation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from bsreal.dynamics.models import DynamicsIR
from bsreal.dynamics.spatial import spatial_inertia, spatial_transform_inverse

FloatArray = NDArray[np.float64]


def rotation_about_axis(axis, angle: float) -> FloatArray:
    """Rodrigues rotation: R = I + sin(a)*K + (1-cos(a))*K^2."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=float)
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def compute_mass_matrix(ir: DynamicsIR, q: np.ndarray) -> FloatArray:
    """CRBA — returns n x n symmetric positive-definite mass matrix M(q)."""
    n = ir.n_joints
    q = np.asarray(q, dtype=float)
    if q.shape != (n,):
        raise ValueError(f"expected q of shape ({n},), got {q.shape}")

    X_lambda: list[FloatArray] = []
    Ic: list[FloatArray] = []
    S: list[FloatArray] = []

    for i in range(n):
        axis_local = ir.joint_axes_local[i]
        R_joint = rotation_about_axis(axis_local, q[i])

        T_pj = ir.parent_to_joint_transforms[i]
        R_pj = T_pj[:3, :3]
        p_pj = T_pj[:3, 3]

        R_total = R_pj @ R_joint
        X_i = spatial_transform_inverse(R_total.T, p_pj)
        X_lambda.append(X_i)

        Ic_i = spatial_inertia(
            float(ir.link_masses[i]), ir.link_inertias[i], ir.link_com_local[i]
        )
        Ic.append(Ic_i)

        s_i = np.zeros(6, dtype=float)
        s_i[:3] = axis_local
        S.append(s_i)

    for i in range(n - 1, -1, -1):
        parent = ir.parent_indices[i]
        if parent >= 0:
            Ic[parent] = Ic[parent] + X_lambda[i].T @ Ic[i] @ X_lambda[i]

    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        F = Ic[i] @ S[i]
        M[i, i] = float(S[i] @ F)
        j = i
        while ir.parent_indices[j] >= 0:
            F = X_lambda[j].T @ F
            j = ir.parent_indices[j]
            M[i, j] = float(S[j] @ F)
            M[j, i] = M[i, j]

    return M

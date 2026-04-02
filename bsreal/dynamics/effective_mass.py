"""Effective mass matrix for dual-arm grasping.

M_eff = diag(M_arm1, M_arm2) + J_grasp^T @ M_obj @ J_grasp

Cross-arm coupling M_12 = J_1^T M_obj J_2, nonzero iff M_obj != 0 (Theorem 3).

Adapted from BinaryStars_sim/physics/effective_mass.py (standalone version only).
"""

from __future__ import annotations

import numpy as np

from bsreal.dynamics.models import DynamicsIR
from bsreal.dynamics.mass_matrix import compute_mass_matrix
from bsreal.dynamics.kinematics import geometric_jacobian


def compute_M_eff_khatib(
    M_arm1: np.ndarray,
    M_arm2: np.ndarray,
    J1: np.ndarray,
    J2: np.ndarray,
    M_obj_6x6: np.ndarray | None = None,
) -> np.ndarray:
    """Reflected-inertia effective mass matrix for dual-arm grasping.

    M_eff = diag(M_arm1, M_arm2) + J_grasp^T @ M_obj @ J_grasp

    Parameters
    ----------
    M_arm1 : (n1, n1) mass matrix of arm 1
    M_arm2 : (n2, n2) mass matrix of arm 2
    J1 : (6, n1) geometric Jacobian of arm 1 end-effector
    J2 : (6, n2) geometric Jacobian of arm 2 end-effector
    M_obj_6x6 : (6, 6) spatial inertia of grasped object.
                 None or zeros -> block-diagonal (Lemma 3).
    """
    n1 = M_arm1.shape[0]
    n2 = M_arm2.shape[0]
    n_total = n1 + n2

    M_eff = np.zeros((n_total, n_total))
    M_eff[:n1, :n1] = M_arm1.copy()
    M_eff[n1:, n1:] = M_arm2.copy()

    if M_obj_6x6 is None or np.allclose(M_obj_6x6, 0.0):
        return M_eff

    M_obj = np.asarray(M_obj_6x6, dtype=float)
    M_eff[:n1, :n1] += J1.T @ M_obj @ J1
    M_eff[n1:, n1:] += J2.T @ M_obj @ J2
    cross = J1.T @ M_obj @ J2
    M_eff[:n1, n1:] = cross
    M_eff[n1:, :n1] = cross.T

    M_eff = (M_eff + M_eff.T) / 2.0
    return M_eff


def make_object_spatial_inertia(
    mass_kg: float,
    geometry: str = "box",
    dims: tuple[float, ...] = (0.1, 0.1, 0.1),
) -> np.ndarray:
    """Create 6x6 spatial inertia matrix for common object shapes.

    Parameters
    ----------
    mass_kg : object mass
    geometry : 'box', 'cylinder', 'sphere', 'none'
    dims : shape-dependent dimensions
    """
    m = mass_kg
    if m <= 0 or geometry == "none":
        return np.zeros((6, 6))

    if geometry == "box":
        lx, ly, lz = dims
        Ixx = m * (ly**2 + lz**2) / 12.0
        Iyy = m * (lx**2 + lz**2) / 12.0
        Izz = m * (lx**2 + ly**2) / 12.0
    elif geometry == "cylinder":
        r, h = dims
        Ixx = m * (3 * r**2 + h**2) / 12.0
        Iyy = Ixx
        Izz = m * r**2 / 2.0
    elif geometry == "sphere":
        r = dims[0]
        Ixx = Iyy = Izz = 2.0 * m * r**2 / 5.0
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    M_obj = np.zeros((6, 6))
    M_obj[0, 0] = m
    M_obj[1, 1] = m
    M_obj[2, 2] = m
    M_obj[3, 3] = Ixx
    M_obj[4, 4] = Iyy
    M_obj[5, 5] = Izz
    return M_obj


def compute_M_eff_for_dual_arm(
    ir_dual: DynamicsIR,
    q: np.ndarray,
    n_per_arm: int,
    M_obj: np.ndarray | None = None,
) -> np.ndarray:
    """Compute M_eff for a dual-arm system.

    Parameters
    ----------
    ir_dual : DynamicsIR for the full dual-arm (n1 + n2 joints, tree topology)
    q : (n_total,) joint configuration
    n_per_arm : number of joints per arm (assumes symmetric: n1 = n2 = n_per_arm)
    M_obj : (6, 6) object spatial inertia, or None for no object

    Returns
    -------
    M_eff : (n_total, n_total) effective mass matrix
    """
    q = np.asarray(q, dtype=float)
    n_total = 2 * n_per_arm

    # Compute full mass matrix via CRBA
    M_full = compute_mass_matrix(ir_dual, q)
    M_arm1 = M_full[:n_per_arm, :n_per_arm]
    M_arm2 = M_full[n_per_arm:, n_per_arm:]

    # Compute EE Jacobians for each arm
    arm1_range = range(0, n_per_arm)
    arm2_range = range(n_per_arm, n_total)
    J1, _ = geometric_jacobian(ir_dual, q, joint_range=arm1_range)
    J2, _ = geometric_jacobian(ir_dual, q, joint_range=arm2_range)

    return compute_M_eff_khatib(M_arm1, M_arm2, J1, J2, M_obj)

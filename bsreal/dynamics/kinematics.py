"""Forward kinematics and geometric Jacobian for articulated bodies.

Adapted from BinaryStars_sim/physics/kinematics.py.
Works with DynamicsIR — supports both serial chains and tree topologies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bsreal.dynamics.models import DynamicsIR
from bsreal.dynamics.mass_matrix import rotation_about_axis

FloatArray = NDArray[np.float64]


def forward_kinematics(ir: DynamicsIR, q: np.ndarray) -> list[np.ndarray]:
    """Compute world-frame 4x4 transforms for all joints.

    Parameters
    ----------
    ir : DynamicsIR with tree structure
    q : (n_joints,) joint configuration (radians)

    Returns
    -------
    T_world : list of (4, 4) world-frame transforms, one per joint
    """
    n = ir.n_joints
    q = np.asarray(q, dtype=float)
    if q.shape != (n,):
        raise ValueError(f"expected q of shape ({n},), got {q.shape}")

    T_world: list[np.ndarray | None] = [None] * n

    for i in range(n):
        T_pj = np.asarray(ir.parent_to_joint_transforms[i], dtype=float)
        R_q = np.eye(4, dtype=float)
        R_q[:3, :3] = rotation_about_axis(ir.joint_axes_local[i], float(q[i]))
        T_local = T_pj @ R_q

        parent = ir.parent_indices[i]
        if parent < 0:
            T_world[i] = T_local
        else:
            T_world[i] = T_world[parent] @ T_local

    return T_world  # type: ignore[return-value]


def geometric_jacobian(
    ir: DynamicsIR,
    q: np.ndarray,
    joint_range: range | None = None,
    ee_offset: np.ndarray | None = None,
) -> tuple[FloatArray, np.ndarray]:
    """Compute 6 x n geometric Jacobian at end-effector.

    Parameters
    ----------
    ir : DynamicsIR
    q : (n_joints,) joint configuration (radians)
    joint_range : if provided, only compute columns for these joints.
        The Jacobian is still (6, len(joint_range)) but uses the full FK.
        The EE is taken as the last joint in joint_range.
    ee_offset : (3,) offset from EE joint frame to EE point (default: zeros)

    Returns
    -------
    J : (6, n_cols) geometric Jacobian
        Rows 0-2: linear velocity  (Jv = z_i x (p_ee - p_i))
        Rows 3-5: angular velocity (Jw = z_i)
    p_ee : (3,) EE position in world frame
    """
    T_world = forward_kinematics(ir, q)

    if joint_range is None:
        joint_range = range(ir.n_joints)

    joints = list(joint_range)
    ee_joint = joints[-1]

    # EE position
    T_ee = T_world[ee_joint]
    if ee_offset is not None:
        p_ee = T_ee[:3, :3] @ np.asarray(ee_offset) + T_ee[:3, 3]
    else:
        p_ee = T_ee[:3, 3].copy()

    n_cols = len(joints)
    J = np.zeros((6, n_cols), dtype=float)

    for col, i in enumerate(joints):
        T_i = T_world[i]
        R_i = T_i[:3, :3]
        p_i = T_i[:3, 3]
        z_i = R_i @ np.asarray(ir.joint_axes_local[i], dtype=float)
        J[:3, col] = np.cross(z_i, p_ee - p_i)
        J[3:, col] = z_i

    return J, p_ee


def compute_ee_jacobian_single_arm(
    ir_single: DynamicsIR,
    q: np.ndarray,
    ee_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Single-arm EE Jacobian convenience function.

    Returns (6, n_joints) Jacobian matrix only (no p_ee).
    """
    J, _ = geometric_jacobian(ir_single, q, ee_offset=ee_offset)
    return J

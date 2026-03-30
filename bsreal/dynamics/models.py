"""Dynamics IR data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class DynamicsIR:
    """Compiled immutable dynamics IR for mass matrix computation."""

    name: str
    topology: str
    parent_indices: tuple[int, ...]
    tree_depth: int
    mass_matrix_bandwidth: int

    n_joints: int
    joint_names: tuple[str, ...]
    joint_axes_local: FloatArray
    parent_to_joint_transforms: tuple[FloatArray, ...]
    base_transform: FloatArray

    link_masses: FloatArray
    link_inertias: FloatArray
    link_com_local: FloatArray
    gravity: FloatArray

    joint_damping: FloatArray | None = None
    joint_friction: FloatArray | None = None

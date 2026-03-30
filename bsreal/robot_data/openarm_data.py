"""OpenArm v10 parameters and DynamicsIR builders.

Adapted from BinaryStars quantum_embodiment.experiments.openarm_data.
freeze_array -> np.asarray (no immutability needed for experiment scripts).
"""

from __future__ import annotations

import math
import numpy as np

from bsreal.dynamics.models import DynamicsIR

# --- Link parameters from config/arm/v10/inertials.yaml ---

OPENARM_LINK_MASSES = (
    1.1416684646202298, 0.2775092746011571, 1.073863338202347,
    0.6348534566833373, 0.6156588026168502, 0.475202773187987,
    0.4659771327380578,
)

OPENARM_LINK_COMS = (
    (0.0011467657911800769, 3.319987657026362e-05, 0.05395284380736254),
    (0.00839629182351943, -2.0145102027597523e-08, 0.03256649300522363),
    (-0.002104752099628911, 0.0005549085042607548, 0.09047470545721961),
    (-0.0029006831074562967, -0.03030575826634669, 0.06339637422196209),
    (-0.003049665024221911, 0.0008866902457326625, 0.043079803024980934),
    (-0.037136587005447405, 0.00033230528343419053, -9.498374522309838e-05),
    (6.875510271106056e-05, 0.01266175250761268, 0.06951945409987448),
)

OPENARM_LINK_INERTIAS = (
    ((0.001567, -1.0e-06, -2.9e-05), (-1.0e-06, 0.001273, 1e-06), (-2.9e-05, 1e-06, 0.001016)),
    ((0.000359, 1e-06, -0.000109), (1e-06, 0.000376, 1e-06), (-0.000109, 1e-06, 0.000232)),
    ((0.004372, 1e-06, 1.1e-05), (1e-06, 0.004319, -3.6e-05), (1.1e-05, -3.6e-05, 0.000661)),
    ((0.000623, -1.0e-06, -1.9e-05), (-1.0e-06, 0.000511, 3.8e-05), (-1.9e-05, 3.8e-05, 0.000334)),
    ((0.000423, -8.0e-06, 6.0e-06), (-8.0e-06, 0.000445, -6.0e-06), (6.0e-06, -6.0e-06, 0.000324)),
    ((0.000143, 1e-06, 1e-06), (1e-06, 0.000157, 1e-06), (1e-06, 1e-06, 0.000159)),
    ((0.000639, 1e-06, 1e-06), (1e-06, 0.000497, 8.9e-05), (1e-06, 8.9e-05, 0.000342)),
)

OPENARM_JOINT_AXES = (
    (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
)

OPENARM_JOINT_XYZ = (
    (0.0, 0.0, 0.0625), (-0.0301, 0.0, 0.06), (0.0301, 0.0, 0.06625),
    (0.0, 0.0315, 0.15375), (0.0, -0.0315, 0.0955), (0.0375, 0.0, 0.1205),
    (-0.0375, 0.0, 0.0),
)

OPENARM_JOINT_RPY = (
    (0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
)

OPENARM_RIGHT_ARM_MOUNT_XYZ = (0.0, -0.031, 0.698)
OPENARM_RIGHT_ARM_MOUNT_YAW = -math.pi / 2
OPENARM_LEFT_ARM_MOUNT_XYZ = (0.0, 0.031, 0.698)
OPENARM_LEFT_ARM_MOUNT_YAW = math.pi / 2


def _rpy_to_rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr],
    ], dtype=float)


def _make_transform(xyz, rpy):
    T = np.eye(4, dtype=float)
    T[:3, :3] = _rpy_to_rotation(*rpy)
    T[:3, 3] = xyz
    return T


def _build_arm_transforms(*, reflect=1, mount_xyz=None, mount_yaw=0.0):
    transforms = []
    for i in range(7):
        xyz = OPENARM_JOINT_XYZ[i]
        rpy = OPENARM_JOINT_RPY[i]
        rpy_reflected = (reflect * rpy[0], reflect * rpy[1], reflect * rpy[2])
        if i == 0 and mount_xyz is not None:
            T_mount = _make_transform(mount_xyz, (0.0, 0.0, mount_yaw))
            T_joint = _make_transform(xyz, rpy_reflected)
            transforms.append(T_mount @ T_joint)
        else:
            transforms.append(_make_transform(xyz, rpy_reflected))
    return transforms


def _build_arm_arrays(*, reflect=1):
    axes = np.array(OPENARM_JOINT_AXES, dtype=float)
    axes[6] = (0.0, float(reflect), 0.0)
    masses = np.array(OPENARM_LINK_MASSES, dtype=float)
    inertias = np.zeros((7, 3, 3), dtype=float)
    for i, I in enumerate(OPENARM_LINK_INERTIAS):
        inertias[i] = np.array(I, dtype=float)
    coms = np.array(OPENARM_LINK_COMS, dtype=float)
    if reflect == -1:
        coms[:, 1] *= -1
    return axes, masses, inertias, coms


def make_openarm_single_arm_ir(*, gravity=(0.0, 0.0, -9.81)):
    axes, masses, inertias, coms = _build_arm_arrays(reflect=1)
    transforms = _build_arm_transforms(reflect=1)
    return DynamicsIR(
        name="openarm_v10_single_arm", topology="serial",
        parent_indices=(-1, 0, 1, 2, 3, 4, 5), tree_depth=7,
        mass_matrix_bandwidth=7, n_joints=7,
        joint_names=("joint1","joint2","joint3","joint4","joint5","joint6","joint7"),
        joint_axes_local=axes,
        parent_to_joint_transforms=tuple(transforms),
        base_transform=np.eye(4, dtype=float),
        link_masses=masses, link_inertias=inertias,
        link_com_local=coms, gravity=np.array(gravity, dtype=float),
    )


def make_openarm_dual_arm_ir(*, gravity=(0.0, 0.0, -9.81)):
    r_axes, r_masses, r_inertias, r_coms = _build_arm_arrays(reflect=1)
    r_transforms = _build_arm_transforms(
        reflect=1, mount_xyz=OPENARM_RIGHT_ARM_MOUNT_XYZ,
        mount_yaw=OPENARM_RIGHT_ARM_MOUNT_YAW)
    l_axes, l_masses, l_inertias, l_coms = _build_arm_arrays(reflect=-1)
    l_transforms = _build_arm_transforms(
        reflect=-1, mount_xyz=OPENARM_LEFT_ARM_MOUNT_XYZ,
        mount_yaw=OPENARM_LEFT_ARM_MOUNT_YAW)
    axes = np.vstack([r_axes, l_axes])
    masses = np.concatenate([r_masses, l_masses])
    inertias = np.concatenate([r_inertias, l_inertias], axis=0)
    coms = np.vstack([r_coms, l_coms])
    transforms = r_transforms + l_transforms
    parent_indices = (-1, 0, 1, 2, 3, 4, 5, -1, 7, 8, 9, 10, 11, 12)
    return DynamicsIR(
        name="openarm_v10_dual_arm", topology="tree",
        parent_indices=parent_indices, tree_depth=7,
        mass_matrix_bandwidth=14, n_joints=14,
        joint_names=(
            "right_joint1","right_joint2","right_joint3","right_joint4",
            "right_joint5","right_joint6","right_joint7",
            "left_joint1","left_joint2","left_joint3","left_joint4",
            "left_joint5","left_joint6","left_joint7",
        ),
        joint_axes_local=axes,
        parent_to_joint_transforms=tuple(transforms),
        base_transform=np.eye(4, dtype=float),
        link_masses=masses, link_inertias=inertias,
        link_com_local=coms, gravity=np.array(gravity, dtype=float),
    )

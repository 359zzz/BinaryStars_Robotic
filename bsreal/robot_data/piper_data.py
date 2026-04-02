"""Piper 6-DOF robot parameters and DynamicsIR builder.

Adapted from BinaryStars quantum_embodiment.experiments.piper_data.
"""

from __future__ import annotations

import math
import numpy as np

from bsreal.dynamics.models import DynamicsIR

_LINK_MASSES_RAW = (
    0.215052383265765, 0.463914239236335, 0.219942452993132,
    0.131814339939458, 0.134101341225523, 0.00699089613564366,
)
_GRIPPER_BASE_MASS = 0.145318531013916

_LINK_COMS_RAW = (
    (0.000121504734057468, 0.000104632162460536, -0.00438597309559853),
    (0.198666145229743, -0.010926924140076, 0.00142121714502687),
    (-0.0202737662122021, -0.133914995944595, -0.000458682652737356),
    (-9.66635791618542e-05, 0.000876064475651083, -0.00496880904640868),
    (-4.10554118924211e-05, -0.0566486692356075, -0.0037205791677906),
    (-8.82590762930069e-05, 9.0598378529832e-06, -0.002),
)
_GRIPPER_BASE_COM = (-0.000183807162235591, 8.05033155577911e-05, 0.0321436689908876)

_LINK_INERTIAS_RAW = (
    ((0.000109639007860341, 2.50631260865109e-07, -1.89352789149844e-07),
     (2.50631260865109e-07, 9.95612262461418e-05, 1.00634716976093e-08),
     (-1.89352789149844e-07, 1.00634716976093e-08, 0.000116363910317385)),
    ((0.000214137415059993, 7.26120579340088e-05, -9.88224861011274e-07),
     (7.26120579340088e-05, 0.00100030277518254, -1.32818212212246e-06),
     (-9.88224861011274e-07, -1.32818212212246e-06, 0.00104417184176783)),
    ((0.00018953849076141, -8.05719205057736e-06, 5.10255053956334e-07),
     (-8.05719205057736e-06, 7.1424497082494e-05, 8.89044974368937e-07),
     (5.10255053956334e-07, 8.89044974368937e-07, 0.000201212938725775)),
    ((3.96965423235175e-05, -2.32268338444837e-08, -1.14702090783249e-07),
     (-2.32268338444837e-08, 5.13319789853892e-05, 9.92852686264567e-08),
     (-1.14702090783249e-07, 9.92852686264567e-08, 4.14768131680711e-05)),
    ((4.10994130543451e-05, -2.06433983793957e-08, 1.29591347668502e-10),
     (-2.06433983793957e-08, 5.27723004189144e-05, 1.9140716904272e-07),
     (1.29591347668502e-10, 1.9140716904272e-07, 4.60418752810541e-05)),
    ((5.43015540542155e-07, 0.0, 0.0),
     (0.0, 5.43015540542155e-07, 0.0),
     (0.0, 0.0, 1.06738869138926e-06)),
)
_GRIPPER_BASE_INERTIA = (
    (0.000101740348396288, -1.43961090652723e-07, -8.72352812740139e-08),
    (-1.43961090652723e-07, 4.16518088621566e-05, 3.27712901952435e-08),
    (-8.72352812740139e-08, 3.27712901952435e-08, 0.000118691325723675),
)

PIPER_JOINT_XYZ = (
    (0.0, 0.0, 0.123), (0.0, 0.0, 0.0), (0.28503, 0.0, 0.0),
    (-0.021984, -0.25075, 0.0), (0.0, 0.0, 0.0), (8.8259e-05, -0.091, 0.0),
)
PIPER_JOINT_RPY = (
    (0.0, 0.0, 0.0), (1.5708, -0.10095, -3.1416), (0.0, 0.0, -1.759),
    (1.5708, 0.0, 0.0), (-1.5708, 0.0, 0.0), (1.5708, 0.0, 0.0),
)
PIPER_JOINT_AXES = tuple((0.0, 0.0, 1.0) for _ in range(6))


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


def _combine_link6_and_gripper():
    m6 = _LINK_MASSES_RAW[5]
    c6 = np.array(_LINK_COMS_RAW[5])
    I6 = np.array(_LINK_INERTIAS_RAW[5])
    mg = _GRIPPER_BASE_MASS
    cg = np.array(_GRIPPER_BASE_COM)
    Ig = np.array(_GRIPPER_BASE_INERTIA)
    m_total = m6 + mg
    c_combined = (m6 * c6 + mg * cg) / m_total
    def _shift(I_com, mass, d):
        return I_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    I_combined = _shift(I6, m6, c6 - c_combined) + _shift(Ig, mg, cg - c_combined)
    return m_total, c_combined, I_combined


PIPER_RIGHT_ARM_MOUNT_XYZ = (0.0, -0.15, 0.0)
PIPER_RIGHT_ARM_MOUNT_YAW = 0.0
PIPER_LEFT_ARM_MOUNT_XYZ = (0.0, 0.15, 0.0)
PIPER_LEFT_ARM_MOUNT_YAW = math.pi


def _build_arm_data(*, reflect=1, mount_xyz=None, mount_yaw=0.0):
    """Build axes, masses, inertias, coms, transforms for one Piper arm."""
    n = 6
    axes = np.array(PIPER_JOINT_AXES, dtype=float)
    m6_combined, c6_combined, I6_combined = _combine_link6_and_gripper()
    masses = np.array(list(_LINK_MASSES_RAW[:5]) + [m6_combined], dtype=float)
    coms = np.zeros((n, 3), dtype=float)
    for i in range(5):
        coms[i] = _LINK_COMS_RAW[i]
    coms[5] = c6_combined
    if reflect == -1:
        coms[:, 1] *= -1
    inertias = np.zeros((n, 3, 3), dtype=float)
    for i in range(5):
        inertias[i] = np.array(_LINK_INERTIAS_RAW[i])
    inertias[5] = I6_combined
    transforms = []
    for i in range(n):
        rpy = PIPER_JOINT_RPY[i]
        rpy_r = (reflect * rpy[0], reflect * rpy[1], reflect * rpy[2])
        if i == 0 and mount_xyz is not None:
            T_mount = _make_transform(mount_xyz, (0.0, 0.0, mount_yaw))
            T_joint = _make_transform(PIPER_JOINT_XYZ[i], rpy_r)
            transforms.append(T_mount @ T_joint)
        else:
            transforms.append(_make_transform(PIPER_JOINT_XYZ[i], rpy_r))
    return axes, masses, inertias, coms, transforms


def make_piper_single_arm_ir(*, gravity=(0.0, 0.0, -9.81)):
    axes, masses, inertias, coms, transforms = _build_arm_data(reflect=1)
    return DynamicsIR(
        name="piper_single_arm", topology="serial",
        parent_indices=(-1, 0, 1, 2, 3, 4), tree_depth=6,
        mass_matrix_bandwidth=6, n_joints=6,
        joint_names=("joint1","joint2","joint3","joint4","joint5","joint6"),
        joint_axes_local=axes,
        parent_to_joint_transforms=tuple(transforms),
        base_transform=np.eye(4, dtype=float),
        link_masses=masses, link_inertias=inertias,
        link_com_local=coms, gravity=np.array(gravity, dtype=float),
    )


def make_piper_dual_arm_ir(*, gravity=(0.0, 0.0, -9.81), arm_separation=0.3):
    """12-DOF tree topology (6+6) dual-arm Piper.

    parent_indices = (-1, 0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10)
    Two independent roots at indices 0 and 6.
    """
    mount_r = (0.0, -arm_separation / 2, 0.0)
    mount_l = (0.0, arm_separation / 2, 0.0)
    r_axes, r_masses, r_inertias, r_coms, r_transforms = _build_arm_data(
        reflect=1, mount_xyz=mount_r, mount_yaw=PIPER_RIGHT_ARM_MOUNT_YAW,
    )
    l_axes, l_masses, l_inertias, l_coms, l_transforms = _build_arm_data(
        reflect=-1, mount_xyz=mount_l, mount_yaw=PIPER_LEFT_ARM_MOUNT_YAW,
    )
    axes = np.vstack([r_axes, l_axes])
    masses = np.concatenate([r_masses, l_masses])
    inertias = np.concatenate([r_inertias, l_inertias], axis=0)
    coms = np.vstack([r_coms, l_coms])
    transforms = r_transforms + l_transforms
    parent_indices = (-1, 0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10)
    return DynamicsIR(
        name="piper_dual_arm", topology="tree",
        parent_indices=parent_indices, tree_depth=6,
        mass_matrix_bandwidth=12, n_joints=12,
        joint_names=(
            "right_joint1","right_joint2","right_joint3",
            "right_joint4","right_joint5","right_joint6",
            "left_joint1","left_joint2","left_joint3",
            "left_joint4","left_joint5","left_joint6",
        ),
        joint_axes_local=axes,
        parent_to_joint_transforms=tuple(transforms),
        base_transform=np.eye(4, dtype=float),
        link_masses=masses, link_inertias=inertias,
        link_com_local=coms, gravity=np.array(gravity, dtype=float),
    )

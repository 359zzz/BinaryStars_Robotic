"""Dual-arm controllers for coordination experiments.

4 controllers with increasing coupling awareness:
1. Decoupled:  independent PID, no coupling compensation
2. J-coupled:  uses M_arm coupling J_ij (cross-arm = 0 always)
3. C-coupled:  uses M_eff coupling (cross-arm > 0 when object present)
4. S-adaptive: switches between Decoupled/C-coupled based on S(rho_L)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from bsreal.dynamics.models import DynamicsIR
from bsreal.dynamics.mass_matrix import compute_mass_matrix
from bsreal.dynamics.coupling import normalized_coupling_matrix
from bsreal.dynamics.effective_mass import compute_M_eff_for_dual_arm


class BaseController(ABC):
    """Base class for dual-arm controllers."""

    def __init__(self, ir: DynamicsIR, n_per_arm: int = 7, robot_type: str = "openarm"):
        self.ir = ir
        self.n_per_arm = n_per_arm
        self.n_total = 2 * n_per_arm
        self.robot_type = robot_type  # "openarm" (MIT torque) or "piper" (position only)

    @abstractmethod
    def compute_action(
        self,
        t: float,
        q_current_rad: np.ndarray,
        qdot_current: np.ndarray,
        q_target_rad: np.ndarray,
        qdot_target: np.ndarray,
    ) -> np.ndarray:
        """Compute control output.

        Returns
        -------
        For OpenArm (MIT mode): torque feedforward (n_total,) in Nm.
            Added to the position PD loop: tau = kp*(q_t-q) + kd*(qd_t-qd) + tau_ff
        For Piper (position only): position compensation (n_total,) in rad.
            Applied as: q_cmd = q_target + q_compensation
        """

    def get_controller_name(self) -> str:
        return self.__class__.__name__

    def _compute_coupling_matrix(self, q_rad: np.ndarray) -> np.ndarray:
        """Compute J_ij from M_arm (no cross-arm coupling)."""
        M = compute_mass_matrix(self.ir, q_rad)
        return normalized_coupling_matrix(M)


class DecoupledController(BaseController):
    """Independent PID, no coupling compensation.

    OpenArm: kp=high, kd=default, tau_ff=0.
    Piper: q_cmd = q_target (no compensation).
    """

    def compute_action(self, t, q_current_rad, qdot_current,
                       q_target_rad, qdot_target):
        return np.zeros(self.n_total)


class JCoupledController(BaseController):
    """Coupling compensation from M_arm's J_ij.

    Cross-arm J_ij = 0 always (separate kinematic trees -> block-diagonal M).
    This represents the standard engineering approach: good within-arm
    compensation but CANNOT compensate cross-arm coupling even with object.

    Compensation:
        tau_ff_i = sum_{j!=i} sign(M_ij) * |J_ij| * [kp_c * e_j + kd_c * edot_j]
    """

    def __init__(self, ir, n_per_arm=7, robot_type="openarm",
                 kp_comp=2.0, kd_comp=0.1, alpha_pos=0.3):
        super().__init__(ir, n_per_arm, robot_type)
        self.kp_comp = kp_comp
        self.kd_comp = kd_comp
        self.alpha_pos = alpha_pos  # position compensation gain for Piper

    def compute_action(self, t, q_current_rad, qdot_current,
                       q_target_rad, qdot_target):
        J_mat = self._compute_coupling_matrix(q_current_rad)
        return self._coupling_compensation(
            J_mat, q_current_rad, qdot_current, q_target_rad, qdot_target,
        )

    def _coupling_compensation(self, J_mat, q_current, qdot_current,
                               q_target, qdot_target):
        e = q_target - q_current
        edot = qdot_target - qdot_current
        n = self.n_total
        output = np.zeros(n)

        for i in range(n):
            tau_ff = 0.0
            for j in range(n):
                if i == j:
                    continue
                J_ij = J_mat[i, j]
                if abs(J_ij) < 1e-6:
                    continue
                sign_ij = 1.0 if J_ij > 0 else -1.0
                if self.robot_type == "openarm":
                    tau_ff += sign_ij * abs(J_ij) * (
                        self.kp_comp * e[j] + self.kd_comp * edot[j]
                    )
                else:
                    # Piper: position compensation
                    tau_ff += sign_ij * abs(J_ij) * self.alpha_pos * e[j]
            output[i] = tau_ff

        return output


class CCoupledController(JCoupledController):
    """Coupling compensation from M_eff (includes cross-arm coupling via object).

    When M_obj != 0: J_cross > 0 (Theorem 3) -> compensates cross-arm dynamics.
    When M_obj == 0: falls back to J-coupled behavior.
    """

    def __init__(self, ir, n_per_arm=7, robot_type="openarm",
                 kp_comp=2.0, kd_comp=0.1, alpha_pos=0.3,
                 M_obj=None):
        super().__init__(ir, n_per_arm, robot_type, kp_comp, kd_comp, alpha_pos)
        self.M_obj = M_obj  # (6,6) object spatial inertia

    def set_object(self, M_obj: np.ndarray | None):
        """Update grasped object (call when grasp state changes)."""
        self.M_obj = M_obj

    def compute_action(self, t, q_current_rad, qdot_current,
                       q_target_rad, qdot_target):
        if self.M_obj is not None and not np.allclose(self.M_obj, 0.0):
            M_eff = compute_M_eff_for_dual_arm(
                self.ir, q_current_rad, self.n_per_arm, self.M_obj,
            )
            J_mat = normalized_coupling_matrix(M_eff)
        else:
            J_mat = self._compute_coupling_matrix(q_current_rad)

        return self._coupling_compensation(
            J_mat, q_current_rad, qdot_current, q_target_rad, qdot_target,
        )


class SAdaptiveController(BaseController):
    """Adaptive controller that blends around an entanglement-entropy gate.

    Low-entropy contexts stay near Decoupled. High-entropy contexts approach
    C-coupled. Around the threshold we use a smooth gate instead of a hard
    switch so the controller remains locally identifiable and avoids abrupt
    mode flips on real hardware.
    """

    def __init__(self, ir, n_per_arm=7, robot_type="openarm",
                 kp_comp=2.0, kd_comp=0.1, alpha_pos=0.3,
                 M_obj=None, s_threshold=1.75, transition_width=0.5, recompute_every=10):
        super().__init__(ir, n_per_arm, robot_type)
        self._decoupled = DecoupledController(ir, n_per_arm, robot_type)
        self._c_coupled = CCoupledController(
            ir, n_per_arm, robot_type, kp_comp, kd_comp, alpha_pos, M_obj,
        )
        self.s_threshold = s_threshold
        self.transition_width = max(1e-6, float(transition_width))
        self.recompute_every = recompute_every
        self._step_count = 0
        self._current_S = 0.0
        self._current_weight = 0.0
        self._using_coupled = False

    def set_object(self, M_obj: np.ndarray | None):
        self._c_coupled.set_object(M_obj)

    @property
    def current_S(self) -> float:
        return self._current_S

    @property
    def using_coupled(self) -> bool:
        return self._using_coupled

    @property
    def current_weight(self) -> float:
        return self._current_weight

    def compute_action(self, t, q_current_rad, qdot_current,
                       q_target_rad, qdot_target):
        self._step_count += 1

        # Recompute S periodically (expensive)
        if self._step_count % self.recompute_every == 1:
            self._current_S = self._compute_entropy(q_current_rad)
            self._current_weight = self._blend_weight(self._current_S)
            self._using_coupled = self._current_weight >= 0.5

        decoupled = self._decoupled.compute_action(
            t, q_current_rad, qdot_current, q_target_rad, qdot_target,
        )
        coupled = self._c_coupled.compute_action(
            t, q_current_rad, qdot_current, q_target_rad, qdot_target,
        )
        return ((1.0 - self._current_weight) * decoupled) + (self._current_weight * coupled)

    def _compute_entropy(self, q_rad: np.ndarray) -> float:
        """Compute von Neumann entropy S(rho_L) of left-arm subsystem.

        Uses the coupling matrix to estimate entanglement.
        S = -sum_i lambda_i * log2(lambda_i) where lambda_i are eigenvalues
        of the reduced density matrix approximated from the coupling structure.
        """
        if self._c_coupled.M_obj is not None and not np.allclose(self._c_coupled.M_obj, 0.0):
            M_eff = compute_M_eff_for_dual_arm(
                self.ir, q_rad, self.n_per_arm, self._c_coupled.M_obj,
            )
        else:
            M_eff = compute_mass_matrix(self.ir, q_rad)

        J_mat = normalized_coupling_matrix(M_eff)
        n = self.n_per_arm

        # Cross-arm coupling block
        cross = J_mat[:n, n:]
        # Singular values of cross-arm block approximate entanglement
        sv = np.linalg.svd(cross, compute_uv=False)
        sv = sv[sv > 1e-10]
        if len(sv) == 0:
            return 0.0

        # Normalize to probabilities
        sv2 = sv**2
        sv2 = sv2 / sv2.sum()
        S = -np.sum(sv2 * np.log2(sv2 + 1e-15))
        return float(S)

    def _blend_weight(self, entropy: float) -> float:
        half_width = 0.5 * self.transition_width
        lower = self.s_threshold - half_width
        upper = self.s_threshold + half_width
        if entropy <= lower:
            return 0.0
        if entropy >= upper:
            return 1.0
        alpha = (entropy - lower) / max(upper - lower, 1e-9)
        return float(alpha * alpha * (3.0 - 2.0 * alpha))


# ── Factory ──────────────────────────────────────────────────────────────────

CONTROLLER_REGISTRY = {
    "decoupled": DecoupledController,
    "j_coupled": JCoupledController,
    "c_coupled": CCoupledController,
    "s_adaptive": SAdaptiveController,
}


def make_controller(
    name: str,
    ir: DynamicsIR,
    n_per_arm: int = 7,
    robot_type: str = "openarm",
    M_obj: np.ndarray | None = None,
    **kwargs,
) -> BaseController:
    """Create a controller by name."""
    cls = CONTROLLER_REGISTRY[name]
    if name in ("c_coupled", "s_adaptive"):
        return cls(ir, n_per_arm, robot_type, M_obj=M_obj, **kwargs)
    elif name == "j_coupled":
        return cls(ir, n_per_arm, robot_type, **kwargs)
    else:
        return cls(ir, n_per_arm, robot_type)

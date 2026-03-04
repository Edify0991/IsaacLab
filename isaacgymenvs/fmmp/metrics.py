from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SwitchingMetrics:
    fall_rate: float
    mean_survival_time: float
    joint_jerk_peak: float
    joint_jerk_integral: float
    com_jerk_peak: float
    foot_slip_distance: float
    ground_penetration_count: int
    torque_peak: float
    torque_rate_peak: float


def _jerk(signal: np.ndarray, dt: float) -> np.ndarray:
    vel = np.gradient(signal, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return np.gradient(acc, dt, axis=0)


def compute_switching_metrics(
    joint_pos: np.ndarray,
    com_pos: np.ndarray,
    foot_pos: np.ndarray,
    foot_contact: np.ndarray,
    foot_heights: np.ndarray,
    torques: np.ndarray,
    falls: np.ndarray,
    survival_times: np.ndarray,
    dt: float,
) -> SwitchingMetrics:
    joint_j = _jerk(joint_pos, dt)
    com_j = _jerk(com_pos, dt)
    torque_rate = np.gradient(torques, dt, axis=0)

    slip = np.linalg.norm(np.diff(foot_pos[..., :2], axis=0), axis=-1)
    slip *= foot_contact[1:]

    return SwitchingMetrics(
        fall_rate=float(np.mean(falls)),
        mean_survival_time=float(np.mean(survival_times)),
        joint_jerk_peak=float(np.max(np.linalg.norm(joint_j, axis=-1))),
        joint_jerk_integral=float(np.sum(np.linalg.norm(joint_j, axis=-1)) * dt),
        com_jerk_peak=float(np.max(np.linalg.norm(com_j, axis=-1))),
        foot_slip_distance=float(np.sum(slip)),
        ground_penetration_count=int(np.sum(foot_heights < 0.0)),
        torque_peak=float(np.max(np.abs(torques))),
        torque_rate_peak=float(np.max(np.abs(torque_rate))),
    )

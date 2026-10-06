"""Exact two-point scattering-weighted distribution ([uri2x-30]).

Port of ``bimodal_skewed_discrete.m`` (``benchmark_skewness.m``/``Scatter2D.m``
in the Overleaf paper folder, added 2026-10-05). Unlike every other
distribution family in this package (including this module's own
``skewed_ensemble`` in ``distributions.py``, [uri2x-27]), this targets the
*scattering-weighted* moments directly and exactly, not the number-weighted
ones -- matching the paper's own stated convention (Sec. "Scattering off a
polydisperse ensemble": "the variance relevant to the present analysis is
V, defined from that weighted distribution"). With only two points, three
free parameters (two positions, one probability split) exactly solve three
target moments (weighted mean=1, weighted variance=V, weighted standardized
skewness=gamma) in closed form -- no approximation, no calibration search
needed, unlike the N-point Cornish-Fisher construction.
"""

from __future__ import annotations

import math

import numpy as np


class BimodalSkewedInfeasible(RuntimeError):
    """Raised when the requested (V, gamma) cannot be realized under the
    min_radius / min_probability constraints (mirrors Scatter2D.m raising
    Scatter2D:InfeasibleBimodalSkewed)."""


def bimodal_skewed_discrete(
    v_rel: float,
    gamma: float,
    weight_power: float = 0.0,
    min_radius: float = 0.5,
    min_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exact two-point distribution with prescribed scattering-weighted moments.

    The ``r**weight_power``-weighted distribution has exactly: weighted mean
    1, weighted central variance ``v_rel``, weighted standardized skewness
    ``gamma``. Returns ``(x, p, info)`` where ``p`` are NUMBER probabilities
    (multiply by ``x**weight_power`` and renormalize to get the weighted
    ensemble, as the rest of this package's simulation pipeline already
    does for every family).

    Raises :class:`BimodalSkewedInfeasible` if the two-point construction
    would violate ``min_radius`` (a point too close to/below zero) or
    ``min_probability`` (a number-probability below the floor once the
    weighted probabilities are converted back to number space) -- both
    exact-translations of the MATLAB function's own feasibility checks.
    """
    if not (math.isfinite(v_rel) and 0 <= v_rel < 1):
        raise ValueError("v_rel must satisfy 0 <= v_rel < 1.")
    if not math.isfinite(gamma):
        raise ValueError("gamma must be a finite scalar.")
    if not (math.isfinite(weight_power) and weight_power >= 0):
        raise ValueError("weight_power must be finite and nonnegative.")
    if not (0 < min_radius < 1):
        raise ValueError("min_radius must satisfy 0 < min_radius < 1.")
    if not (math.isfinite(min_probability) and 0 <= min_probability and 2 * min_probability < 1):
        raise ValueError("min_probability must satisfy 0 <= 2*min_probability < 1.")

    info: dict = {
        "feasible": False, "reason": "",
        "weighted_probabilities": np.array([np.nan, np.nan]),
        "actual_mean": np.nan, "actual_variance": np.nan, "actual_gamma": np.nan,
    }

    if v_rel == 0.0:
        if gamma != 0.0:
            raise ValueError("For v_rel=0, skewness is undefined; use gamma=0.")
        x = np.array([1.0, 1.0])
        p = np.array([0.5, 0.5])
        info.update(feasible=True, weighted_probabilities=np.array([0.5, 0.5]),
                    actual_mean=1.0, actual_variance=0.0, actual_gamma=0.0)
        return x, p, info

    s = math.sqrt(v_rel)
    D = math.sqrt(gamma**2 + 4.0)
    a = 0.5 * s * (D - gamma)
    b = 0.5 * s * (D + gamma)
    x = np.array([1.0 - a, 1.0 + b])

    if not np.all(np.isfinite(x)) or np.any(x < min_radius):
        info["reason"] = f"Minimum radius failed: radii={x.tolist()}, minimum={min_radius}."
        raise BimodalSkewedInfeasible(info["reason"])

    q = np.array([b / (a + b), a / (a + b)])  # weighted probabilities
    raw = q / (x**weight_power)
    p = raw / raw.sum()

    if not np.all(np.isfinite(p)) or np.any(p < min_probability):
        info["reason"] = f"Minimum number probability failed: p={p.tolist()}, minimum={min_probability}."
        raise BimodalSkewedInfeasible(info["reason"])

    qw = p * x**weight_power
    qw = qw / qw.sum()
    mu = float(np.sum(qw * x))
    mu2 = float(np.sum(qw * (x - mu) ** 2))
    mu3 = float(np.sum(qw * (x - mu) ** 3))
    gamma_actual = mu3 / mu2**1.5

    tol = 5e-10
    if abs(mu - 1) > tol or abs(mu2 - v_rel) > tol or abs(gamma_actual - gamma) > tol:
        raise RuntimeError(
            f"Weighted moment verification failed: mu={mu}, mu2={mu2} (target {v_rel}), "
            f"gamma={gamma_actual} (target {gamma})."
        )

    info.update(feasible=True, weighted_probabilities=q,
                actual_mean=mu, actual_variance=mu2, actual_gamma=gamma_actual)
    return x, p, info

"""Tests for tenor_saxs.bimodal_skewed ([uri2x-30]).

Primary coverage is a cross-check against the MATLAB reference's own saved
results (skewness_benchmark_results2ndorderonly_rg3_rmin0.5_bimodal.mat,
produced by benchmark_skewness.m / Scatter2D.m's dist_type='bimodal_skewed'
in the Overleaf paper folder) -- skipped if that file isn't present, since
it lives outside this repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tenor_saxs.bimodal_skewed import BimodalSkewedInfeasible, bimodal_skewed_discrete

_MAT_PATH = Path(
    "/Users/roybeck/Library/CloudStorage/Dropbox/Apps/Overleaf/Rg  Variance paper/"
    "skewness_benchmark_results2ndorderonly_rg3_rmin0.5_bimodal.mat"
)


def test_v_zero_is_monodisperse():
    x, p, info = bimodal_skewed_discrete(0.0, 0.0)
    np.testing.assert_allclose(x, [1.0, 1.0])
    np.testing.assert_allclose(p, [0.5, 0.5])
    assert info["feasible"]


def test_v_zero_nonzero_gamma_raises():
    with pytest.raises(ValueError):
        bimodal_skewed_discrete(0.0, 1.0)


def test_symmetric_at_gamma_zero():
    """gamma=0 must give a == b, i.e. x = [1-s, 1+s]."""
    x, p, info = bimodal_skewed_discrete(0.09, 0.0, weight_power=0.0)
    s = np.sqrt(0.09)
    np.testing.assert_allclose(sorted(x), [1 - s, 1 + s], atol=1e-12)
    np.testing.assert_allclose(p, [0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("v_rel", [0.01, 0.1, 0.25])
@pytest.mark.parametrize("gamma", [-0.5, 0.0, 0.5, 1.5])
@pytest.mark.parametrize("weight_power", [0, 2, 4, 6])
def test_weighted_moments_hit_exactly(v_rel, gamma, weight_power):
    try:
        x, p, info = bimodal_skewed_discrete(v_rel, gamma, weight_power, min_radius=0.01, min_probability=0.0)
    except BimodalSkewedInfeasible:
        pytest.skip("infeasible at this corner")
    qw = p * x**weight_power
    qw = qw / qw.sum()
    mu = np.sum(qw * x)
    var = np.sum(qw * (x - mu) ** 2)
    skew = np.sum(qw * (x - mu) ** 3) / var**1.5
    assert mu == pytest.approx(1.0, abs=1e-9)
    assert var == pytest.approx(v_rel, abs=1e-9)
    assert skew == pytest.approx(gamma, abs=1e-9)


def test_min_radius_infeasible_raises():
    # Large V, large negative gamma pushes the lower point below a tight floor.
    with pytest.raises(BimodalSkewedInfeasible):
        bimodal_skewed_discrete(0.3, -1.0, weight_power=0.0, min_radius=0.9)


@pytest.mark.skipif(not _MAT_PATH.exists(), reason="MATLAB reference .mat not available in this environment")
def test_reproduces_matlab_reference_exactly():
    import scipy.io as sio

    m = sio.loadmat(str(_MAT_PATH), squeeze_me=True, struct_as_record=False)
    V_grid = m["V_values"]
    gamma_grid = m["gamma_values"]
    wp_values = m["weightPowers"]
    radii_mat = m["normalizedRadii"]
    p_mat = m["numericalProbabilities"]
    run_ok = m["runOK"]

    max_err_r = max_err_p = 0.0
    n_checked = 0
    for ic in range(len(wp_values)):
        for iV, V in enumerate(V_grid):
            for ig, gamma in enumerate(gamma_grid):
                ok_mat = bool(run_ok[iV, ig, ic])
                try:
                    x, p, info = bimodal_skewed_discrete(float(V), float(gamma), float(wp_values[ic]), min_radius=0.5, min_probability=0.0)
                    ok_py = True
                except (BimodalSkewedInfeasible, ValueError, RuntimeError):
                    ok_py = False
                assert ok_mat == ok_py, f"feasibility mismatch at ic={ic} iV={iV} ig={ig}"
                if ok_mat and ok_py:
                    n_checked += 1
                    max_err_r = max(max_err_r, np.max(np.abs(np.sort(x) - np.sort(radii_mat[iV, ig, ic, :]))))
                    max_err_p = max(max_err_p, np.max(np.abs(np.sort(p) - np.sort(p_mat[iV, ig, ic, :]))))
    assert n_checked > 1000
    assert max_err_r < 1e-9
    assert max_err_p < 1e-9

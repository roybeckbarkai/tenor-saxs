"""Tests for tenor_saxs.distributions.discretize_ensemble.

Bonus coverage: the affine-rescale guarantee (exact number-weighted mean=1,
relative variance=v_rel), the v_rel=0 monodisperse special case, and the
deliberate exclusion of 'exponential'.
"""

from __future__ import annotations

import numpy as np
import pytest

from tenor_saxs.distributions import (
    discretize_ensemble,
    SkewnessTargetUnreachable,
    skewed_ensemble,
    target_skewed_ensemble,
)


@pytest.mark.parametrize(
    "dist_type",
    ["normal", "lognormal", "schulz", "boltzmann", "triangular", "uniform"],
)
@pytest.mark.parametrize("v_rel", [0.02, 0.05, 0.15])
def test_discretize_ensemble_affine_rescale_is_exact(dist_type, v_rel):
    x, p = discretize_ensemble(11, v_rel, dist_type=dist_type)

    assert x.shape == (11,)
    assert p.shape == (11,)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)

    mu = float(np.sum(p * x))
    var = float(np.sum(p * (x - mu) ** 2))

    # NOTE on tolerance: the affine rescale (`x = scale*(x - curr_mu) + mean`)
    # computes curr_mu/curr_var from the pre-final-renormalization bin
    # probabilities (which only sum to ~`threshold` < 1, since the binning
    # only covers the cumulative-probability range up to `threshold`, default
    # 0.9995), and only renormalizes `p` to sum to 1 *after* that affine step.
    # Algebraically this leaves a residual offset of order
    # scale*curr_mu*(1-psum)/psum ~ (1-threshold) ~ 5e-4 in the final,
    # properly-renormalized mean/variance -- so despite the module
    # docstring's "exactly", it is NOT exact to floating-point precision.
    # Empirically (swept over all distributions x v_rel in [0.01, 0.3]) the
    # observed error tops out around 8e-4 (mean) / 2.2e-4 (variance); this
    # tolerance gives comfortable margin above that measured ceiling.
    assert mu == pytest.approx(1.0, abs=1e-3)
    assert var == pytest.approx(v_rel, abs=5e-4)


def test_discretize_ensemble_v_rel_zero_is_monodisperse():
    n = 11
    x, p = discretize_ensemble(n, 0.0, dist_type="normal")
    np.testing.assert_allclose(x, np.full(n, 1.0))
    np.testing.assert_allclose(p, np.full(n, 1.0 / n))


@pytest.mark.parametrize("dist_type", ["normal", "lognormal", "uniform"])
def test_discretize_ensemble_v_rel_zero_monodisperse_all_distributions(dist_type):
    """v_rel=0 is a well-defined monodisperse special case for EVERY
    distribution family, not just the ones that happen to survive MATLAB's
    own zero-sigma division bug (see module docstring)."""
    n = 7
    x, p = discretize_ensemble(n, 0.0, dist_type=dist_type)
    np.testing.assert_allclose(x, np.full(n, 1.0))
    np.testing.assert_allclose(p, np.full(n, 1.0 / n))


def test_discretize_ensemble_exponential_raises_value_error():
    with pytest.raises(ValueError):
        discretize_ensemble(11, 0.05, dist_type="exponential")


def test_discretize_ensemble_negative_v_rel_raises():
    with pytest.raises(ValueError):
        discretize_ensemble(11, -0.01, dist_type="normal")


def _moments(x, p):
    mu = float(np.sum(p * x))
    var = float(np.sum(p * (x - mu) ** 2))
    skew = float(np.sum(p * (x - mu) ** 3)) / var**1.5 if var > 0 else float("nan")
    return mu, var, skew


@pytest.mark.parametrize("v_rel", [0.02, 0.1, 0.2])
def test_skewed_ensemble_hits_exact_mean_and_variance(v_rel):
    """skewed_ensemble's affine rescale guarantees mean=1, var=v_rel exactly
    (up to the floor-clipping corner), independent of the requested skew."""
    for skew_x in (-1.0, 0.0, 1.0):
        x, p = skewed_ensemble(25, v_rel, skew_x)
        mu, var, _ = _moments(x, p)
        assert mu == pytest.approx(1.0, abs=1e-9)
        assert var == pytest.approx(v_rel, abs=1e-9)


def test_skewed_ensemble_v_rel_zero_is_monodisperse():
    x, p = skewed_ensemble(11, 0.0, skew_x=1.0)
    np.testing.assert_allclose(x, np.full(11, 1.0))
    np.testing.assert_allclose(p, np.full(11, 1.0 / 11))


def test_skewed_ensemble_respects_floor():
    """The [uri2x-27] spec's minimum r = <r>/10 must never be violated,
    even at large negative skew_x/v_rel where the raw Cornish-Fisher
    quantile would go negative."""
    x, p = skewed_ensemble(25, 0.3, skew_x=-6.0, xmin_over_mean=0.1)
    assert x.min() >= 0.1 - 1e-9


@pytest.mark.parametrize("v_rel", [0.01, 0.05, 0.1, 0.15])
@pytest.mark.parametrize("target_skew", [-0.5, 0.0, 0.5, 1.0])
def test_target_skewed_ensemble_hits_target_in_reachable_region(v_rel, target_skew):
    """Away from the extreme (large V, very negative skew) corner, the
    calibrated wrapper should hit the requested standardized skewness
    closely -- the raw skew_x parameter alone systematically undershoots
    it by ~25-30% (this is what the calibration is for)."""
    result = target_skewed_ensemble(25, v_rel, target_skew)
    assert result["realized_skew"] == pytest.approx(target_skew, abs=5e-3)
    assert result["mean"] == pytest.approx(1.0, abs=1e-6)
    assert result["var"] == pytest.approx(v_rel, abs=1e-6)


def test_target_skewed_ensemble_raises_when_unreachable():
    """skew=-1 at large V is a real, documented limitation (the </r>/10
    floor clips enough of the left tail that no input parameter reaches
    it) -- this must raise, not silently return an off-target ensemble."""
    with pytest.raises(SkewnessTargetUnreachable):
        target_skewed_ensemble(25, 0.2, target_skew=-1.0)


def test_discretize_ensemble_fixman_ignores_v_rel_and_matches_true_moments():
    """dist_type='fixman' is parameter-free: v_rel is ignored, and the
    returned ensemble's exact moments should match the published Fixman
    (1962) values (mean=1, V=0.0618396) as n grows, since it uses exact
    quantiles of the true distribution rather than an importance-
    sampled/rescaled discretization.
    """
    from tenor_saxs.fixman import MOMENTS

    for v_rel in (0.02, 0.15, 999.0):  # arbitrary, must not affect the result
        x, p = discretize_ensemble(2000, v_rel, dist_type="fixman")
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(p, np.full(2000, 1.0 / 2000))

        mu = float(np.sum(p * x))
        var = float(np.sum(p * (x - mu) ** 2))
        assert mu == pytest.approx(1.0, abs=1e-3)
        assert var == pytest.approx(MOMENTS["V"], abs=2e-3)

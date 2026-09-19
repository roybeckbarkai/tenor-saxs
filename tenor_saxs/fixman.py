"""
Fixman (1962) distribution of the radius of gyration of an ideal polymer chain.

Reference
---------
M. Fixman, "Radius of Gyration of Polymer Chains", J. Chem. Phys. 36, 306-310
(1962), doi:10.1063/1.1732501.  (Part II: J. Chem. Phys. 36, 3123, 1962.)

This is the distribution of Rg over the CONFORMATIONS of a single unbranched
ideal (Gaussian / theta) chain -- the random-coil reference ensemble for an IDP.
It is not a polydispersity distribution over different particles.

There is no elementary closed form for the density.  Fixman gives the
characteristic function.  With  Y = Rg^2 / <Rg^2>  (so <Y> = 1):

    Y = (2/pi^2) * sum_{k=1..inf}  chi2_3,k / k^2            (Rouse / KL modes)

    E[exp(-lam Y)] = [ sinh(2 sqrt(lam)) / (2 sqrt(lam)) ]^(-3/2)

Tails (both non-Gaussian):

    P(Rg) ~ exp( -(9/4) <Rg^2>/Rg^2 )        Rg -> 0   (essential singularity)
    P(Rg) ~ exp( -(pi^2/4) Rg^2/<Rg^2> )     Rg -> inf

In reduced units the distribution is PARAMETER-FREE: it does not depend on N or
on the segment length b.  Everything below is therefore a fixed number, not a
family.

    <Rg>/sqrt(<Rg^2>)        = 0.9704414
    V = <Rg^2>/<Rg>^2 - 1    = 0.0618396      sqrt(V) = 0.2486757
    <Rg^4>/<Rg^2>^2          = 19/15 exactly
    Var(Rg^2)/<Rg^2>^2       = 4/15 exactly
    skewness of Rg           = 0.7068
    mode/<Rg>                = 0.8967

The Flory-Fisk closed form (flory_fisk_density below) is the version actually
fitted in the IDP literature.  It matches Fixman's WIDTH to ~15% (V = 0.0738 vs
0.0618) but NOT its SHAPE: skewness 0.291 vs 0.707, and mode/<Rg> 0.959 vs
0.897.  If the point of the test is sensitivity to skewness, the two are not
interchangeable.

NUMERICAL PITFALL (this cost a wrong answer once, do not re-derive it):
evaluating  (sinh(z)/z)**-1.5  directly with complex z is WRONG at large |z|.
Along the inversion contour z = sqrt(2u)(1-1j) the phase of sinh(z)/z winds past
pi, and numpy's principal-branch power then jumps by exp(+-3 i pi) = -1.  The
inverted "density" looks plausible but rings, integrates to 1.01 and gives
V = 0.0503 instead of 0.0618.  Use log_sinhz_over_z() below, which carries the
continuous branch analytically (log sinh z = z - log2 + log1p(-exp(-2z))).

Self-test:  python fixman.py
"""

import numpy as np

__all__ = ["MOMENTS", "rg2_laplace", "rg2_density", "rg_density_reduced",
           "sample_rg", "ensemble_rg", "flory_fisk_density", "weighted_V",
           "moments_from_grid"]

# ----------------------------------------------------------------- exact values
MOMENTS = {
    "mean_over_rms":  0.9704414290,   # <Rg>/sqrt(<Rg^2>)
    "V":              0.0618395898,   # <Rg^2>/<Rg>^2 - 1
    "sqrtV":          0.2486756718,
    "m4_over_m2sq":   19.0 / 15.0,    # <Rg^4>/<Rg^2>^2, exact
    "relvar_rg2":     4.0 / 15.0,     # Var(Rg^2)/<Rg^2>^2, exact
    "skew_rg":        0.7068,         # exact density grid (Monte Carlo: 0.7077)
    "mode_over_mean": 0.8967,
    "tail_small":     "exp(-(9/4) <Rg^2>/Rg^2)",
    "tail_large":     "exp(-(pi^2/4) Rg^2/<Rg^2>)",
}

MOMENTS_FLORY_FISK = {                # for comparison, all exact / analytic
    "mean_over_rms":  0.9650297,
    "V":              0.0737872,
    "sqrtV":          0.2716380,
    "m4_over_m2sq":   9.0 / 7.0,
    "relvar_rg2":     2.0 / 7.0,
    "skew_rg":        0.2906,
    "mode_over_mean": 0.9594,         # sqrt(6/7) / <s>
}


def _log_sinhz_over_z(z):
    """log(sinh z / z) on the CONTINUOUS branch, for Re(z) >= 0. See module docstring."""
    out = np.empty_like(np.asarray(z, dtype=complex))
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < 1.0
    w = z[small] ** 2
    out[small] = np.log1p(w / 6 + w**2 / 120 + w**3 / 5040 + w**4 / 362880)
    zb = z[~small]
    out[~small] = zb - np.log(2.0) + np.log1p(-np.exp(-2 * zb)) - np.log(zb)
    return out


def rg2_laplace(lam):
    """E[exp(-lam * Rg^2/<Rg^2>)] -- Fixman's closed form, real lam >= 0."""
    lam = np.asarray(lam, dtype=float)
    z = 2 * np.sqrt(lam)
    out = np.ones_like(lam)
    nz = lam > 0
    out[nz] = (np.sinh(z[nz]) / z[nz]) ** -1.5
    return out


def rg2_density(y_max=14.0, L=40.0, n=1 << 21):
    """Exact density of Y = Rg^2/<Rg^2> by inverting the characteristic function.

    Returns (y, f) on a uniform grid.  L is the periodisation length (must be
    well past y_max); n the FFT size.  Defaults reproduce <Y> = 1 and
    <Y^2> = 19/15 to six decimals.
    """
    j = np.arange(n // 2 + 1)
    u = 2 * np.pi * j / L
    z = np.sqrt(2 * u) * (1 - 1j)
    phi = np.exp(-1.5 * _log_sinhz_over_z(z))          # characteristic function
    f = (n / L) * np.fft.irfft(np.conj(phi), n)
    y = np.arange(n) * L / n
    keep = y <= y_max
    return y[keep], np.clip(f[keep], 0.0, None)


def rg_density_reduced(x, grid=None):
    """Density of x = Rg/<Rg> (unit mean), evaluated at x by interpolation."""
    y, f = grid if grid is not None else rg2_density()
    ms = MOMENTS["mean_over_rms"]
    xg = np.sqrt(y) / ms
    pg = f * 2 * np.sqrt(y) * ms
    return np.interp(np.asarray(x, dtype=float), xg, pg, left=0.0, right=0.0)


def sample_rg(n, rg_mean=1.0, n_modes=400, rng=None):
    """Draw n independent Rg values (exact in law up to the mode truncation).

    Uses Y = (2/pi^2) sum_k chi2_3,k / k^2 ; modes beyond n_modes are replaced by
    their mean (their variance is O(n_modes^-3) and utterly negligible).
    """
    rng = np.random.default_rng() if rng is None else rng
    k = np.arange(1, n_modes + 1)
    lam = 2.0 / (np.pi**2 * k**2)
    y = np.zeros(n)
    for _ in range(3):                                  # 3 Cartesian components
        y += (rng.standard_normal((n, n_modes)) ** 2 * lam).sum(1)
    y += 3.0 * (2.0 / np.pi**2) * (np.pi**2 / 6 - (1.0 / k**2).sum())   # tail mean
    return np.sqrt(y) * rg_mean / MOMENTS["mean_over_rms"]


def ensemble_rg(n, rg_mean=1.0, grid=None):
    """n DETERMINISTIC Rg values at the (i-0.5)/n quantiles of the distribution.

    This is the low-discrepancy analogue of sample_rg: use it to build a finite
    ensemble for a TENOR-SAXS forward simulation without Monte Carlo noise.
    Mean of the returned values -> rg_mean as n grows.
    """
    y, f = grid if grid is not None else rg2_density()
    cdf = np.cumsum(f) * (y[1] - y[0])
    cdf /= cdf[-1]
    q = (np.arange(n) + 0.5) / n
    yq = np.interp(q, cdf, y)
    return np.sqrt(yq) * rg_mean / MOMENTS["mean_over_rms"]


def flory_fisk_density(s):
    """Flory & Fisk (J. Chem. Phys. 44, 2243, 1966) closed-form approximation.

    P(s) ~ s^6 exp(-7 s^2 / 2), with s = Rg/sqrt(<Rg^2>).  Normalised.
    Slightly too broad (V = 0.07379 vs Fixman 0.06184, <Rg^4>/<Rg^2>^2 = 9/7 vs
    19/15) and much too symmetric (skew 0.291 vs 0.707); it also replaces the
    small-Rg essential singularity by a power law.
    """
    s = np.asarray(s, dtype=float)
    from scipy.special import gamma
    norm = 2 * (3.5) ** 3.5 / gamma(3.5)
    return norm * s**6 * np.exp(-3.5 * s**2)


def moments_from_grid(p, grid=None):
    """<(Rg/sqrt(<Rg^2>))^p> by quadrature on the exact density grid."""
    y, f = grid if grid is not None else rg2_density()
    return float(np.trapezoid(f * y ** (p / 2.0), y))


def weighted_V(w, grid=None):
    """Scattering-weighted relative variance at weight power w (paper's Eq. for V).

    V_w = <Rg^(w+2)> <Rg^w> / <Rg^(w+1)>^2 - 1.
    w = 0 is the physically relevant case for a conformational ensemble of one
    IDP (fixed mass, so no R^6-type weighting); then V_w = V = 0.0618.
    """
    g = grid if grid is not None else rg2_density()
    a, b, c = (moments_from_grid(w + 2, g), moments_from_grid(w, g),
               moments_from_grid(w + 1, g))
    return a * b / c**2 - 1.0


if __name__ == "__main__":
    g = rg2_density()
    y, f = g
    print("norm            %.6f  (1)" % np.trapezoid(f, y))
    print("<Y>             %.6f  (1)" % np.trapezoid(f * y, y))
    print("<Y^2>           %.6f  (19/15 = %.6f)" % (np.trapezoid(f * y**2, y), 19 / 15))
    print("<Rg>/rms        %.6f  (%.6f)" % (moments_from_grid(1, g), MOMENTS["mean_over_rms"]))
    print("V               %.6f  (%.6f)" % (1 / moments_from_grid(1, g) ** 2 - 1, MOMENTS["V"]))
    for w in (0, 2, 4, 6):
        print("V_w  w=%d        %.6f" % (w, weighted_V(w, g)))
    s = sample_rg(200000, rng=np.random.default_rng(0))
    print("sampler  <Rg>   %.4f (1.0)   V %.4f (%.4f)"
          % (s.mean(), (s**2).mean() / s.mean() ** 2 - 1, MOMENTS["V"]))
    e = ensemble_rg(2000)
    print("ensemble <Rg>   %.4f (1.0)   V %.4f (%.4f)"
          % (e.mean(), (e**2).mean() / e.mean() ** 2 - 1, MOMENTS["V"]))

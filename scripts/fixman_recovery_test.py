"""[uri2x-29] Recovery test: does TENOR-SAXS recover the Fixman (1962) V?

The Fixman distribution is the Rg distribution over the CONFORMATIONS of a
single ideal (Gaussian/theta) chain -- the random-coil reference ensemble for
an IDP -- not a polydispersity distribution over different particles. It is
parameter-free: true V = 0.0618396 (sqrt(V) = 0.2486757) always, at
weight_power=0 (a single IDP molecule has fixed mass, so no R^6-type
mass-weighting applies -- same reasoning the paper already uses for the
Gaussian-chain "conformations at fixed contour length" case in
Tab. Typical-values-for). Paired with the exact Gaussian-chain form factor,
since Fixman's distribution *is* the exact conformational-fluctuation
distribution of that same chain model.

Unlike every other distribution family in this codebase there is no width
knob to sweep -- this is a single point, so a clean pass/fail rather than a
curve (see Fixman_uri2x-29/README.md in the Overleaf paper folder).

Usage:
    .venv/bin/python3 scripts/fixman_recovery_test.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import benchmark, fixman, formfactors, protocol, psf, simulation  # noqa: E402

SD_DIST = 360.0
WAVELENGTH = 0.1
DET_SIDE = 3.5
DET_PIX = 500
PSF0 = psf.bartlett2d(3, 15)
PXN = np.array([91, 81, 117, 127])  # round-4 noise-benchmark quartet
RG_APPARENT = 3.0  # nm, matches the noise-benchmark's apparent Rg
GC = formfactors.GUINIER_TABLE["gaussian_chain"]
TRUE_V = fixman.MOMENTS["V"]
TRUE_SQRT_V = fixman.MOMENTS["sqrtV"]


def run_one(n_radii: int, noise: float, rng: np.random.Generator | None) -> float:
    result = simulation.scatter2d(
        rg=RG_APPARENT / np.sqrt(1 + TRUE_V),  # true R0 -> apparent Rg = R0*sqrt(1+V)
        noise=noise,
        v_rel=TRUE_V,  # ignored for dist_type="fixman"; kept for clarity/logging
        phi2=GC.phi2,
        det_pix=DET_PIX,
        sd_dist=SD_DIST,
        wavelength=WAVELENGTH,
        det_side=DET_SIDE,
        psf0=PSF0,
        dist_type="fixman",
        n_radii=n_radii,
        weight_power=GC.weight_power,
        rng=rng,
    )
    r = protocol.tenor_protocol(
        result.intensity, result.qx, result.qy, GC.phi2, pxn=PXN,
        wavelength=WAVELENGTH, observables=("Yg100",),
    )
    return r.best_v


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"True Fixman V = {TRUE_V:.6f}  (sqrt(V) = {TRUE_SQRT_V:.6f})")
    print()

    print("--- noiseless, vs ensemble size n_radii ---")
    for n_radii in (11, 25, 100, 1000):
        best_v = run_one(n_radii, noise=0.0, rng=None)
        print(f"  n_radii={n_radii:5d}  recovered V={best_v:.6f}  sqrt(V)={np.sqrt(best_v):.6f}  "
              f"delta(sqrt V)={np.sqrt(best_v) - TRUE_SQRT_V:+.6f}")
    print()

    print("--- noisy, n_radii=25, 10 replicates per photon density ---")
    rng = np.random.default_rng(0)
    rows = []
    dq2 = benchmark._dq(SD_DIST, WAVELENGTH, DET_SIDE, DET_PIX) ** 2
    for n_q2 in (1e6, 1e7, 1e8, 1e9):
        peak_photons = n_q2 * dq2
        vs = []
        for _ in range(10):
            best_v = run_one(25, noise=-peak_photons, rng=rng)
            if np.isfinite(best_v):
                vs.append(best_v)
        vs = np.asarray(vs)
        valid_frac = vs.size / 10.0
        if vs.size:
            bias = np.sqrt(np.mean(vs)) - TRUE_SQRT_V if np.mean(vs) >= 0 else float("nan")
            rms = np.sqrt(np.mean((np.sqrt(np.clip(vs, 0, None)) - TRUE_SQRT_V) ** 2))
        else:
            bias = rms = float("nan")
        print(f"  N_q2={n_q2:.0e}  valid={valid_frac:.0%}  mean sqrt(V)={np.sqrt(np.mean(vs)) if vs.size else float('nan'):.4f}  "
              f"bias={bias:+.4f}  rms={rms:.4f}")
        rows.append((n_q2, valid_frac, bias, rms))

    import csv
    csv_path = out_dir / "fixman_recovery_test.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_q2", "valid_fraction", "sqrtV_bias", "sqrtV_rms", "true_sqrtV"])
        for n_q2, valid_frac, bias, rms in rows:
            w.writerow([n_q2, valid_frac, bias, rms, TRUE_SQRT_V])
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)

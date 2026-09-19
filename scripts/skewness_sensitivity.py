"""[uri2x-27] Skewness sensitivity test: does the theory's V^{3/2}-order skewness
term actually bias the Y observables at realistic V, and how much?

Directly motivated by the [uri2x-25] finding that Appendix B's cumulant
expansion has a skewness-driven term at O(u^3 V^{3/2}) (legitimately small
for small V, per the paper's own line-1024 argument) sitting alongside the
phi'''-driven term at O(u^3) (NOT V-suppressed, and the one found to be
missing from Table 1). This script tests the V^{3/2} one empirically: hold V
fixed, vary the ensemble's standardized skewness of X=Rg^2,
gamma_1(X) = kappa_3(X)/Var(X)**1.5, independently, and see where the
Y observables start to depend on skewness (not just V) -- i.e. where the
"only weakly dependent on distribution shape" claim starts to break down.

Ensembles are built with tenor_saxs.distributions.target_skewed_ensemble
(Cornish-Fisher quantiles at N=25 equal-probability points, calibrated to
hit the target skewness exactly, floored at <r>/10 per the [uri2x-27] spec --
see that function's docstring for why calibration is needed and where the
floor makes a target genuinely unreachable, which happens here for
skew=-1 at V >~ 0.09).

Noiseless. Five form factors (matching the other sensitivity-analysis
scripts): spherical shell, solid sphere, thin rod, Gaussian chain, thin disk.

Usage:
    .venv/bin/python3 scripts/skewness_sensitivity.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tenor_saxs import formfactors, protocol, psf, simulation  # noqa: E402
from tenor_saxs.distributions import SkewnessTargetUnreachable, target_skewed_ensemble  # noqa: E402

SD_DIST = 360.0
WAVELENGTH = 0.1
DET_SIDE = 3.5
DET_PIX = 500
PSF0 = psf.bartlett2d(3, 15)
PXN = np.array([87, 85, 125, 123])  # tenor_protocol's general-purpose default quartet
N_RADII = 25  # "N=25 should be enough" per the uri2x-27 spec
R0 = 5.0  # nm, baseline apparent Rg (matches Tab. beam_simul_param baseline)

SHAPES = [
    ("spherical_shell", formfactors.GUINIER_TABLE["spherical_shell"].phi2, formfactors.GUINIER_TABLE["spherical_shell"].weight_power),
    ("solid_sphere", formfactors.GUINIER_TABLE["solid_sphere"].phi2, formfactors.GUINIER_TABLE["solid_sphere"].weight_power),
    ("thin_rod", formfactors.GUINIER_TABLE["thin_rod"].phi2, formfactors.GUINIER_TABLE["thin_rod"].weight_power),
    ("gaussian_chain", formfactors.GUINIER_TABLE["gaussian_chain"].phi2, formfactors.GUINIER_TABLE["gaussian_chain"].weight_power),
    ("thin_disk", 0.000666, formfactors.weight_power_for_name("thin_disk")),
]

V_GRID = np.linspace(0.01, 0.30, 12)  # matches the paper's 12-point noise-benchmark V grid
SKEW_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
OBSERVABLES = ("Yg100", "Yg210", "Ym210")
COLORS = {-1.0: "#1f77b4", -0.5: "#7fb3d5", 0.0: "#666666", 0.5: "#e59866", 1.0: "#e74c3c"}


def run_one(phi2: float, weight_power: int, v_rel: float, skew_x_input: float) -> dict:
    result = simulation.scatter2d(
        rg=R0, noise=0, v_rel=v_rel, phi2=phi2, det_pix=DET_PIX, sd_dist=SD_DIST,
        wavelength=WAVELENGTH, det_side=DET_SIDE, psf0=PSF0,
        dist_type="skewed", n_radii=N_RADII, weight_power=weight_power, skew_x=skew_x_input,
    )
    r = protocol.tenor_protocol(
        result.intensity, result.qx, result.qy, phi2, pxn=PXN,
        wavelength=WAVELENGTH, observables=OBSERVABLES,
    )
    return r.observed


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for shape_name, phi2, weight_power in SHAPES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for skew in SKEW_GRID:
            vs, obs = {name: [] for name in OBSERVABLES}, None
            sqrtV_used = []
            for v_rel in V_GRID:
                try:
                    calib = target_skewed_ensemble(N_RADII, v_rel, skew)
                except SkewnessTargetUnreachable:
                    print(f"  {shape_name}: skew={skew:+.1f} unreachable at V={v_rel:.3f} (floor-limited) -- skipped")
                    continue
                observed = run_one(phi2, weight_power, v_rel, calib["skew_x_input"])
                sqrtV_used.append(np.sqrt(v_rel))
                for name in OBSERVABLES:
                    vs[name].append(observed[name])
                rows.append({
                    "shape": shape_name, "skew_target": skew, "skew_realized": calib["realized_skew"],
                    "V": v_rel, "sqrtV": np.sqrt(v_rel), "floored": calib["floored"],
                    **{name: observed[name] for name in OBSERVABLES},
                })
            for ax, name in zip(axes, OBSERVABLES):
                ax.plot(sqrtV_used, vs[name], "-o", ms=4, color=COLORS[skew], label=f"skew={skew:+.1f}")
        for ax, name in zip(axes, OBSERVABLES):
            ax.set_xlabel(r"$\sqrt{V}$")
            ax.set_ylabel(name)
            ax.set_title(name)
        axes[0].legend(fontsize=8, loc="best")
        fig.suptitle(f"{shape_name}: Y observables vs sqrt(V), by ensemble skewness gamma_1(X) (N={N_RADII}, noiseless)")
        fig.tight_layout()
        out_path = out_dir / f"skewness_sensitivity_{shape_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    csv_path = out_dir / "skewness_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["shape", "skew_target", "skew_realized", "V", "sqrtV", "floored", *OBSERVABLES]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "tenor-saxs-v2-data" / "figures")
    args = parser.parse_args()
    main(args.out_dir)

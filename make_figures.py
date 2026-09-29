#!/usr/bin/env python3
"""Rebuild the five paper figures from retained numerical results."""

from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "modal_residues.jl",
    "closure_depth.jl",
    "brusselator_lags.jl",
    "spectral_diagnostic.jl",
    "network_benchmark.jl",
)
PAPER_FIGURES = (
    "dimerization_modal_residues.pdf",
    "krylov_depth_population.png",
    "brusselator_lag_regimes.png",
    "brusselator_spectral_diagnostic.png",
    "closure_network_benchmark.png",
)

for script in SCRIPTS:
    subprocess.run(
        ["julia", "--startup-file=no", "--project=.", ROOT / "plot" / script],
        cwd=ROOT,
        check=True,
    )

destination = ROOT / "paper" / "figures"
destination.mkdir(parents=True, exist_ok=True)
for name in PAPER_FIGURES:
    shutil.copy2(ROOT / "figures" / name, destination / name)

print(f"Wrote {len(PAPER_FIGURES)} figures to {destination}")

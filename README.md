# Koopman Operator Inference for the Chemical Master Equation

This directory is the self-contained submission and reproducibility package for
the 23-page SIADS manuscript.

## Contents

- `paper/main.pdf`: submission-ready manuscript.
- `paper/main.tex`: manuscript source, bibliography, SIAM class, and the five
  final figures.
- `plot/`: fast figure regeneration from retained numerical results.
- `experiments/`: exact finite-state calculations, the cached Brusselator lag
  and spectral analyses, and the four-network closure benchmark.
- `src/`: only the matrix-logarithm, semigroup-defect, stoichiometric extraction,
  and nonnegative least-squares routines used by these experiments.
- `results/`: the CSV inputs behind every figure plus the retained Brusselator
  Gram matrices and all twenty held-out benchmark seeds.
- `Project.toml` and `Manifest.toml`: pinned Julia environment.

## Fast reproduction

The package was verified with Julia 1.13. From this directory:

```text
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python3 make_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`make_figures.py` reads only retained CSV files, writes the five figures to
`figures/`, and copies them into `paper/figures/`.

## Figure-to-code map

| Manuscript figure | Fast plot | Full numerical calculation | Retained input |
|---|---|---|---|
| Dimerization modal residues | `plot/modal_residues.jl` | `experiments/siads_dimerization_modal_residues.jl` | `results/siads_log_diagnostics/dimerization_modal_residues.csv` |
| Generator-image depth | `plot/closure_depth.jl` | `experiments/siads_krylov_depth_population.jl` | `results/siads_log_diagnostics/krylov_depth_*.csv` |
| Brusselator interval sweep | `plot/brusselator_lags.jl` | `experiments/siads_brusselator_lag_regimes.jl` | `results/siads_lag_regimes/summary.csv` and `grams_1000_10000.jls` |
| Projected spectrum | `plot/spectral_diagnostic.jl` | `experiments/siads_brusselator_spectral_diagnostics.jl` | `results/siads_lag_regimes/spectrum.csv`, `spectral_diagnostics.csv`, and the Gram cache |
| Four-network closure benchmark | `plot/network_benchmark.jl` | `experiments/koopman_closure_benchmark.jl` | `results/koopman_closure_benchmark/heldout20_*` and `selected` |

## Numerical regeneration

The two finite dimerization calculations are deterministic:

```text
julia --project=. experiments/siads_dimerization_modal_residues.jl
julia --project=. experiments/siads_krylov_depth_population.jl
```

The Brusselator commands reuse the retained per-trajectory Gram matrices, so
they do not rerun the expensive stochastic simulation:

```text
julia --project=. experiments/siads_brusselator_lag_regimes.jl
julia --project=. experiments/siads_brusselator_spectral_diagnostics.jl
```

The network benchmark can be recomputed from the stochastic simulator. These
runs are substantially more expensive. All use the frozen seeds
`10101,10201,...,12001`.

```text
CLOSURE_BENCHMARK_TAG=heldout20_lv \
CLOSURE_BENCHMARK_MODEL=Lotka \
CLOSURE_BENCHMARK_SEEDS=10101,10201,10301,10401,10501,10601,10701,10801,10901,11001,11101,11201,11301,11401,11501,11601,11701,11801,11901,12001 \
julia --project=. experiments/koopman_closure_benchmark.jl

CLOSURE_BENCHMARK_TAG=heldout20_bru \
CLOSURE_BENCHMARK_MODEL=Brusselator \
CLOSURE_BENCHMARK_SEEDS=10101,10201,10301,10401,10501,10601,10701,10801,10901,11001,11101,11201,11301,11401,11501,11601,11701,11801,11901,12001 \
julia --project=. experiments/koopman_closure_benchmark.jl

CLOSURE_BENCHMARK_DESIGN=excited CLOSURE_BENCHMARK_TAG=heldout20_mm \
CLOSURE_BENCHMARK_MODEL=Michaelis CLOSURE_BENCHMARK_INDEPENDENT_PAIRS=1 \
CLOSURE_BENCHMARK_MM_LAG=6 CLOSURE_BENCHMARK_MM_PATHS=8000 \
CLOSURE_BENCHMARK_MAX_ADDED=1 \
CLOSURE_BENCHMARK_SEEDS=10101,10201,10301,10401,10501,10601,10701,10801,10901,11001,11101,11201,11301,11401,11501,11601,11701,11801,11901,12001 \
julia --project=. experiments/koopman_closure_benchmark.jl

CLOSURE_BENCHMARK_DESIGN=excited CLOSURE_BENCHMARK_TAG=heldout20_mapk \
CLOSURE_BENCHMARK_MODEL=MAPK CLOSURE_BENCHMARK_INDEPENDENT_PAIRS=1 \
CLOSURE_BENCHMARK_MAPK_LAG=1.6 CLOSURE_BENCHMARK_MAPK_PATHS=8000 \
CLOSURE_BENCHMARK_MAX_ADDED=3 \
CLOSURE_BENCHMARK_SEEDS=10101,10201,10301,10401,10501,10601,10701,10801,10901,11001,11101,11201,11301,11401,11501,11601,11701,11801,11901,12001 \
julia --project=. experiments/koopman_closure_benchmark.jl

julia --project=. experiments/assemble_selected_closure_benchmark.jl
```

The retained outputs allow reviewers to inspect and replot the reported results
without repeating those simulations.

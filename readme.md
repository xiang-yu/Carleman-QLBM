# Carleman-QLBM

This repository contains Julia scripts for testing Carleman-linearized lattice Boltzmann models (CLBM) against reference lattice Boltzmann evolutions.

## Quick Start

From the repository root:

```bash
julia --project=.
```

Inside Julia, the three most common entry points are:

```julia
include("src/CLBM/clbm_run.jl")
```

Runs the standard CLBM case from the current configuration.

```julia
include("src/CLBM/plot_multigrid_domain_average.jl")
```

Plots the multi-grid domain-averaged `f_1`, `f_2`, `f_3` comparison between `CLBM` and the centered finite-difference `LBM`, plus absolute and relative errors.

```bash
julia --project=. src/CLBM/unit_tests_minimal.jl
```

Runs the minimal regression tests for the corrected multi-grid implementation.

## Setup

Clone the repository and define the project paths:

```bash
git clone git@github.com:xiang-yu/Carleman-QLBM.git "$HOME/Carleman-QLBM"
```

Add the following to your shell startup file, for example `~/.bashrc` or `~/.zshrc`:

```bash
export QCFD_HOME=$HOME/Carleman-QLBM/
export QCFD_SRC=$QCFD_HOME/src/
```

Then activate the Julia environment from the repository root:

```bash
cd "$HOME/Carleman-QLBM"
julia --project=.
```

Inside Julia:

```julia
using Pkg
Pkg.instantiate()
```

## Main CLBM run

To run the standard CLBM script:

```julia
include("src/CLBM/clbm_run.jl")
```

This uses the configuration in [src/CLBM/clbm_config.jl](src/CLBM/clbm_config.jl).

## Important note on the multi-grid comparison

There are two different notions of “LBM streaming” in this repository:

- The classical lattice update in [src/LBM/streaming.jl](src/LBM/streaming.jl) uses exact lattice shifts.
- The multi-grid CLBM validation added here uses the semi-discrete n-point LBE with centered finite-difference streaming, implemented through [src/CLBM/streaming_Carleman.jl](src/CLBM/streaming_Carleman.jl).

The multi-grid CLBM tests and plots below compare CLBM against the centered finite-difference n-point LBE, because that is the discretization used by the current multi-grid Carleman operator.

## Multi-grid CLBM vs LBM tests

The main regression file is [src/CLBM/unit_tests_minimal.jl](src/CLBM/unit_tests_minimal.jl).

Run it from the repository root:

```bash
julia --project=. src/CLBM/unit_tests_minimal.jl
```

This file contains the following relevant checks.

### 1. `ngrid = 2` periodic uniform-state regression

This test verifies the corrected multi-grid collision and streaming implementation for the simplest periodic case.

It checks that:

- the n-point collision operator reduces to the single-point collision law on a uniform field,
- the multi-grid CLBM evolution matches the corresponding reference evolution,
- the populations remain non-negative.

The expected success message is:

```text
✅ ngrid=2 periodic CLBM matches LBM
```

### 2. `ngrid = 3` nonuniform streaming + collision regression

This is the main multi-grid validation for active streaming.

It uses a nonuniform initial field over 3 grid points:

```julia
phi_ini = vcat(
	f_ini_test(0.12),
	f_ini_test(0.00),
	f_ini_test(-0.08),
)
```

It verifies that the first CLBM level matches the direct semi-discrete n-point LBE right-hand side:

$$
\partial_t \phi = -S\phi + F_1\phi + F_2\phi^{[2]} + F_3\phi^{[3]}.
$$

It also checks that one explicit Euler update from the direct n-point LBE matches the first-level CLBM update.

The expected success message is:

```text
✅ ngrid=3 nonuniform centered-difference streaming + collision matches direct n-point LBE
```

## Plotting the multi-grid domain-averaged comparison

The plotting script is [src/CLBM/plot_multigrid_domain_average.jl](src/CLBM/plot_multigrid_domain_average.jl).

Run it from Julia with:

```julia
include("src/CLBM/plot_multigrid_domain_average.jl")
```

This script does the following:

- sets `ngrid = 3`,
- uses the same nonuniform initial condition as the regression test,
- evolves the direct centered finite-difference n-point LBE,
- evolves the CLBM with the same centered-difference streaming operator,
- computes domain averages for `f_1`, `f_2`, and `f_3`,
- plots:
	- top row: `⟨f_m⟩` for `LBM` and `CLBM`,
	- middle row: `|f_m^CLBM - f_m^LBM|`,
	- bottom row: `|f_m^CLBM - f_m^LBM| / f_m^LBM`.

The legend labels are intentionally simple:

- `LBM`
- `CLBM`

These plots compare the particle distribution functions `f_1`, `f_2`, and `f_3`, not macroscopic velocity.

### Numerical summaries printed by the plotting script

The script prints the maximum domain-averaged absolute difference and the per-component maxima. A recent run produced:

```text
Max domain-averaged absolute difference = 0.0004599192792154039
Max |Δ⟨f_1⟩| = 0.00022995963960767418
Max |Δ⟨f_2⟩| = 0.0004599192792154039
Max |Δ⟨f_3⟩| = 0.00022995963960761867
```

This is a multi-step nonuniform test, so the difference is no longer exactly machine zero. The one-step/RHS regression in `unit_tests_minimal.jl` remains the stricter operator-level check.

## Which script to run for which result

### Run a standard CLBM case

```julia
include("src/CLBM/clbm_run.jl")
```

Use this for the standard configured CLBM run and plot.

### Verify the corrected multi-grid implementation

```bash
julia --project=. src/CLBM/unit_tests_minimal.jl
```

Use this to check:

- `ngrid = 2` periodic uniform validation,
- `ngrid = 3` nonuniform centered-difference operator validation.

### Display domain-averaged CLBM vs LBM curves and errors

```julia
include("src/CLBM/plot_multigrid_domain_average.jl")
```

Use this to get:

- domain-averaged `f_1`, `f_2`, `f_3` overlap,
- absolute error versus time,
- relative error versus time.

## Relevant files

- [src/CLBM/clbm_run.jl](src/CLBM/clbm_run.jl): main CLBM run script
- [src/CLBM/clbm_config.jl](src/CLBM/clbm_config.jl): shared configuration
- [src/CLBM/unit_tests_minimal.jl](src/CLBM/unit_tests_minimal.jl): minimal regression tests
- [src/CLBM/plot_multigrid_domain_average.jl](src/CLBM/plot_multigrid_domain_average.jl): domain-averaged multi-grid comparison plot
- [src/CLBM/timeMarching.jl](src/CLBM/timeMarching.jl): CLBM and direct n-point time marching helpers
- [src/CLBM/streaming_Carleman.jl](src/CLBM/streaming_Carleman.jl): centered finite-difference streaming operators
- [src/LBM/streaming.jl](src/LBM/streaming.jl): exact lattice-shift streaming implementation

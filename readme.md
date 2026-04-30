# Carleman-QLBM

This repository contains Julia scripts for testing Carleman-linearized lattice Boltzmann models (CLBE) against reference lattice Boltzmann evolutions.

This README was prepared with assistance from GPT-5.4.

## Repository layout

The Julia source is organized into three sibling directories under `src/`:

- [`src/LBM/`](src/LBM/) — the discrete lattice Boltzmann method: exact lattice-shift streaming and collision in integer lattice-time iterations.
- [`src/LBE/`](src/LBE/) — the direct semi-discrete n-point lattice Boltzmann equation: the continuous-in-time ODE that CLBE linearizes. Primary apples-to-apples reference for CLBE.
- [`src/CLBE/`](src/CLBE/) — the Carleman-linearized LBE: the lifted system and the sparse/dense time-marching that evolves it.

## Quick Start

From the repository root:

```bash
julia --project=.
```

Inside Julia, the three most common entry points are:

```julia
include("src/CLBE/clbe_multigrid_run.jl")
main(comparison_ngrid=6, local_use_sparse=true, local_n_time=100, l_plot=true,
     coeff_method=:numerical, initial_condition=:legacy, dt_override=0.05)
```

Runs the primary D1Q3 CLBE driver through the legacy-compatible `main(...)` wrapper. For `ngrid = 1`, it preserves the historical single-point collision test. For `ngrid >= 3`, it runs the multigrid collision+streaming comparison workflow. Internally, the driver now builds explicit configuration objects before preparing the runtime state.

If you want to use the newer configuration-oriented API directly:

```julia
include("src/CLBE/clbe_multigrid_run.jl")

core_cfg, case_cfg = build_d1q3_multigrid_configs(
    comparison_ngrid=6,
    local_use_sparse=true,
    local_n_time=100,
    local_truncation_order=4,
    coeff_method=:numerical,
    initial_condition=:sinusoidal,
    u_ini=0.1,
    dt_override=0.05,
)

run_d1q3_multigrid(case_cfg, core_cfg; l_plot=true)
```

```julia
include("src/CLBE/plot_multigrid_domain_average.jl")
main(local_n_time=100, comparison_ngrid=6, local_truncation_order=3,
     coeff_method=:numerical, initial_condition=:legacy)
```

Plots the multi-grid domain-averaged `f_1`, `f_2`, `f_3` comparison between `CLBE` and the centered finite-difference `LBM`, plus absolute and relative errors. This plotting script now reuses the shared D1Q3 multigrid runner instead of rebuilding its own mutable runtime configuration.

```julia
include("src/CLBE/plot_truncation_order_error_comparison.jl")
main(k_values=[3, 4], comparison_ngrid=6, local_n_time=100,
     coeff_method=:numerical, initial_condition=:legacy)
```

Plots the domain-averaged absolute and relative error histories for multiple truncation orders `k`.

```bash
julia --project=. src/CLBE/unit_tests_minimal.jl
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

## Main CLBE run

To run the standard CLBE script:

```julia
include("src/CLBE/clbe_multigrid_run.jl")
main(comparison_ngrid=6, local_use_sparse=true, local_n_time=100, l_plot=true,
     coeff_method=:numerical, initial_condition=:legacy, dt_override=0.05)
```

This uses the defaults defined in [src/CLBE/clbe_config.jl](src/CLBE/clbe_config.jl), but the D1Q3 driver no longer relies on silently scattering multigrid-specific overrides throughout the file. Instead, it constructs explicit run configurations and then prepares the runtime state from them.

### Configuration and driver structure

The D1Q3 code path is now organized into three layers:

- [src/CLBE/clbe_config.jl](src/CLBE/clbe_config.jl)
  - keeps the legacy global defaults for backward compatibility,
  - defines `CLBECoreConfig` for shared CLBE/LBE parameters,
  - defines `D1Q3MultigridConfig` for the D1Q3 test-case / driver parameters,
  - provides helper constructors such as `default_clbe_core_config()`, `default_d1q3_multigrid_config(...)`, and `build_d1q3_multigrid_configs(...)`.

- [src/CLBE/clbe_multigrid_run.jl](src/CLBE/clbe_multigrid_run.jl)
  - acts as the D1Q3 driver/test-case layer,
  - provides `run_d1q3_multigrid(case_cfg, core_cfg; l_plot=...)`,
  - preserves `main(...)` as a thin compatibility wrapper for quick interactive use,
  - accepts `dt_override=...` if you want to set the D1Q3 time step explicitly.

- Core numerical files such as [src/CLBE/timeMarching.jl](src/CLBE/timeMarching.jl), [src/CLBE/carleman_transferA.jl](src/CLBE/carleman_transferA.jl), and [src/LBE/direct_LBE.jl](src/LBE/direct_LBE.jl)
  - remain the dynamical core used by the driver.

This separation is meant to make the code easier to read: for example, `LX = LY = LZ = 1` in `clbe_config.jl` now clearly means “default legacy values,” not “the D1Q3 multigrid driver is permanently single-site.”

#### Config-based D1Q3 example

```julia
include("src/CLBE/clbe_multigrid_run.jl")

core_cfg, case_cfg = build_d1q3_multigrid_configs(
    comparison_ngrid=6,
    local_use_sparse=true,
    local_n_time=100,
    local_truncation_order=4,
    coeff_method=:numerical,
    initial_condition=:sinusoidal,
    u_ini=0.1,
    dt_override=0.05,
)

run_d1q3_multigrid(case_cfg, core_cfg; l_plot=true)
```

The legacy-compatible wrapper remains available:

```julia
include("src/CLBE/clbe_multigrid_run.jl")
main(comparison_ngrid=6, local_use_sparse=true, local_n_time=100,
     local_truncation_order=4, coeff_method=:numerical,
     initial_condition=:sinusoidal, u_ini=0.1, dt_override=0.05, l_plot=true)
```

If you do not pass `dt_override`, the D1Q3 multigrid workflow uses the current default stability convention

```julia
dt = tau_value / 10
```

rather than the legacy global default `dt = 1.0`.

### D1Q3 initial-condition options

The D1Q3 multigrid workflow now supports two named initial-condition families:

- `initial_condition=:legacy` — the original repository behavior,
- `initial_condition=:sinusoidal` — the sinusoidal state described in `LBE_theory.tex`.

For the sinusoidal option, the spatial velocity profile is

$$
u_i = u_{\mathrm{ini}} \sin\left(\frac{2\pi i}{N_x}\right), \qquad i = 1,\dots,N_x,
$$

equivalently,

$$
\mathbf{g}(r_x^\star,0) = \mathbf{g}^{\mathrm{eq}}\!\left(u^\star = u^\star_{\mathrm{ini}} \sin\frac{2\pi r_x^\star}{N_x}\right).
$$

The original repository / legacy D1Q3 initial condition means:

- for `comparison_ngrid == 3`, the historical profile `[0.12, 0.00, -0.08]`,
- otherwise, a linear ramp from `0.12` to `-0.08` across the grid.

Use `u_ini=...` to set the sinusoidal amplitude. For example, to run the sinusoidal initial condition:

```julia
include("src/CLBE/clbe_multigrid_run.jl")
main(comparison_ngrid=6, local_use_sparse=true, local_n_time=100,
     l_plot=true, coeff_method=:numerical,
     initial_condition=:sinusoidal, u_ini=0.1, dt_override=0.05)
```

To reproduce the original repository behavior explicitly:

```julia
include("src/CLBE/clbe_multigrid_run.jl")
main(comparison_ngrid=6, local_use_sparse=true, local_n_time=100,
     l_plot=true, coeff_method=:numerical,
     initial_condition=:legacy, dt_override=0.05)
```

### Carleman coefficient-generation mode

The CLBE drivers support two Carleman coefficient-generation modes:

- `:numerical` — the default and recommended option for routine runs
- `:symbolic` — an optional alternative when you explicitly want symbolic coefficient generation

The repository default in [src/CLBE/clbe_config.jl](src/CLBE/clbe_config.jl) is now:

```julia
global coeff_generation_method = :numerical
```

You can still override this at run time by passing `coeff_method` into the driver you call. For example:

```julia
include("src/CLBE/clbe_multigrid_run.jl")
main(comparison_ngrid=6, local_use_sparse=true, local_n_time=100, coeff_method=:symbolic)
```

Important:

- [src/CLBE/clbe_multigrid_run.jl](src/CLBE/clbe_multigrid_run.jl), [src/CLBE/plot_multigrid_domain_average.jl](src/CLBE/plot_multigrid_domain_average.jl), and [src/CLBE/plot_truncation_order_error_comparison.jl](src/CLBE/plot_truncation_order_error_comparison.jl) no longer auto-run on `include(...)`.
- After including them, explicitly call `main(...)` with the parameters you want.
- If you want more explicit control than `main(...)`, use `build_d1q3_multigrid_configs(...)` plus `run_d1q3_multigrid(...)`.
- For D1Q3 runs, you can set the time step either through `main(..., dt_override=...)` or through `build_d1q3_multigrid_configs(..., dt_override=...)`.

The legacy entry point [src/CLBE/clbe_run.jl](src/CLBE/clbe_run.jl) is retained as a thin compatibility wrapper that simply includes [src/CLBE/clbe_multigrid_run.jl](src/CLBE/clbe_multigrid_run.jl).

## Important note on the multi-grid comparison

There are two different notions of "LBM streaming" in this repository:

- The classical lattice update in [src/LBM/streaming.jl](src/LBM/streaming.jl) uses exact lattice shifts.
- The multi-grid CLBE validation added here uses the semi-discrete n-point LBE with centered finite-difference streaming, implemented through [src/CLBE/streaming_Carleman.jl](src/CLBE/streaming_Carleman.jl).

The multi-grid CLBE tests and plots below compare CLBE against the centered finite-difference n-point LBE, because that is the discretization used by the current multi-grid Carleman operator.

## Multi-grid CLBE vs LBM tests

The main regression file is [src/CLBE/unit_tests_minimal.jl](src/CLBE/unit_tests_minimal.jl).

Run it from the repository root:

```bash
julia --project=. src/CLBE/unit_tests_minimal.jl
```

This file contains the following relevant checks.

### 1. `ngrid = 2` periodic uniform-state regression

This test verifies the corrected multi-grid collision and streaming implementation for the simplest periodic case.

It checks that:

- the n-point collision operator reduces to the single-point collision law on a uniform field,
- the multi-grid CLBE evolution matches the corresponding reference evolution,
- the populations remain non-negative.

The expected success message is:

```text
✅ ngrid=2 periodic CLBE matches LBM
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

It verifies that the first CLBE level matches the direct semi-discrete n-point LBE right-hand side:

$$
\partial_t \phi = -S\phi + F_1\phi + F_2\phi^{[2]} + F_3\phi^{[3]}.
$$

It also checks that one explicit Euler update from the direct n-point LBE matches the first-level CLBE update.

The expected success message is:

```text
✅ ngrid=3 nonuniform centered-difference streaming + collision matches direct n-point LBE
```

## Experimental D2Q9 CLBE status

The D2Q9 CLBE path is currently **experimental**. The active driver is [src/CLBE/clbe_tg2d_run.jl](src/CLBE/clbe_tg2d_run.jl).

Important implementation notes for this path:

- `poly_order` is taken from [src/CLBE/clbe_config.jl](src/CLBE/clbe_config.jl),
- `local_truncation_order` is a runtime input argument,
- the driver now enforces `local_truncation_order >= poly_order`,
- the sparse time-marching path in [src/CLBE/timeMarching.jl](src/CLBE/timeMarching.jl) now avoids building the lifted streaming operator through a dense intermediate matrix.

### Current minimal benchmark

The current debug benchmark is the periodic D2Q9 Taylor–Green case

- `nx = ny = 3`,
- `local_truncation_order = 3`,
- `poly_order = 3`,
- `coeff_method = :numerical`,
- `local_n_time = 3`.

A representative run from the repository root is:

```bash
julia --project=. -e 'ENV["QCFD_SRC"]=pwd()*"/src/"; ENV["QCFD_HOME"]=pwd(); include("src/CLBE/clbe_tg2d_run.jl"); main(nx=3, ny=3, amplitude=0.02, rho_value=1.0, local_n_time=3, l_plot=false, boundary_setup=false, coeff_method=:numerical, local_truncation_order=3)'
```

### Current Carleman size and cost

For the `3 × 3`, `k = 3` D2Q9 test, the lifted CLBE state has

- `VT size = (538083, 3)` for the short `nt = 3` run.

The current sparse build also reports:

- dense-equivalent matrix size estimate: `2157.2 GB`,
- assembled sparse Carleman matrix size: `14,004,306` nonzeros.

This means the D2Q9 path is now runnable for the minimal debug case, but it is still computationally heavy even on a `3 × 3` grid.

### Current error metrics

For the completed periodic `3 × 3`, `k = 3`, `nt = 3` run, the driver reported:

```text
Max distribution absolute difference = 0.01126333024919772
Max distribution relative difference = 0.10682459909765085
Max velocity absolute error norm = 0.04242715067873692
Max velocity relative error norm = 155.89166394701334
```

So the current status is:

- the minimal D2Q9 CLBE case now runs to completion,
- the sparse assembly bottleneck has been reduced enough for the short debug case,
- the CLBE solution is **not yet quantitatively matching** the reference D2Q9 LBM.

### What to do next

Before trying larger D2Q9 runs, the next work should focus on **model/operator accuracy**:

1. compare the first CLBE update against the direct semi-discrete D2Q9 n-point LBE right-hand side,
2. isolate whether the remaining discrepancy comes from the collision lift, the streaming lift, or the lifted-state layout,
3. add a D2Q9 operator-level regression test analogous to the existing D1Q3 multigrid RHS/update checks,
4. only after that, extend the D2Q9 benchmark to larger grids or longer times.

Until those checks are completed, treat the D2Q9 CLBE driver as a development/debugging path rather than a validated production workflow.

## Plotting the multi-grid domain-averaged comparison

The plotting script is [src/CLBE/plot_multigrid_domain_average.jl](src/CLBE/plot_multigrid_domain_average.jl).

Run it from Julia with:

```julia
include("src/CLBE/plot_multigrid_domain_average.jl")
main(local_n_time=100, comparison_ngrid=6, local_truncation_order=3,
     coeff_method=:numerical, initial_condition=:legacy, dt_override=0.05)
```

Parameter mapping for this script:

- `comparison_ngrid` sets the D1Q3 spatial grid count,
- `local_truncation_order` sets the Carleman truncation order `k`,
- `local_n_time` sets the number of time steps,
- `coeff_method` selects Carleman coefficient generation, with `:numerical` as the default,
- `initial_condition` selects `:legacy` or `:sinusoidal`,
- `u_ini` sets the sinusoidal amplitude when `initial_condition=:sinusoidal`,
- `dt_override` sets an explicit D1Q3 time step.

For example, the call above runs the `D1Q3`, `ngrid = 6`, `k = 3`, `nt = 100` case with the original/legacy initial condition.

To run the sinusoidal initial condition instead:

```julia
include("src/CLBE/plot_multigrid_domain_average.jl")
main(local_n_time=100, comparison_ngrid=6, local_truncation_order=3,
     coeff_method=:numerical, initial_condition=:sinusoidal, u_ini=0.1, dt_override=0.05)
```

This script does the following:

- sets `ngrid = comparison_ngrid` for the requested D1Q3 run,
- uses the selected D1Q3 initial condition (`:legacy` or `:sinusoidal`),
- evolves the direct centered finite-difference n-point LBE,
- evolves the CLBE with the same centered-difference streaming operator,
- computes domain averages for `f_1`, `f_2`, and `f_3`,
- plots:
	- top row: `⟨f_m⟩` for `LBM` and `CLBE`,
	- middle row: `|f_m^CLBE - f_m^LBM|`,
	- bottom row: `|f_m^CLBE - f_m^LBM| / f_m^LBM`.

If `output_basename` is omitted, the output PDF name is generated automatically and includes the model and key parameters, for example:

- `plot_multigrid_domain_average_D1Q3_ngrid6_k3_nt100.pdf`

### Numerical summaries printed by the plotting script

The script prints the maximum domain-averaged absolute difference and the per-component maxima. A recent run produced:

```text
Max domain-averaged absolute difference = 0.0004599192792154039
Max |Δ⟨f_1⟩| = 0.00022995963960767418
Max |Δ⟨f_2⟩| = 0.0004599192792154039
Max |Δ⟨f_3⟩| = 0.00022995963960761867
```

This is a multi-step nonuniform test, so the difference is no longer exactly machine zero. The one-step/RHS regression in `unit_tests_minimal.jl` remains the stricter operator-level check.

## Comparing errors for different truncation orders

The truncation-order comparison script is [src/CLBE/plot_truncation_order_error_comparison.jl](src/CLBE/plot_truncation_order_error_comparison.jl).

Run the default comparison from Julia with:

```julia
include("src/CLBE/plot_truncation_order_error_comparison.jl")
main(k_values=[3, 4], comparison_ngrid=6, local_n_time=100,
     coeff_method=:numerical, initial_condition=:legacy, dt_override=0.05)
```

By default, this script:

- uses the selected D1Q3 initial condition for each compared truncation order,
- compares truncation orders `k = 3` and `k = 4`,
- reuses the same multigrid CLBE/LBM comparison workflow used by `plot_multigrid_domain_average.jl` for each requested `k`,
- plots the domain-averaged absolute and relative errors for `f_1`, `f_2`, and `f_3`.

The script saves the PDF automatically to the figures directory. By default, if `QCFD_QCLBM_FIG_DIR` is not set, it uses:

- `$HOME/Documents/git-tex/QC/QCFD-QCLBM/figs/plot_truncation_order_error_comparison_D1Q3.pdf`

You can also rerun the comparison with different parameters from Julia without editing the file:

```julia
include("src/CLBE/plot_truncation_order_error_comparison.jl")
main(k_values=[3, 4], comparison_ngrid=3, local_n_time=40,
     coeff_method=:numerical, initial_condition=:sinusoidal, u_ini=0.1, dt_override=0.05)
```

Useful variations include:

- changing `k_values` to compare more truncation orders,
- increasing `local_n_time` to study long-time error growth,
- changing `comparison_ngrid` if you want to test a different n-point setup,
- switching between `initial_condition=:legacy` and `initial_condition=:sinusoidal`,
- changing `u_ini` to vary the sinusoidal amplitude,
- changing `dt_override` to control the D1Q3 time step explicitly,
- switching to `coeff_method=:symbolic` if you want symbolic Carleman coefficient generation.

Interpretation notes:

- for strict implementation validation, rely on the one-step and RHS agreement checks in [src/CLBE/unit_tests_minimal.jl](src/CLBE/unit_tests_minimal.jl).

## 2D Taylor-Green flow workflow

The repository now also contains a 2D D2Q9 Taylor-Green (TG) workflow for:

- a pure numerical LBM baseline,
- a periodic analytical TG benchmark,
- a boundary-initialized TG visualization case,
- an experimental 2D CLBE-vs-LBM comparison driver.

The main files are:

- [src/LBM/tg_d2q9_lbm_run.jl](src/LBM/tg_d2q9_lbm_run.jl): pure numerical D2Q9 TG LBM runner,
- [src/CLBE/clbe_tg2d_run.jl](src/CLBE/clbe_tg2d_run.jl): 2D CLBE-vs-LBM TG driver,
- [src/LBM/streaming.jl](src/LBM/streaming.jl): current numerical D2Q9 streaming and wall treatment.

### Periodic TG benchmark against the analytical solution

The periodic Taylor-Green vortex is the exact theory benchmark consistent with the 2D TG setup described in the paper notes (`r_x, r_y \in [-\pi, \pi)`).

The implemented velocity field is:

$$
u_x = A \sin(r_x)\cos(r_y), \qquad
u_y = -A \cos(r_x)\sin(r_y).
$$

The analytical decay used for the numerical LBM benchmark is:

$$
\exp\left[-\nu\left(\left(\frac{2\pi}{n_x}\right)^2 + \left(\frac{2\pi}{n_y}\right)^2\right)t\right],
\qquad
\nu = c_s^2(\tau - 0.5), \quad c_s^2 = \frac{1}{3}.
$$

Run the periodic benchmark from Julia with:

```julia
include("src/LBM/tg_d2q9_lbm_run.jl")
run_tg_d2q9_lbm(
	nx=16,
	ny=16,
	amplitude=0.02,
	tau_value=0.8,
	n_time=20,
	l_noslipBC=false,
	compare_analytical=true,
	l_plot=true,
)
```

If `output_file` is omitted and `l_plot=true`, the script now saves the figure as a PDF by default to:

- `$HOME/Documents/git-tex/QC/QCFD-QCLBM/figs/tg_d2q9_periodic_vs_analytical_nx16_ny16_nt20.pdf`

The periodic comparison figure includes:

- numerical and analytical speed fields,
- numerical velocity vectors,
- a centerline `u_x` profile,
- absolute and relative error histories over time.

A recent `16×16`, `tau=0.8`, `A=0.02`, `t=19` periodic run gave:

- `L2 error = 0.00248280363869588`
- `relative L2 error = 0.019715478854995132`
- `max abs velocity error = 0.00021907064084869012`

### Boundary-initialized TG visualization case

The boundary-value TG run is not the exact analytical Taylor-Green benchmark. Instead, it is a useful visual baseline using the repository's current D2Q9 wall treatment:

- periodic in `x`,
- top/bottom no-slip bounce-back in `y`.

Run it with:

```julia
include("src/LBM/tg_d2q9_lbm_run.jl")
run_tg_d2q9_boundary_lbm(
	nx=16,
	ny=16,
	amplitude=0.02,
	tau_value=0.8,
	n_time=20,
	l_plot=true,
)
```

By default, this saves a PDF to:

- `$HOME/Documents/git-tex/QC/QCFD-QCLBM/figs/tg_d2q9_boundary_nx16_ny16_nt20.pdf`

### 2D CLBE TG driver

The 2D CLBE driver is:

- [src/CLBE/clbe_tg2d_run.jl](src/CLBE/clbe_tg2d_run.jl)

It is structured so that:

- the numerical D2Q9 LBM run provides the reference history,
- numerical Carleman coefficient generation is the default path, while symbolic derivation remains available as an option,
- periodic and boundary-aware streaming operators can be selected on the CLBE side.

Run a small 2D CLBE TG test from Julia with:

```julia
include("src/CLBE/clbe_tg2d_run.jl")
main(nx=3, ny=3, amplitude=0.02, local_n_time=4, l_plot=false, coeff_method=:numerical)
```

Important note:

- the pure numerical D2Q9 LBM TG benchmark is working and validated against the analytical periodic solution,
- the full 2D CLBE path is still experimental because symbolic D2Q9 Carleman coefficient generation is much heavier than the 1D D1Q3 workflow.

## Which script to run for which result

### Run a standard CLBE case

```julia
include("src/CLBE/clbe_multigrid_run.jl")
main(comparison_ngrid=6, local_use_sparse=true, local_n_time=100, l_plot=true, coeff_method=:numerical)
```

Use this for the standard configured CLBE run and plot.

### Verify the corrected multi-grid implementation

```bash
julia --project=. src/CLBE/unit_tests_minimal.jl
```

Use this to check:

- `ngrid = 2` periodic uniform validation,
- `ngrid = 3` nonuniform centered-difference operator validation.

### Display domain-averaged CLBE vs LBM curves and errors

```julia
include("src/CLBE/plot_multigrid_domain_average.jl")
main(local_n_time=100, comparison_ngrid=6, local_truncation_order=3, coeff_method=:numerical)
```

### Compare multiple truncation orders

```julia
include("src/CLBE/plot_truncation_order_error_comparison.jl")
main(k_values=[3, 4], comparison_ngrid=6, local_n_time=100, coeff_method=:numerical)
```

Use this to compare `k` values at fixed `D1Q3` grid size and time horizon.

Use this to get:

- domain-averaged `f_1`, `f_2`, `f_3` overlap,
- absolute error versus time,
- relative error versus time.

### Compare error histories for different truncation orders

```julia
include("src/CLBE/plot_truncation_order_error_comparison.jl")
main(k_values=[3, 4], comparison_ngrid=6, local_n_time=100, coeff_method=:numerical)
```

Use this to get:

- `k = 3` versus `k = 4` error curves under the shared D1Q3 multigrid time-step convention,
- domain-averaged absolute and relative errors for each `f_m`,
- a quick visual check of how truncation order affects long-time behavior.

## Relevant files

- [src/CLBE/clbe_multigrid_run.jl](src/CLBE/clbe_multigrid_run.jl): primary D1Q3 CLBE multigrid driver and compatibility entry point
- [src/CLBE/clbe_run.jl](src/CLBE/clbe_run.jl): compatibility wrapper for the operational driver
- [src/CLBE/clbe_config.jl](src/CLBE/clbe_config.jl): legacy defaults plus explicit config structs/helpers
- [src/CLBE/unit_tests_minimal.jl](src/CLBE/unit_tests_minimal.jl): minimal regression tests
- [src/CLBE/plot_multigrid_domain_average.jl](src/CLBE/plot_multigrid_domain_average.jl): domain-averaged multi-grid comparison plot
- [src/CLBE/plot_truncation_order_error_comparison.jl](src/CLBE/plot_truncation_order_error_comparison.jl): truncation-order error comparison plot
- [src/CLBE/timeMarching.jl](src/CLBE/timeMarching.jl): CLBE time marching helpers (sparse and dense)
- [src/CLBE/streaming_Carleman.jl](src/CLBE/streaming_Carleman.jl): centered finite-difference streaming operators
- [src/LBE/direct_LBE.jl](src/LBE/direct_LBE.jl): direct semi-discrete n-point LBE reference ODE
- [src/LBM/streaming.jl](src/LBM/streaming.jl): exact lattice-shift streaming implementation

# Running the production D2Q9 CLBE

```bash
cd "$HOME/Carleman-QLBM"
julia --project=. -e '
using Pkg; Pkg.instantiate()
ENV["QCFD_HOME"] = pwd()
ENV["QCFD_SRC"] = pwd() * "/src/"
include("src/CLBE/clbe_tg2d_run.jl")
Main.main(
    nx=3,
    ny=3,
    amplitude=0.02,
    local_n_time=10,
    l_plot=false,
    boundary_setup=false,
    coeff_method=:numerical,
    local_truncation_order=3,
    reference_model=:direct_lbe,
    integrator=:matrix_exponential,
    direct_lbe_integrator=:exponential_euler,
)
'
```

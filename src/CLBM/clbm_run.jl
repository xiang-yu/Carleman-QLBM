println("ℹ️  `src/CLBM/clbm_run.jl` is kept as a compatibility wrapper.")
println("   The primary operational driver is `src/CLBM/clbm_multigrid_run.jl`.")

include("clbm_multigrid_run.jl")

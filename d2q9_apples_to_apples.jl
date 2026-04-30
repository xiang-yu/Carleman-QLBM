using Printf
using LinearAlgebra
using SparseArrays

ENV["QCFD_HOME"] = pwd()
ENV["QCFD_SRC"]  = pwd() * "/src/"

include("src/CLBE/clbe_tg2d_run.jl")

# D2Q9 apples-to-apples benchmark — mirrors the D1Q3 multigrid strategy:
#
#   * CLBM uses the assembled sparse (C - S) at truncation order k, marched by
#     explicit Euler with the repository's default dt = 0.1.
#   * Reference = direct semi-discrete n-point LBE, marched by explicit Euler
#     with the SAME dt, SAME F^(1), F^(2), F^(3), SAME centered-difference S.
#
# Both sides use rho_value = 1.0001 so the `rho_reference == 1.0 ?` shortcut in
# numerical_collision_rhs falls through to rho_local = sum(state), giving a
# well-posed polynomial decomposition (no frozen-ρ constant term).
#
# If the CLBM construction is correct, the only error this benchmark exposes
# is pure Carleman truncation at order k. The error should shrink with k and
# stay bounded over long nt, as it does in the D1Q3 multigrid test.

struct Case
    label::String
    nx::Int
    ny::Int
    k::Int
    amplitude::Float64
    n_time::Int
end

cases = Case[
    Case("nx3_k3_nt100", 3, 3, 3, 0.02, 100),
    Case("nx3_k4_nt100", 3, 3, 4, 0.02, 100),
    Case("nx4_k3_nt100", 4, 4, 3, 0.02, 100),
    Case("nx6_k3_nt100", 6, 6, 3, 0.02, 100),
]

function run_case(case::Case)
    nx = case.nx; ny = case.ny
    # QCFD convention: ngrid = LX * LY * LZ.
    global LX = nx; global LY = ny; global LZ = 1
    global ngrid = LX * LY * LZ
    ngrid_local = ngrid
    global Q = 9; global D = 2; global use_sparse = true
    global force_factor = 0.0
    global rho0 = 1.0001          # same as D1Q3 multigrid
    global lTaylor = true
    global truncation_order = case.k
    # Explicit-Euler stability: override the LBM-unit default dt=1.0 from
    # clbe_config.jl. Both CLBM and direct n-point LBE use the same dt here,
    # so the self-consistency comparison remains exact.
    global dt = 0.1
    # Keep global dt at clbe_config.jl default (0.1) -- same as D1Q3 strategy.

    setup = build_carleman_setup(rho_value=rho0, nspatial=ngrid_local, method=:numerical)
    global w_value = setup.numeric_weights
    global e_value = setup.numeric_velocities
    global F1_ngrid = setup.carleman_F1
    global F2_ngrid = setup.carleman_F2
    global F3_ngrid = setup.carleman_F3

    # Initial state — same as TG driver, but with rho_value = 1.0001.
    phi_ini = tg2d_initial_condition(nx, ny, case.amplitude, rho0,
        w_value, e_value, 1.0, 3.0, 9.0/2.0, -3.0/2.0)

    # Centered-difference periodic D2Q9 streaming (the CLBM and direct n-point
    # LBE must agree on the streaming operator).
    S_lbm, _ = streaming_operator_D2Q9_interleaved_periodic(nx, ny, 1.0, 1.0)

    t0 = time()

    # Direct semi-discrete n-point LBE (explicit Euler, same F, same S, same dt)
    phiT_lbe = timeMarching_direct_LBE_ngrid(phi_ini, dt, case.n_time,
        F1_ngrid, F2_ngrid, F3_ngrid; S_lbm=S_lbm)

    # CLBM assembled sparse time marching (same F, same S, same dt)
    phiT_clbm, _VT = timeMarching_state_CLBM_sparse(
        setup.symbolic_collision, setup.symbolic_state, 1.0, Q, case.k,
        dt, phi_ini, case.n_time;
        S_lbm=S_lbm, nspatial=ngrid_local,
    )

    elapsed = time() - t0
    abs_err = abs.(phiT_clbm .- phiT_lbe)
    rel_err = abs_err ./ max.(abs.(phiT_lbe), eps(Float64))

    avg_lbe = zeros(Q, case.n_time)
    avg_clbm = zeros(Q, case.n_time)
    for nt = 1:case.n_time
        avg_lbe[:, nt]  = vec(mean(reshape(phiT_lbe[:, nt],  Q, ngrid_local), dims=2))
        avg_clbm[:, nt] = vec(mean(reshape(phiT_clbm[:, nt], Q, ngrid_local), dims=2))
    end
    avg_abs_err = abs.(avg_clbm .- avg_lbe)
    avg_rel_err = avg_abs_err ./ max.(abs.(avg_lbe), eps(Float64))

    return (
        case = case,
        elapsed = elapsed,
        ngrid_local = ngrid_local,
        phiT_lbe = phiT_lbe,
        phiT_clbm = phiT_clbm,
        abs_err = abs_err,
        rel_err = rel_err,
        avg_abs_err = avg_abs_err,
        avg_rel_err = avg_rel_err,
    )
end

using Statistics

summaries = String[]

for case in cases
    println("\n" * "="^72)
    @printf("Running %s: nx=ny=%d, k=%d, A=%.3f, n_time=%d\n",
            case.label, case.nx, case.k, case.amplitude, case.n_time)
    println("="^72)
    GC.gc()
    r = run_case(case)

    println()
    @printf "Wall time: %.1f s\n" r.elapsed
    @printf "Per-site distribution error (|φ_CLBM − φ_direct|):\n"
    @printf "  overall max |Δφ|          = %.3e\n" maximum(r.abs_err)
    @printf "  overall max rel |Δφ|      = %.3e\n" maximum(r.rel_err)
    @printf "  final-step max |Δφ|       = %.3e\n" maximum(r.abs_err[:, end])
    @printf "  final-step max rel |Δφ|   = %.3e\n" maximum(r.rel_err[:, end])
    @printf "Domain-averaged error (|<φ_CLBM> − <φ_direct>|):\n"
    @printf "  overall max |Δ<φ>|        = %.3e\n" maximum(r.avg_abs_err)
    @printf "  final-step max |Δ<φ>|     = %.3e\n" maximum(r.avg_abs_err[:, end])
    @printf "  per-component (max abs / final abs):\n"
    for m = 1:9
        @printf "    f_%d: max = %.3e   final = %.3e\n" m maximum(r.avg_abs_err[m, :]) r.avg_abs_err[m, end]
    end

    push!(summaries, @sprintf("%-16s  nx=%2d  k=%d  nt=%3d  |Δφ|_max=%.3e  |Δφ|_final=%.3e  |Δ<φ>|_max=%.3e",
        case.label, case.nx, case.k, case.n_time,
        maximum(r.abs_err), maximum(r.abs_err[:, end]), maximum(r.avg_abs_err)))
end

println("\n\n" * "="^88)
println("D2Q9 CLBM vs direct n-point LBE  (apples-to-apples, same F/S/dt, rho0=1.0001)")
println("="^88)
for s in summaries; println(s); end

# Save summary
open("data/d2q9_apples_to_apples_summary.txt", "w") do io
    println(io, "D2Q9 CLBM vs direct n-point LBE benchmark")
    println(io, "rho0=1.0001, lTaylor=true, coeff_method=:numerical, dt=0.1, periodic TG")
    println(io, "Self-consistency: both sides use the same F^(1/2/3), same S, same dt.")
    println(io, "This mirrors the D1Q3 plot_multigrid_domain_average benchmark.")
    println(io)
    for s in summaries; println(io, s); end
end
println("\nSaved summary to data/d2q9_apples_to_apples_summary.txt")

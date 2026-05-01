using Printf
using LinearAlgebra
using SparseArrays
using Random
using Dates

if !haskey(ENV, "QCFD_HOME")
    ENV["QCFD_HOME"] = pwd()
end
if !haskey(ENV, "QCFD_SRC")
    ENV["QCFD_SRC"] = joinpath(pwd(), "src") * "/"
end

include("src/CLBE/clbe_tg2d_run.jl")

const GRID_SWEEP = [(3, 3), (6, 6)]
const MAX_SPECTRAL_DIM = 2_000_000
const MAX_ASSEMBLY_DIM = 5_000_000

function summary_output_path()
    outdir = joinpath(pwd(), "data")
    mkpath(outdir)
    return joinpath(outdir, "d2q9_cfull_norm_vs_grid_summary.txt")
end

lifted_dimension(Q, truncation_order, nspatial) = sum((Q * nspatial)^k for k in 1:truncation_order)

function power_iteration_spectral_norm(A::SparseMatrixCSC; maxiter::Int=60, tol::Float64=1e-8, seed::Int=1234)
    n = size(A, 2)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    x ./= norm(x)

    sigma_prev = 0.0
    converged = false
    iterations = 0

    for iter = 1:maxiter
        y = A * x
        sigma = norm(y)
        iterations = iter
        if iszero(sigma)
            return 0.0, true, iter
        end

        z = transpose(A) * y
        z_norm = norm(z)
        if iszero(z_norm)
            return sigma, true, iter
        end

        x .= z ./ z_norm
        rel_change = abs(sigma - sigma_prev) / max(sigma, eps(Float64))
        if iter > 1 && rel_change < tol
            converged = true
            sigma_prev = sigma
            break
        end
        sigma_prev = sigma
    end

    return sigma_prev, converged, iterations
end

function build_full_generator(nx, ny; truncation_order_val=3, coeff_method=:numerical, rho_value=1.0001, boundary_setup=false)
    runtime = prepare_d2q9_carleman_runtime(
        nx=nx,
        ny=ny,
        rho_value=rho_value,
        coeff_method=coeff_method,
        local_truncation_order=truncation_order_val,
        boundary_setup=boundary_setup,
    )

    C_full, bt, _ = build_full_clbe_generator_sparse(
        runtime.setup.symbolic_collision,
        runtime.setup.symbolic_state,
        tau_value,
        Q,
        truncation_order_val,
        poly_order,
        force_factor,
        runtime.w_value,
        runtime.e_value;
        S_lbm=runtime.S_lbm,
        nspatial=runtime.ngrid,
    )

    dropzeros!(C_full)
    return C_full, bt, runtime
end

function write_summary(path, results)
    open(path, "w") do io
        println(io, "D2Q9 full CLBE generator norm sweep")
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "Target matrix: C_full = C - S")
        println(io, "Periodic D2Q9 setup, k = 3, numerical coefficients")
        println(io)
        println(io, @sprintf("%6s %6s %12s %14s %16s %12s %9s %10s  %s", "nx", "ny", "dim", "nnz", "spectral_est", "inf_norm", "conv", "iters", "status"))
        for r in results
            spectral_str = isfinite(r.spectral_est) ? @sprintf("%.8e", r.spectral_est) : "NaN"
            inf_str = isfinite(r.inf_norm) ? @sprintf("%.8e", r.inf_norm) : "NaN"
            println(io, @sprintf("%6d %6d %12d %14d %16s %12s %9s %10d  %s",
                r.nx, r.ny, r.dim, r.nnz, spectral_str, inf_str, string(r.converged), r.iterations, r.status))
        end
    end
end

function main()
    summary_path = summary_output_path()
    println("Running D2Q9 full-generator norm sweep for grids = $(GRID_SWEEP)")
    println("Summary text: $summary_path")
    println()

    results = NamedTuple[]

    for (nx, ny) in GRID_SWEEP
        GC.gc()
        println("="^80)
        @printf("Assembling D2Q9 C_full for nx = %d, ny = %d\n", nx, ny)
        nspatial = nx * ny
        predicted_dim = lifted_dimension(9, 3, nspatial)
        @printf("  predicted lifted dimension = %d\n", predicted_dim)

        warnings = String[]
        if predicted_dim > MAX_ASSEMBLY_DIM
            warning_msg = @sprintf(
                "WARNING: predicted dim %d exceeds MAX_ASSEMBLY_DIM %d; continuing because this sweep is configured for large-memory / supercomputer runs.",
                predicted_dim,
                MAX_ASSEMBLY_DIM,
            )
            println("  " * warning_msg)
            push!(warnings, warning_msg)
        end

        status = "ok"
        dim = predicted_dim
        nnz_val = 0
        spectral_est = NaN
        converged = false
        iterations = 0
        inf_norm_val = NaN

        try
            t_build = time()
            C_full, _, runtime = build_full_generator(nx, ny)
            build_elapsed = time() - t_build
            dim = size(C_full, 1)
            nnz_val = nnz(C_full)
            @printf("  lifted dimension = %d\n", dim)
            @printf("  nnz(C_full)      = %d\n", nnz_val)
            @printf("  assembly time    = %.3f s\n", build_elapsed)

            t_inf = time()
            inf_norm_val = opnorm(C_full, Inf)
            inf_elapsed = time() - t_inf
            @printf("  inf norm         = %.8e\n", inf_norm_val)
            @printf("  inf-norm time    = %.3f s\n", inf_elapsed)

            if dim > MAX_SPECTRAL_DIM
                warning_msg = @sprintf(
                    "WARNING: lifted dim %d exceeds MAX_SPECTRAL_DIM %d; continuing spectral estimation because this sweep is configured for large-memory / supercomputer runs.",
                    dim,
                    MAX_SPECTRAL_DIM,
                )
                println("  " * warning_msg)
                push!(warnings, warning_msg)
            end

            t_spectral = time()
            spectral_est, converged, iterations = power_iteration_spectral_norm(C_full)
            spectral_elapsed = time() - t_spectral
            @printf("  spectral est     = %.8e\n", spectral_est)
            @printf("  spectral time    = %.3f s (converged=%s, iterations=%d)\n", spectral_elapsed, string(converged), iterations)

            if !isempty(warnings)
                status = join(warnings, " | ")
            end

            results_entry = (
                nx=nx,
                ny=ny,
                dim=dim,
                nnz=nnz_val,
                spectral_est=spectral_est,
                inf_norm=inf_norm_val,
                converged=converged,
                iterations=iterations,
                status=status,
                coeff_method=runtime.setup.method,
            )
            push!(results, results_entry)
        catch err
            status = sprint(showerror, err)
            println("  ERROR: $status")
            push!(results, (
                nx=nx,
                ny=ny,
                dim=dim,
                nnz=nnz_val,
                spectral_est=spectral_est,
                inf_norm=inf_norm_val,
                converged=converged,
                iterations=iterations,
                status=status,
                coeff_method=:numerical,
            ))
        end

        write_summary(summary_path, results)
    end

    println("\n" * "-"^80)
    for r in results
        println(r)
    end
    println("Summary written to: $summary_path")
end

main()
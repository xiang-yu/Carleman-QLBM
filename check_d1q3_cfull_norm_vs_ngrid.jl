using Printf
using LinearAlgebra
using SparseArrays
using Random
using Dates
using PyPlot

if !haskey(ENV, "QCFD_HOME")
    ENV["QCFD_HOME"] = pwd()
end
if !haskey(ENV, "QCFD_SRC")
    ENV["QCFD_SRC"] = joinpath(pwd(), "src") * "/"
end

include("src/CLBE/clbe_multigrid_run.jl")

const NGRID_SWEEP = [1, 3, 6, 12, 24, 48]
const MAX_SPECTRAL_DIM = 3_500_000

function tex_output_dir()
    if !haskey(ENV, "TEXPATH")
        error("TEXPATH is not set. Please export TEXPATH so the figure can be saved to \$TEXPATH/QC/QCFD-CarlemanLBE/figs/.")
    end
    outdir = joinpath(ENV["TEXPATH"], "QC", "QCFD-CarlemanLBE", "figs")
    mkpath(outdir)
    return outdir
end

function summary_output_path()
    outdir = joinpath(pwd(), "data")
    mkpath(outdir)
    return joinpath(outdir, "d1q3_cfull_norm_vs_ngrid_summary.txt")
end

function lifted_dimension(Q, truncation_order, ngrid)
    return sum((Q * ngrid)^k for k in 1:truncation_order)
end

function power_iteration_spectral_norm(A::SparseMatrixCSC; maxiter::Int=80, tol::Float64=1e-8, seed::Int=1234)
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

function build_full_generator(ngrid_val; truncation_order_val=3, coeff_method=:numerical)
    core_cfg, case_cfg = build_d1q3_multigrid_configs(
        comparison_ngrid=ngrid_val,
        local_use_sparse=true,
        local_n_time=1,
        local_truncation_order=truncation_order_val,
        coeff_method=coeff_method,
        initial_condition=:legacy,
        dt_override=1.0,
    )

    runtime = prepare_d1q3_runtime(core_cfg, case_cfg)
    S_lbm = ngrid_val > 1 ? streaming_operator_D1Q3_interleaved(ngrid_val, 1.0)[1] : nothing

    C_full, bt, _ = build_full_clbe_generator_sparse(
        runtime.omega,
        runtime.f,
        core_cfg.tau_value,
        core_cfg.Q,
        case_cfg.truncation_order,
        core_cfg.poly_order,
        core_cfg.force_factor,
        runtime.w_value,
        runtime.e_value;
        S_lbm=S_lbm,
        nspatial=ngrid_val,
    )

    dropzeros!(C_full)
    return C_full, bt, core_cfg, case_cfg
    end

function write_summary(path, results)
    open(path, "w") do io
        println(io, "D1Q3 full CLBE generator norm sweep")
        println(io, "Generated: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        println(io, "Target matrix: C_full = C - S")
        println(io, "Norm columns: inf_norm is exact; spectral_est is a power-iteration estimate of the spectral/operator 2-norm when computed")
        println(io)
        println(io, @sprintf("%6s %12s %14s %16s %12s %9s %10s", "ngrid", "dim", "nnz", "spectral_est", "inf_norm", "conv", "iters"))
        for r in results
            println(io, @sprintf(
                "%6d %12d %14d %16.8e %12.8e %9s %10d",
                r.ngrid,
                r.dim,
                r.nnz,
                r.spectral_est,
                r.inf_norm,
                string(r.converged),
                r.iterations,
            ))
        end
    end
end

function plot_results(results, output_pdf)
    ngrids = [r.ngrid for r in results]
    inf_vals = [r.inf_norm for r in results]
    spectral_pairs = [(r.ngrid, r.spectral_est) for r in results if isfinite(r.spectral_est)]
    spectral_ng = [p[1] for p in spectral_pairs]
    spectral_vals = [p[2] for p in spectral_pairs]

    close("all")
    figure(figsize=(7.2, 4.8))
    plot(ngrids, inf_vals, "--s", color="tab:red", linewidth=1.8, markersize=5, label=L"\|C_{\mathrm{full}}\|_\infty")
    if !isempty(spectral_pairs)
        plot(spectral_ng, spectral_vals, "-o", color="tab:blue", linewidth=2.0, markersize=6, label=L"\|C_{\mathrm{full}}\|_2\;\mathrm{(power\ estimate)}")
    end
    xscale("log", base=2)
    yscale("log", base=10)
    xlabel("ngrid")
    ylabel("matrix norm")
    title("D1Q3 full CLBE generator norm vs ngrid")
    grid(true, which="both", alpha=0.3)
    legend(loc="best", fontsize=10)
    tight_layout()
    savefig(output_pdf, bbox_inches="tight")
    println("Saved figure to: $output_pdf")
end

function main()
    output_dir = tex_output_dir()
    output_pdf = joinpath(output_dir, "d1q3_cfull_norm_vs_ngrid.pdf")
    summary_path = summary_output_path()

    println("Running D1Q3 full-generator norm sweep for ngrid = $(NGRID_SWEEP)")
    println("Output PDF: $output_pdf")
    println("Summary text: $summary_path")
    println()

    results = NamedTuple[]

    for ngrid_val in NGRID_SWEEP
        GC.gc()
        println("="^80)
        @printf("Assembling C_full for ngrid = %d\n", ngrid_val)
        t_build = time()
        C_full, _, core_cfg, case_cfg = build_full_generator(ngrid_val)
        build_elapsed = time() - t_build

        dim = size(C_full, 1)
        nnz_val = nnz(C_full)
        @printf("  lifted dimension = %d\n", dim)
        @printf("  nnz(C_full)      = %d\n", nnz_val)
        @printf("  assembly time    = %.3f s\n", build_elapsed)

        t_norm = time()
        inf_norm_val = opnorm(C_full, Inf)
        inf_elapsed = time() - t_norm
        @printf("  inf norm         = %.8e\n", inf_norm_val)

        spectral_est = NaN
        converged = false
        iterations = 0
        spectral_elapsed = 0.0
        if dim <= MAX_SPECTRAL_DIM
            t_spectral = time()
            spectral_est, converged, iterations = power_iteration_spectral_norm(C_full)
            spectral_elapsed = time() - t_spectral
            @printf("  spectral est     = %.8e\n", spectral_est)
            @printf("  spectral time    = %.3f s (converged=%s, iterations=%d)\n", spectral_elapsed, string(converged), iterations)
        else
            println("  spectral est     = skipped (matrix too large for iterative 2-norm estimate in this sweep)")
        end
        @printf("  inf-norm time    = %.3f s\n", inf_elapsed)

        push!(results, (
            ngrid=ngrid_val,
            dim=dim,
            nnz=nnz_val,
            spectral_est=spectral_est,
            inf_norm=inf_norm_val,
            converged=converged,
            iterations=iterations,
            assembly_time_s=build_elapsed,
            norm_time_s=inf_elapsed + spectral_elapsed,
            Q=core_cfg.Q,
            truncation_order=case_cfg.truncation_order,
        ))

        write_summary(summary_path, results)
    end

    println("\n" * "-"^80)
    println(@sprintf("%6s %12s %14s %16s %12s %9s %10s", "ngrid", "dim", "nnz", "spectral_est", "inf_norm", "conv", "iters"))
    for r in results
        println(@sprintf(
            "%6d %12d %14d %16.8e %12.8e %9s %10d",
            r.ngrid,
            r.dim,
            r.nnz,
            r.spectral_est,
            r.inf_norm,
            string(r.converged),
            r.iterations,
        ))
    end

    plot_results(results, output_pdf)
    println("Summary written to: $summary_path")
end

main()
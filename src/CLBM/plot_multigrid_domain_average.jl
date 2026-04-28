l_sympy = true
QCFD_SRC = ENV["QCFD_SRC"]
QCFD_HOME = ENV["QCFD_HOME"]

using PyPlot
using HDF5
using LaTeXStrings

include(QCFD_HOME * "/visualization/plot_kit.jl")
include(QCFD_HOME * "/visualization/plot_CLBM_LBM.jl")

if l_sympy
    using SymPy
    using LinearAlgebra
    include(QCFD_SRC * "CLBM/coeffs_poly.jl")
else
    using Symbolics
end

include("clbm_config.jl")
include(QCFD_SRC * "CLBM/collision_sym.jl")
include(QCFD_SRC * "CLBM/carleman_transferA.jl")
include(QCFD_SRC * "CLBM/carleman_transferA_ngrid.jl")
include(QCFD_SRC * "CLBM/LBM_const_subs.jl")
include(QCFD_SRC * "LBM/lbm_cons.jl")
include(QCFD_SRC * "LBM/lbm_const_sym.jl")
include(QCFD_SRC * "LBM/forcing.jl")
include(QCFD_SRC * "LBM/f_initial.jl")
include(QCFD_SRC * "CLBM/streaming_Carleman.jl")
include(QCFD_SRC * "CLBM/timeMarching.jl")

function multigrid_initial_condition(comparison_ngrid)
    if comparison_ngrid == 3
        return vcat(
            # The arguments to f_ini_test are velocity-like perturbation inputs uu,
            # not particle populations. For uu = -0.08 the resulting D1Q3 state
            # remains positive componentwise.
            f_ini_test(0.12),
            f_ini_test(0.00),
            f_ini_test(-0.08),
        )
    end

    velocity_profile = collect(range(0.12, -0.08, length=comparison_ngrid))
    return reduce(vcat, (f_ini_test(velocity_profile[i]) for i = 1:comparison_ngrid))
end

function compute_domain_average_comparison(; local_n_time=max(n_time, 40), comparison_ngrid=3, local_truncation_order=truncation_order, coeff_method=coeff_generation_method)
    global ngrid = comparison_ngrid
    global use_sparse = true
    global truncation_order = local_truncation_order
    global coeff_generation_method = coeff_method

    w, e, w_val, e_val = lbm_const_sym()
    global w_value = w_val
    global e_value = e_val

    f, omega, u, rho = collision(Q, D, w, e, rho0, lTaylor, lorder2)
    global F1_ngrid, F2_ngrid, F3_ngrid = get_coeff_LBM_Fi_ngrid(
        poly_order,
        Q,
        f,
        omega,
        tau_value,
        comparison_ngrid;
        method=coeff_generation_method,
        w_value_input=w_value,
        e_value_input=e_value,
        rho_value_input=rho0,
        lTaylor_input=lTaylor,
        D_input=D,
    )

    phi_ini = multigrid_initial_condition(comparison_ngrid)

    S_lbm, _ = streaming_operator_D1Q3_interleaved(comparison_ngrid, 1)
    phiT_lbe = timeMarching_direct_LBE_ngrid(phi_ini, dt, local_n_time, F1_ngrid, F2_ngrid, F3_ngrid; S_lbm=S_lbm)
    phiT_clbm, VT = timeMarching_state_CLBM_sparse(omega, f, tau_value, Q, local_truncation_order, dt, phi_ini, local_n_time)

    avg_lbe = domain_average_distribution_history(phiT_lbe, Q, comparison_ngrid)
    avg_clbm = domain_average_distribution_history(phiT_clbm, Q, comparison_ngrid)
    avg_abs_err = abs.(avg_clbm .- avg_lbe)
    avg_rel_err = avg_abs_err ./ max.(abs.(avg_lbe), eps(Float64))

    return (
        phi_ini=phi_ini,
        phiT_lbe=phiT_lbe,
        phiT_clbm=phiT_clbm,
        VT=VT,
        avg_lbe=avg_lbe,
        avg_clbm=avg_clbm,
        avg_abs_err=avg_abs_err,
        avg_rel_err=avg_rel_err,
        comparison_ngrid=comparison_ngrid,
        local_n_time=local_n_time,
        local_truncation_order=local_truncation_order,
    )
end

function plot_domain_average_comparison!(comparison_data; figure_size=(12, 10))
    avg_lbe = comparison_data.avg_lbe
    avg_clbm = comparison_data.avg_clbm
    avg_abs_err = comparison_data.avg_abs_err
    avg_rel_err = comparison_data.avg_rel_err
    comparison_ngrid = comparison_data.comparison_ngrid
    local_n_time = comparison_data.local_n_time
    local_truncation_order = comparison_data.local_truncation_order

    time = 1:local_n_time
    figure(figsize=figure_size)
    for m = 1:Q
        subplot(3, Q, m)
        plot(time, avg_lbe[m, :], "ok", label="LBM")
        plot(time, avg_clbm[m, :], "-", color="r", linewidth=1.8, label="CLBM")
        xlabel("Time step")
        ylabel(latexstring("\\langle f_{$m} \\rangle"))
        if m == 1
            legend(loc="best")
        end

        subplot(3, Q, Q + m)
        semilogy(time, avg_abs_err[m, :], "-", color="b", linewidth=1.8)
        xlabel("Time step")
        ylabel(latexstring("|\\langle f_{$m} \\rangle^{\\mathrm{CLBM}} - \\langle f_{$m} \\rangle^{\\mathrm{LBM}}|"))

        subplot(3, Q, 2 * Q + m)
        semilogy(time, avg_rel_err[m, :], "-", color="g", linewidth=1.8)
        xlabel("Time step")
        ylabel(latexstring("|\\langle f_{$m} \\rangle^{\\mathrm{CLBM}} - \\langle f_{$m} \\rangle^{\\mathrm{LBM}}| / \\langle f_{$m} \\rangle^{\\mathrm{LBM}}"))
    end
    suptitle("Domain-averaged CLBM vs LBM, ngrid = $comparison_ngrid, k = $local_truncation_order")
    tight_layout(rect=(0, 0, 1, 0.96))

    return gcf(), avg_abs_err, avg_rel_err
end

"""
    main(; local_n_time, comparison_ngrid, local_truncation_order, output_basename, coeff_method)

Run domain-averaged CLBM vs LBM comparison for D1Q3 multigrid.

Arguments:
    local_n_time: Number of time steps
    comparison_ngrid: Number of grid points
    local_truncation_order: Carleman truncation order (overrides global truncation_order)
    output_basename: Output file basename (optional)
    coeff_method: Coefficient generation method (optional)

Example usage:
    main(local_n_time=100, comparison_ngrid=6, local_truncation_order=4)
"""
function main(; local_n_time=max(n_time, 40), comparison_ngrid=3, local_truncation_order=truncation_order, output_basename=nothing, coeff_method=coeff_generation_method)
    comparison_data = compute_domain_average_comparison(
        local_n_time=local_n_time,
        comparison_ngrid=comparison_ngrid,
        local_truncation_order=local_truncation_order,
        coeff_method=coeff_method,
    )
    _, avg_abs_err, avg_rel_err = plot_domain_average_comparison!(comparison_data)

    if output_basename === nothing
        output_basename = "plot_multigrid_domain_average_D1Q3_ngrid$(comparison_ngrid)_k$(local_truncation_order)_nt$(local_n_time)"
    end

    output_dir = get(ENV, "QCFD_QCLBM_FIG_DIR", joinpath(homedir(), "Documents", "git-tex", "QC", "QCFD-QCLBM", "figs"))
    mkpath(output_dir)
    output_file = joinpath(output_dir, output_basename * ".pdf")
    savefig(output_file)

    println("n_time used for CLBM/LBM comparison = ", local_n_time)
    println("Max domain-averaged absolute difference = ", maximum(avg_abs_err))
    println("Max domain-averaged relative difference = ", maximum(avg_rel_err))
    for m = 1:Q
        println("Max |Δ⟨f_$m⟩| = ", maximum(avg_abs_err[m, :]))
        println("Max |Δ⟨f_$m⟩| / max(|⟨f_$m⟩|, eps) = ", maximum(avg_rel_err[m, :]))
    end
    println("Saved figure to: ", output_file)

    display(gcf())
    show()

    return output_file, avg_abs_err, avg_rel_err
end

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
            f_ini_test(0.12),
            f_ini_test(0.00),
            f_ini_test(-0.08),
        )
    end

    velocity_profile = collect(range(0.12, -0.08, length=comparison_ngrid))
    return reduce(vcat, (f_ini_test(velocity_profile[i]) for i = 1:comparison_ngrid))
end

function main(; k_values=[3, 4], local_dt=1.0, comparison_ngrid=3, local_n_time=n_time)
    global ngrid = comparison_ngrid
    global use_sparse = true

    w, e, w_val, e_val = lbm_const_sym()
    global w_value = w_val
    global e_value = e_val

    f, omega, u, rho = collision(Q, D, w, e, rho0, lTaylor, lorder2)
    global F1_ngrid, F2_ngrid, F3_ngrid = get_coeff_LBM_Fi_ngrid(poly_order, Q, f, omega, tau_value, comparison_ngrid)

    phi_ini = multigrid_initial_condition(comparison_ngrid)

    S_lbm, _ = streaming_operator_D1Q3_interleaved(comparison_ngrid, 1)
    phiT_lbe = timeMarching_direct_LBE_ngrid(phi_ini, local_dt, local_n_time, F1_ngrid, F2_ngrid, F3_ngrid; S_lbm=S_lbm)
    avg_lbe = domain_average_distribution_history(phiT_lbe, Q, comparison_ngrid)

    avg_clbm_by_k = Dict{Int, Matrix{Float64}}()
    avg_abs_err_by_k = Dict{Int, Matrix{Float64}}()
    avg_rel_err_by_k = Dict{Int, Matrix{Float64}}()

    for k in k_values
        phiT_clbm, _ = timeMarching_state_CLBM_sparse(omega, f, tau_value, Q, k, local_dt, phi_ini, local_n_time)
        avg_clbm = domain_average_distribution_history(phiT_clbm, Q, comparison_ngrid)
        avg_abs_err = abs.(avg_clbm .- avg_lbe)
        avg_rel_err = avg_abs_err ./ max.(abs.(avg_lbe), eps(Float64))

        avg_clbm_by_k[k] = avg_clbm
        avg_abs_err_by_k[k] = avg_abs_err
        avg_rel_err_by_k[k] = avg_rel_err
    end

    time = 1:local_n_time
    color_cycle = ["b", "r", "g", "m", "c", "k"]
    linestyle_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "d", "v", ">", "<", "p", "h"]
    colors = Dict(k => color_cycle[mod1(i, length(color_cycle))] for (i, k) in enumerate(k_values))
    linestyles = Dict(k => linestyle_cycle[mod1(i, length(linestyle_cycle))] for (i, k) in enumerate(k_values))
    markers = Dict(k => marker_cycle[mod1(i, length(marker_cycle))] for (i, k) in enumerate(k_values))

    figure(figsize=(12, 7))
    for m = 1:Q
        subplot(2, Q, m)
        for k in k_values
            semilogy(
                time,
                avg_abs_err_by_k[k][m, :],
                linestyle=linestyles[k],
                color=colors[k],
                linewidth=1.8,
                marker=markers[k],
                markersize=4,
                markerfacecolor=colors[k],
                markeredgecolor=colors[k],
                label="k = $k",
            )
        end
        xlabel("Time step")
        ylabel(latexstring("|\\langle f_{$m} \\rangle^{\\mathrm{CLBM}} - \\langle f_{$m} \\rangle^{\\mathrm{LBM}}|"))
        if m == 1
            legend(loc="best")
        end

        subplot(2, Q, Q + m)
        for k in k_values
            semilogy(
                time,
                avg_rel_err_by_k[k][m, :],
                linestyle=linestyles[k],
                color=colors[k],
                linewidth=1.8,
                marker=markers[k],
                markersize=4,
                markerfacecolor=colors[k],
                markeredgecolor=colors[k],
                label="k = $k",
            )
        end
        xlabel("Time step")
        ylabel(latexstring("|\\langle f_{$m} \\rangle^{\\mathrm{CLBM}} - \\langle f_{$m} \\rangle^{\\mathrm{LBM}}| / \\langle f_{$m} \\rangle^{\\mathrm{LBM}}"))
    end

    suptitle("Truncation-order error comparison, ngrid = $comparison_ngrid")
    tight_layout(rect=(0, 0, 1, 0.95))

    output_dir = get(ENV, "QCFD_QCLBM_FIG_DIR", joinpath(homedir(), "Documents", "git-tex", "QC", "QCFD-QCLBM", "figs"))
    mkpath(output_dir)
    output_file = joinpath(output_dir, "plot_truncation_order_error_comparison_D1Q3.pdf")
    savefig(output_file)

    println("dt used for CLBM/LBM comparison = ", local_dt)
    for k in k_values
        println("truncation order k = ", k)
        println("  overall max domain-averaged absolute error = ", maximum(avg_abs_err_by_k[k]))
        println("  overall max domain-averaged relative error = ", maximum(avg_rel_err_by_k[k]))
        for m = 1:Q
            println("  f_$(m): max abs error = ", maximum(avg_abs_err_by_k[k][m, :]))
            println("  f_$(m): max rel error = ", maximum(avg_rel_err_by_k[k][m, :]))
        end
    end
    println("Saved figure to: ", output_file)

    display(gcf())
    show()

    return output_file
end


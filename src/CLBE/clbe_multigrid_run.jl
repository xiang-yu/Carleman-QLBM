l_sympy = true
QCFD_SRC = ENV["QCFD_SRC"]  
QCFD_HOME = ENV["QCFD_HOME"]  

using HDF5
using PyPlot
using LaTeXStrings

include(QCFD_HOME * "/visualization/plot_kit.jl")

if l_sympy
    using SymPy
    using LinearAlgebra
    include(QCFD_SRC * "CLBE/coeffs_poly.jl")
else
    using Symbolics
end

include("clbe_config.jl")

include(QCFD_SRC * "CLBE/collision_sym.jl")
include(QCFD_SRC * "CLBE/carleman_transferA.jl")
include(QCFD_SRC * "CLBE/carleman_transferA_ngrid.jl")
include(QCFD_SRC * "CLBE/LBM_const_subs.jl")
include(QCFD_SRC * "LBM/lbm_cons.jl")
include(QCFD_SRC * "LBM/lbm_const_sym.jl")
include(QCFD_SRC * "CLBE/CLBE_collision_test.jl")
include(QCFD_SRC * "LBM/forcing.jl")
include(QCFD_SRC * "LBE/direct_LBE.jl")

function multigrid_initial_condition(comparison_ngrid; initial_condition=:legacy, u_ini=0.1)
    return d1q3_multigrid_initial_condition(comparison_ngrid; initial_condition=initial_condition, u_ini=u_ini)
end

function run_legacy_collision_driver(w, e, f, omega; local_use_sparse=use_sparse, local_n_time=n_time, local_truncation_order=truncation_order, l_plot=true)
    if ngrid == 1 && local_use_sparse
        println("ℹ️  Using the original dense legacy collision test for ngrid = 1 to preserve single-point behavior")
        local_use_sparse = false
    end

    validate_sparse_setting(local_use_sparse, ngrid)

    if local_use_sparse
        println("Using SPARSE Carleman matrix implementation (use_sparse=$local_use_sparse, ngrid=$ngrid)")
        fT, VT_f, VT = CLBM_collision_test_sparse(Q, omega, f, local_truncation_order, dt, tau_value, e_value, local_n_time, l_plot)
    else
        println("Using DENSE Carleman matrix implementation (use_sparse=$local_use_sparse, ngrid=$ngrid)")
        C, bt, F0 = carleman_C(Q, local_truncation_order, poly_order, f, omega, tau_value, force_factor, w_value, e_value)
        fT, VT_f, VT = CLBM_collision_test(Q, omega, f, C, local_truncation_order, dt, tau_value, e_value, local_n_time, l_plot)
    end

    title("CLBM-D1Q3, τ=" * string(tau_value) * ", u_0 = 0.1")
    display(gcf())
    show()

    return fT, VT_f, VT
end

function run_multigrid_driver(w, e, f, omega; comparison_ngrid=ngrid, local_n_time=max(n_time, 40), local_truncation_order=truncation_order, l_plot=true, initial_condition=:legacy, u_ini=0.1)
    if !use_sparse
        println("ℹ️  Multigrid collision+streaming validation uses the sparse CLBM path; overriding use_sparse=false")
    end

    if comparison_ngrid == 2
        println("⚠️  ngrid = 2 is a degenerate periodic centered-difference case.")
        println("   The left and right neighbors coincide, so the centered streaming derivative collapses to zero.")
        println("   Use ngrid >= 3 for meaningful multigrid collision+streaming validation.")
    end

    # QCFD convention: ngrid = LX * LY * LZ. For the D1Q3 multigrid test
    # the flow is 1D, so LY = LZ = 1 and LX = comparison_ngrid.
    global LX = comparison_ngrid
    global LY = 1
    global LZ = 1
    global ngrid = LX * LY * LZ
    global use_sparse = true
    # Explicit-Euler stability on the lifted Carleman operator: the config
    # default dt = 1.0 (LBM lattice-time unit) is unstable for multi-step
    # runs. Override to tau_value / 10 for this driver. Matches D2Q9.
    global dt = tau_value / 10
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

    phi_ini = multigrid_initial_condition(comparison_ngrid; initial_condition=initial_condition, u_ini=u_ini)

    S_lbm, _ = streaming_operator_D1Q3_interleaved(comparison_ngrid, 1)
    phiT_lbe = timeMarching_direct_LBE_ngrid(phi_ini, dt, local_n_time, F1_ngrid, F2_ngrid, F3_ngrid; S_lbm=S_lbm)
    phiT_clbm, VT = timeMarching_state_CLBM_sparse(omega, f, tau_value, Q, local_truncation_order, dt, phi_ini, local_n_time)

    avg_lbe = domain_average_distribution_history(phiT_lbe, Q, comparison_ngrid)
    avg_clbm = domain_average_distribution_history(phiT_clbm, Q, comparison_ngrid)
    avg_abs_err = abs.(avg_clbm .- avg_lbe)
    avg_rel_err = avg_abs_err ./ max.(abs.(avg_lbe), eps(Float64))

    if l_plot
        close("all")
        time = 1:local_n_time
        figure(figsize=(12, 10))
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
        suptitle("ngrid = $comparison_ngrid, k = $local_truncation_order, IC = $(initial_condition)")
        tight_layout(rect=(0, 0, 1, 0.96))
        display(gcf())
        show()
    end

    println("n_time used for CLBE/LBM multigrid comparison = ", local_n_time)
    println("Initial condition = ", initial_condition)
    println("Sinusoidal amplitude u_ini = ", u_ini)
    println("Max domain-averaged absolute difference = ", maximum(avg_abs_err))
    println("Max domain-averaged relative difference = ", maximum(avg_rel_err))
    for m = 1:Q
        println("Max |Δ⟨f_$m⟩| = ", maximum(avg_abs_err[m, :]))
        println("Max |Δ⟨f_$m⟩| / max(|⟨f_$m⟩|, eps) = ", maximum(avg_rel_err[m, :]))
    end

    return phiT_lbe, phiT_clbm, VT, avg_abs_err, avg_rel_err
end

"""
    main(; comparison_ngrid, local_use_sparse, local_n_time, local_truncation_order, l_plot, coeff_method, initial_condition, u_ini)

Run D1Q3 CLBE/LBM multigrid driver with explicit truncation order.

Arguments:
    comparison_ngrid: Number of grid points
    local_use_sparse: Use sparse Carleman matrix
    local_n_time: Number of time steps
    local_truncation_order: Carleman truncation order (overrides global truncation_order)
    l_plot: Plot results
    coeff_method: Coefficient generation method (optional)
    initial_condition: D1Q3 initial-condition selector (`:legacy` or `:sinusoidal`)
    u_ini: Velocity amplitude used by the sinusoidal initializer

Example usage:
    main(comparison_ngrid=6, local_truncation_order=4, local_n_time=100, initial_condition=:sinusoidal, u_ini=0.1)
"""
function main(; comparison_ngrid=ngrid, local_use_sparse=use_sparse, local_n_time=n_time, local_truncation_order=truncation_order, l_plot=true, coeff_method=coeff_generation_method, initial_condition=:legacy, u_ini=0.1)
    # QCFD convention: ngrid = LX * LY * LZ. For the D1Q3 multigrid test
    # the flow is 1D, so LY = LZ = 1 and LX = comparison_ngrid.
    global LX = comparison_ngrid
    global LY = 1
    global LZ = 1
    global ngrid = LX * LY * LZ
    global use_sparse = local_use_sparse
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

    if comparison_ngrid == 1
        println("Running legacy single-point CLBM collision test (ngrid = 1)")
        return run_legacy_collision_driver(w, e, f, omega; local_use_sparse=local_use_sparse, local_n_time=local_n_time, local_truncation_order=local_truncation_order, l_plot=l_plot)
    end

    println("Running validated multigrid CLBM collision+streaming comparison (ngrid = $comparison_ngrid, coeff_method = $coeff_generation_method, initial_condition = $(initial_condition))")
    return run_multigrid_driver(w, e, f, omega; comparison_ngrid=comparison_ngrid, local_n_time=max(local_n_time, 40), local_truncation_order=local_truncation_order, l_plot=l_plot, initial_condition=initial_condition, u_ini=u_ini)
end

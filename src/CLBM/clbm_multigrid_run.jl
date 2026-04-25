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
include(QCFD_SRC * "CLBM/CLBM_collision_test.jl")
include(QCFD_SRC * "LBM/forcing.jl")

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

function run_legacy_collision_driver(w, e, f, omega; local_use_sparse=use_sparse, local_n_time=n_time, l_plot=true)
    if ngrid == 1 && local_use_sparse
        println("ℹ️  Using the original dense legacy collision test for ngrid = 1 to preserve single-point behavior")
        local_use_sparse = false
    end

    validate_sparse_setting(local_use_sparse, ngrid)

    if local_use_sparse
        println("Using SPARSE Carleman matrix implementation (use_sparse=$local_use_sparse, ngrid=$ngrid)")
        fT, VT_f, VT = CLBM_collision_test_sparse(Q, omega, f, truncation_order, dt, tau_value, e_value, local_n_time, l_plot)
    else
        println("Using DENSE Carleman matrix implementation (use_sparse=$local_use_sparse, ngrid=$ngrid)")
        C, bt, F0 = carleman_C(Q, truncation_order, poly_order, f, omega, tau_value, force_factor, w_value, e_value)
        fT, VT_f, VT = CLBM_collision_test(Q, omega, f, C, truncation_order, dt, tau_value, e_value, local_n_time, l_plot)
    end

    title("CLBM-D1Q3, τ=" * string(tau_value) * ", u_0 = 0.1")
    display(gcf())
    show()

    return fT, VT_f, VT
end

function run_multigrid_driver(w, e, f, omega; comparison_ngrid=ngrid, local_n_time=max(n_time, 40), l_plot=true)
    if !use_sparse
        println("ℹ️  Multigrid collision+streaming validation uses the sparse CLBM path; overriding use_sparse=false")
    end

    if comparison_ngrid == 2
        println("⚠️  ngrid = 2 is a degenerate periodic centered-difference case.")
        println("   The left and right neighbors coincide, so the centered streaming derivative collapses to zero.")
        println("   Use ngrid >= 3 for meaningful multigrid collision+streaming validation.")
    end

    global ngrid = comparison_ngrid
    global use_sparse = true
    global F1_ngrid, F2_ngrid, F3_ngrid = get_coeff_LBM_Fi_ngrid(poly_order, Q, f, omega, tau_value, comparison_ngrid)

    phi_ini = multigrid_initial_condition(comparison_ngrid)

    S_lbm, _ = streaming_operator_D1Q3_interleaved(comparison_ngrid, 1)
    phiT_lbe = timeMarching_direct_LBE_ngrid(phi_ini, dt, local_n_time, F1_ngrid, F2_ngrid, F3_ngrid; S_lbm=S_lbm)
    phiT_clbm, VT = timeMarching_state_CLBM_sparse(omega, f, tau_value, Q, truncation_order, dt, phi_ini, local_n_time)

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
        suptitle("ngrid = $comparison_ngrid, k = $truncation_order")
        tight_layout(rect=(0, 0, 1, 0.96))
        display(gcf())
        show()
    end

    println("n_time used for CLBM/LBM multigrid comparison = ", local_n_time)
    println("Max domain-averaged absolute difference = ", maximum(avg_abs_err))
    println("Max domain-averaged relative difference = ", maximum(avg_rel_err))
    for m = 1:Q
        println("Max |Δ⟨f_$m⟩| = ", maximum(avg_abs_err[m, :]))
        println("Max |Δ⟨f_$m⟩| / max(|⟨f_$m⟩|, eps) = ", maximum(avg_rel_err[m, :]))
    end

    return phiT_lbe, phiT_clbm, VT, avg_abs_err, avg_rel_err
end

function main(; comparison_ngrid=ngrid, local_use_sparse=use_sparse, local_n_time=n_time, l_plot=true)
    global ngrid = comparison_ngrid
    global use_sparse = local_use_sparse

    w, e, w_val, e_val = lbm_const_sym()
    global w_value = w_val
    global e_value = e_val

    f, omega, u, rho = collision(Q, D, w, e, rho0, lTaylor, lorder2)
    global F1_ngrid, F2_ngrid, F3_ngrid = get_coeff_LBM_Fi_ngrid(poly_order, Q, f, omega, tau_value, comparison_ngrid)

    if comparison_ngrid == 1
        println("Running legacy single-point CLBM collision test (ngrid = 1)")
        return run_legacy_collision_driver(w, e, f, omega; local_use_sparse=local_use_sparse, local_n_time=local_n_time, l_plot=l_plot)
    end

    println("Running validated multigrid CLBM collision+streaming comparison (ngrid = $comparison_ngrid)")
    return run_multigrid_driver(w, e, f, omega; comparison_ngrid=comparison_ngrid, local_n_time=max(local_n_time, 40), l_plot=l_plot)
end

main()
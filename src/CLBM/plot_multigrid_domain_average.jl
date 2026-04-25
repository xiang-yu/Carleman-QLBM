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

global ngrid = 3
global use_sparse = true
local_n_time = max(n_time, 40)

w, e, w_val, e_val = lbm_const_sym()
global w_value = w_val
global e_value = e_val

f, omega, u, rho = collision(Q, D, w, e, rho0, lTaylor, lorder2)
global F1_ngrid, F2_ngrid, F3_ngrid = get_coeff_LBM_Fi_ngrid(poly_order, Q, f, omega, tau_value, ngrid)

phi_ini = vcat(
    f_ini_test(0.12),
    f_ini_test(0.00),
    f_ini_test(-0.08),
)

S_lbm, _ = streaming_operator_D1Q3_interleaved(ngrid, 1)
phiT_lbe = timeMarching_direct_LBE_ngrid(phi_ini, dt, local_n_time, F1_ngrid, F2_ngrid, F3_ngrid; S_lbm=S_lbm)
phiT_clbm, VT = timeMarching_state_CLBM_sparse(omega, f, tau_value, Q, truncation_order, dt, phi_ini, local_n_time)

avg_lbe = domain_average_distribution_history(phiT_lbe, Q, ngrid)
avg_clbm = domain_average_distribution_history(phiT_clbm, Q, ngrid)
avg_abs_err = abs.(avg_clbm .- avg_lbe)
avg_rel_err = avg_abs_err ./ abs.(avg_lbe)

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
tight_layout()

output_dir = get(ENV, "QCFD_QCLBM_FIG_DIR", "/Users/xiangyu.li/Documents/git-tex/QC/QCFD-QCLBM/figs")
mkpath(output_dir)
output_file = joinpath(output_dir, "plot_multigrid_domain_average.pdf")
savefig(output_file)

println("Max domain-averaged absolute difference = ", maximum(abs.(avg_clbm .- avg_lbe)))
for m = 1:Q
    println("Max |Δ⟨f_$m⟩| = ", maximum(avg_abs_err[m, :]))
end
println("Saved figure to: ", output_file)

display(gcf())
show()
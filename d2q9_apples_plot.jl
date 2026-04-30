using Printf
using LinearAlgebra
using SparseArrays
using Statistics
using PyPlot
using LaTeXStrings

ENV["QCFD_HOME"] = pwd()
ENV["QCFD_SRC"]  = pwd() * "/src/"

include("src/CLBE/clbe_tg2d_run.jl")

# D2Q9 CLBE vs direct n-point LBE (apples-to-apples):
#   * domain-averaged density and velocity magnitude over time (both)
#   * absolute and relative error of domain-averaged density
#   * absolute and relative error of domain-averaged velocity magnitude
#   * per-site profile absolute and relative error

function per_site_macroscopic(phi, Q, ngrid, e_val)
    n_time = size(phi, 2)
    ρ_field = zeros(ngrid, n_time)
    umag_field = zeros(ngrid, n_time)
    for nt = 1:n_time
        f = reshape(phi[:, nt], Q, ngrid)
        for s = 1:ngrid
            ρs = sum(f[:, s])
            ρ_field[s, nt] = ρs
            ux = sum(e_val[k][1] * f[k, s] for k = 1:Q) / max(ρs, eps())
            uy = sum(e_val[k][2] * f[k, s] for k = 1:Q) / max(ρs, eps())
            umag_field[s, nt] = sqrt(ux^2 + uy^2)
        end
    end
    return ρ_field, umag_field
end

function make_plot(nx, ny, k, output_path; local_n_time=n_time)
    result = run_tg2d_clbe_comparison(
        nx=nx,
        ny=ny,
        amplitude=0.02,
        rho_value=1.0001,
        local_n_time=local_n_time,
        boundary_setup=false,
        coeff_method=:numerical,
        local_truncation_order=k,
        reference_model=:direct_lbe,
    )

    ngrid_local = nx * ny
    phiT_lbe = result.phiT_ref
    phiT_clbm = result.phiT_clbm
    e_value = result.e_value

    ρ_lbe_f,  u_lbe_f  = per_site_macroscopic(phiT_lbe,  Q, ngrid_local, e_value)
    ρ_clbm_f, u_clbm_f = per_site_macroscopic(phiT_clbm, Q, ngrid_local, e_value)

    ρ_lbe  = vec(mean(ρ_lbe_f,  dims=1))
    ρ_clbm = vec(mean(ρ_clbm_f, dims=1))
    u_lbe  = vec([sqrt(mean(u_lbe_f[:,  nt].^2)) for nt = 1:local_n_time])
    u_clbm = vec([sqrt(mean(u_clbm_f[:, nt].^2)) for nt = 1:local_n_time])

    abs_err_ρ = abs.(ρ_clbm .- ρ_lbe)
    rel_err_ρ = abs_err_ρ ./ max.(abs.(ρ_lbe), eps())
    abs_err_u = abs.(u_clbm .- u_lbe)
    rel_err_u = abs_err_u ./ max.(abs.(u_lbe), eps())

    abs_err_profile = abs.(phiT_clbm .- phiT_lbe)
    rel_err_profile = abs_err_profile ./ max.(abs.(phiT_lbe), eps(Float64))
    prof_abs_max = vec(maximum(abs_err_profile, dims=1))
    prof_abs_l2  = vec([sqrt(sum(abs_err_profile[:, i].^2)) for i = 1:local_n_time])
    prof_rel_max = vec(maximum(rel_err_profile, dims=1))

    t = dt .* (0:local_n_time-1)
    safelog(x) = max.(x, 1e-10)

    close("all")
    fig = figure(figsize=(14, 10))

    ax = subplot(4, 2, 1)
    plot(t, ρ_lbe,  "-k", linewidth=2.0, label="direct n-point LBE")
    plot(t, ρ_clbm, "--r", linewidth=1.6, label="CLBE")
    y_mid = mean(ρ_lbe); y_range = max(maximum(abs.(ρ_lbe .- y_mid)), 1e-8) * 5 + 1e-6
    ax.set_ylim(y_mid - y_range, y_mid + y_range)
    xlabel(L"t"); ylabel(L"\langle \rho \rangle")
    title("Domain-averaged density"); legend(loc="best"); grid(true)

    subplot(4, 2, 2)
    semilogy(t, safelog(u_lbe),  "-k", linewidth=2.0, label="direct n-point LBE")
    semilogy(t, safelog(u_clbm), "--r", linewidth=1.6, label="CLBE")
    xlabel(L"t"); ylabel(L"\sqrt{\langle |u|^2 \rangle}")
    title("Domain-averaged velocity magnitude"); legend(loc="best"); grid(true)

    subplot(4, 2, 3)
    semilogy(t, safelog(abs_err_ρ), "-b", linewidth=1.8)
    xlabel(L"t"); ylabel(L"|\langle\rho\rangle_{\mathrm{CLBE}} - \langle\rho\rangle_{\mathrm{LBE}}|")
    title("Absolute error, ⟨ρ⟩"); grid(true)

    subplot(4, 2, 4)
    semilogy(t, safelog(abs_err_u), "-b", linewidth=1.8)
    xlabel(L"t"); ylabel(L"|\langle|u|\rangle_{\mathrm{CLBE}} - \langle|u|\rangle_{\mathrm{LBE}}|")
    title("Absolute error, ⟨|u|⟩"); grid(true)

    subplot(4, 2, 5)
    semilogy(t, safelog(rel_err_ρ), "-m", linewidth=1.8)
    xlabel(L"t"); ylabel(L"|\Delta\langle\rho\rangle|/|\langle\rho\rangle_{\mathrm{LBE}}|")
    title("Relative error, ⟨ρ⟩"); grid(true)

    subplot(4, 2, 6)
    semilogy(t, safelog(rel_err_u), "-m", linewidth=1.8)
    xlabel(L"t"); ylabel(L"|\Delta\langle|u|\rangle|/|\langle|u|\rangle_{\mathrm{LBE}}|")
    title("Relative error, ⟨|u|⟩"); grid(true)

    subplot(4, 2, 7)
    semilogy(t, safelog(prof_abs_max), "-b", linewidth=1.8, label=L"\max_{i,x}|\Delta\phi|")
    semilogy(t, safelog(prof_abs_l2),  "--g", linewidth=1.4, label=L"\|\Delta\phi\|_2")
    xlabel(L"t"); ylabel("profile absolute error")
    title("Per-site profile absolute error"); legend(loc="best"); grid(true)

    subplot(4, 2, 8)
    semilogy(t, safelog(prof_rel_max), "-m", linewidth=1.8)
    xlabel(L"t"); ylabel(L"\max_{i,x}|\Delta\phi/\phi_{\mathrm{LBE}}|")
    title("Per-site profile relative error"); grid(true)

    suptitle("D2Q9 CLBE vs direct n-point LBE (apples-to-apples)\nnx=$(nx), k=$(k), A=0.02, ρ₀=1.0001, nt=$(local_n_time), dt=$(dt)")
    tight_layout(rect=(0, 0, 1, 0.96))
    savefig(output_path, bbox_inches="tight")
    println("Saved plot: $output_path")

    # Print key diagnostics so we can verify the plot didn't silently misreport
    @printf "\nDiagnostic values at the first and last step:\n"
    @printf "  ⟨ρ⟩_LBE[1]   = %.8e   ⟨ρ⟩_CLBE[1]   = %.8e\n" ρ_lbe[1]   ρ_clbm[1]
    @printf "  ⟨ρ⟩_LBE[end] = %.8e   ⟨ρ⟩_CLBE[end] = %.8e\n" ρ_lbe[end] ρ_clbm[end]
    @printf "  ⟨|u|⟩_LBE[1]   = %.8e   ⟨|u|⟩_CLBE[1]   = %.8e\n" u_lbe[1]   u_clbm[1]
    @printf "  ⟨|u|⟩_LBE[end] = %.8e   ⟨|u|⟩_CLBE[end] = %.8e\n" u_lbe[end] u_clbm[end]
    @printf "  max |Δ⟨ρ⟩|   = %.3e   max rel = %.3e\n" maximum(abs_err_ρ) maximum(rel_err_ρ)
    @printf "  max |Δ⟨|u|⟩| = %.3e   max rel = %.3e\n" maximum(abs_err_u) maximum(rel_err_u)
    @printf "  max |Δφ|∞   = %.3e   max rel profile = %.3e\n" maximum(prof_abs_max) maximum(prof_rel_max)
end

# By default, use `n_time` from src/CLBE/clbe_config_2D.jl. To override it for a
# specific plot, call for example `make_plot(3, 3, 3, output_path; local_n_time=100)`.
output_path = "data/d2q9_apples_to_apples_nx3_k3_nt$(n_time).pdf"
make_plot(3, 3, 3, output_path)

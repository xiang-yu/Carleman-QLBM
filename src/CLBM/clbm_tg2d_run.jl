l_sympy = true
QCFD_SRC = ENV["QCFD_SRC"]
QCFD_HOME = ENV["QCFD_HOME"]

using PyPlot
using LaTeXStrings

include(QCFD_HOME * "/visualization/plot_kit.jl")

if l_sympy
    using SymPy
    using LinearAlgebra
    using SparseArrays
    include(QCFD_SRC * "CLBM/coeffs_poly.jl")
else
    using Symbolics
end

include("clbm_config.jl")

include(QCFD_SRC * "CLBM/collision_sym.jl")
include(QCFD_SRC * "CLBM/carleman_transferA.jl")
include(QCFD_SRC * "CLBM/carleman_transferA_ngrid.jl")
include(QCFD_SRC * "CLBM/LBM_const_subs.jl")
include(QCFD_SRC * "LBM/lbm_const_sym.jl")
include(QCFD_SRC * "LBM/cal_feq.jl")
include(QCFD_SRC * "LBM/tg_d2q9_lbm_run.jl")
include(QCFD_SRC * "CLBM/timeMarching.jl")

function streaming_operator_D2Q9_interleaved_periodic(nx, ny, hx, hy)
    e = [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ]

    n_velocities = 9
    n_total = n_velocities * nx * ny

    I_idx = Int[]
    J_idx = Int[]
    V_vals = Float64[]

    global_index(vel_idx, i, j) = ((j - 1) * nx + (i - 1)) * n_velocities + vel_idx

    for j in 1:ny
        for i in 1:nx
            for vel in 1:n_velocities
                row_idx = global_index(vel, i, j)
                ex, ey = e[vel]

                if ex == 0 && ey == 0
                    push!(I_idx, row_idx)
                    push!(J_idx, row_idx)
                    push!(V_vals, 0.0)
                    continue
                end

                if ex != 0
                    left_i = i == 1 ? nx : i - 1
                    right_i = i == nx ? 1 : i + 1

                    push!(I_idx, row_idx)
                    push!(J_idx, global_index(vel, left_i, j))
                    push!(V_vals, -ex / (2 * hx))

                    push!(I_idx, row_idx)
                    push!(J_idx, global_index(vel, right_i, j))
                    push!(V_vals, ex / (2 * hx))
                end

                if ey != 0
                    bottom_j = j == 1 ? ny : j - 1
                    top_j = j == ny ? 1 : j + 1

                    push!(I_idx, row_idx)
                    push!(J_idx, global_index(vel, i, bottom_j))
                    push!(V_vals, -ey / (2 * hy))

                    push!(I_idx, row_idx)
                    push!(J_idx, global_index(vel, i, top_j))
                    push!(V_vals, ey / (2 * hy))
                end
            end
        end
    end

    return sparse(I_idx, J_idx, V_vals, n_total, n_total), e
end

function opposite_velocity_index_D2Q9(vel)
    opposite_map = Dict(
        1 => 1,
        2 => 3,
        3 => 2,
        4 => 5,
        5 => 4,
        6 => 9,
        7 => 8,
        8 => 7,
        9 => 6,
    )
    return opposite_map[vel]
end

function streaming_operator_D2Q9_interleaved_boundary_aware(nx, ny, hx, hy)
    e = [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
    ]

    n_velocities = 9
    n_total = n_velocities * nx * ny

    I_idx = Int[]
    J_idx = Int[]
    V_vals = Float64[]

    global_index(vel_idx, i, j) = ((j - 1) * nx + (i - 1)) * n_velocities + vel_idx

    for j in 1:ny
        for i in 1:nx
            for vel in 1:n_velocities
                row_idx = global_index(vel, i, j)
                ex, ey = e[vel]

                if ex == 0 && ey == 0
                    push!(I_idx, row_idx)
                    push!(J_idx, row_idx)
                    push!(V_vals, 0.0)
                    continue
                end

                if ex != 0
                    left_i = i == 1 ? nx : i - 1
                    right_i = i == nx ? 1 : i + 1

                    push!(I_idx, row_idx)
                    push!(J_idx, global_index(vel, left_i, j))
                    push!(V_vals, -ex / (2 * hx))

                    push!(I_idx, row_idx)
                    push!(J_idx, global_index(vel, right_i, j))
                    push!(V_vals, ex / (2 * hx))
                end

                if ey != 0
                    reflected_vel = opposite_velocity_index_D2Q9(vel)

                    if j > 1
                        push!(I_idx, row_idx)
                        push!(J_idx, global_index(vel, i, j - 1))
                        push!(V_vals, -ey / (2 * hy))
                    else
                        push!(I_idx, row_idx)
                        push!(J_idx, global_index(reflected_vel, i, j))
                        push!(V_vals, -ey / (2 * hy))
                    end

                    if j < ny
                        push!(I_idx, row_idx)
                        push!(J_idx, global_index(vel, i, j + 1))
                        push!(V_vals, ey / (2 * hy))
                    else
                        push!(I_idx, row_idx)
                        push!(J_idx, global_index(reflected_vel, i, j))
                        push!(V_vals, ey / (2 * hy))
                    end
                end
            end
        end
    end

    return sparse(I_idx, J_idx, V_vals, n_total, n_total), e
end

function tg_velocity_field(i, j, nx, ny, amplitude)
    x = 2 * π * (i - 1) / nx
    y = 2 * π * (j - 1) / ny
    ux = amplitude * sin(x) * cos(y)
    uy = -amplitude * cos(x) * sin(y)
    return ux, uy
end

function d2q9_equilibrium(rho, ux, uy, w_value, e_value, a_value, b_value, c_value, d_value)
    vcx = [velocity[1] for velocity in e_value]
    vcy = [velocity[2] for velocity in e_value]
    return cal_feq(rho, ux, uy, w_value, vcx, vcy, a_value, b_value, c_value, d_value)
end

function tg2d_initial_condition(nx, ny, amplitude, rho_value, w_value, e_value, a_value, b_value, c_value, d_value)
    phi_ini = zeros(9 * nx * ny)

    for j in 1:ny
        for i in 1:nx
            ux, uy = tg_velocity_field(i, j, nx, ny, amplitude)
            feq = d2q9_equilibrium(rho_value, ux, uy, w_value, e_value, a_value, b_value, c_value, d_value)
            start_idx = ((j - 1) * nx + (i - 1)) * 9 + 1
            phi_ini[start_idx:start_idx + 8] .= feq
        end
    end

    return phi_ini
end

function macroscopic_fields_from_state(phi, nx, ny, e_value)
    rho = zeros(nx, ny)
    ux = zeros(nx, ny)
    uy = zeros(nx, ny)

    for j in 1:ny
        for i in 1:nx
            start_idx = ((j - 1) * nx + (i - 1)) * 9 + 1
            fi = phi[start_idx:start_idx + 8]
            rho_local = sum(fi)
            rho[i, j] = rho_local
            ux[i, j] = sum(fi[k] * e_value[k][1] for k = 1:9) / rho_local
            uy[i, j] = sum(fi[k] * e_value[k][2] for k = 1:9) / rho_local
        end
    end

    return rho, ux, uy
end

function velocity_error_history(phiT_ref, phiT_clbm, nx, ny, e_value)
    n_time = size(phiT_ref, 2)
    abs_err = zeros(n_time)
    rel_err = zeros(n_time)

    for nt in 1:n_time
        _, ux_ref, uy_ref = macroscopic_fields_from_state(phiT_ref[:, nt], nx, ny, e_value)
        _, ux_clbm, uy_clbm = macroscopic_fields_from_state(phiT_clbm[:, nt], nx, ny, e_value)

        ref_vec = vcat(vec(ux_ref), vec(uy_ref))
        err_vec = vcat(vec(ux_clbm - ux_ref), vec(uy_clbm - uy_ref))
        abs_err[nt] = norm(err_vec)
        rel_err[nt] = abs_err[nt] / max(norm(ref_vec), eps(Float64))
    end

    return abs_err, rel_err
end

function plot_tg2d_comparison(phiT_ref, phiT_clbm, nx, ny, e_value, local_n_time, truncation_order; case_label="2D TG periodic test")
    _, ux_ref, uy_ref = macroscopic_fields_from_state(phiT_ref[:, end], nx, ny, e_value)
    _, ux_clbm, uy_clbm = macroscopic_fields_from_state(phiT_clbm[:, end], nx, ny, e_value)
    ux_err = ux_clbm - ux_ref
    uy_err = uy_clbm - uy_ref
    abs_err, rel_err = velocity_error_history(phiT_ref, phiT_clbm, nx, ny, e_value)

    close("all")
    figure(figsize=(14, 8))

    subplot(2, 3, 1)
    imshow(ux_ref', origin="lower", cmap="RdBu")
    colorbar()
    title(L"u_x^{\mathrm{LBM}}")

    subplot(2, 3, 2)
    imshow(ux_clbm', origin="lower", cmap="RdBu")
    colorbar()
    title(L"u_x^{\mathrm{CLBM}}")

    subplot(2, 3, 3)
    imshow(ux_err', origin="lower", cmap="RdBu")
    colorbar()
    title(L"u_x^{\mathrm{CLBM}} - u_x^{\mathrm{LBM}}")

    subplot(2, 3, 4)
    imshow(uy_ref', origin="lower", cmap="RdBu")
    colorbar()
    title(L"u_y^{\mathrm{LBM}}")

    subplot(2, 3, 5)
    imshow(uy_clbm', origin="lower", cmap="RdBu")
    colorbar()
    title(L"u_y^{\mathrm{CLBM}}")

    subplot(2, 3, 6)
    semilogy(1:local_n_time, abs_err, "-b", linewidth=1.8, label="absolute")
    semilogy(1:local_n_time, rel_err, "--g", linewidth=1.8, label="relative")
    xlabel("Time step")
    ylabel("Velocity error norm")
    legend(loc="best")
    title("Velocity error history")

    suptitle("$case_label, grid = $(nx)×$(ny), k = $truncation_order")
    tight_layout(rect=(0, 0, 1, 0.96))
    display(gcf())
    show()
end

function build_numerical_tg_reference(; nx, ny, amplitude, rho_value, tau_value, local_n_time, boundary_setup)
    case_label = boundary_setup ? "2D TG boundary-initialized test" : "2D TG periodic test"

    if boundary_setup
        println("Using the current D2Q9 LBM boundary-value TG initializer as the shared setup.")
        println("ℹ️  Reference LBM history uses the current top/bottom no-slip scheme from src/LBM/streaming.jl.")
        _, _, _, _, _, _, _, reference_phi_history = run_tg_d2q9_boundary_lbm(
            nx=nx,
            ny=ny,
            amplitude=amplitude,
            rho_value=rho_value,
            tau_value=tau_value,
            n_time=local_n_time,
            return_phi_history=true,
        )
    else
        println("Using the current pure numerical D2Q9 LBM as the periodic reference simulation.")
        _, _, _, _, _, reference_phi_history = run_tg_d2q9_lbm(
            nx=nx,
            ny=ny,
            amplitude=amplitude,
            rho_value=rho_value,
            tau_value=tau_value,
            n_time=local_n_time,
            l_noslipBC=false,
            return_phi_history=true,
        )
    end

    return (
        case_label=case_label,
        reference_initial_state=copy(reference_phi_history[:, 1]),
        reference_phi_history=reference_phi_history,
    )
end

function build_symbolic_carleman_setup(; rho_value, nspatial)
    println("Using symbolic D2Q9 LBM only to derive Carleman operators.")

    symbolic_weights, symbolic_velocities, numeric_weights, numeric_velocities, a_sym, b_sym, c_sym, d_sym, a_val, b_val, c_val, d_val = lbm_const_sym(Q_local=Q, D_local=D)
    symbolic_state, symbolic_collision, symbolic_velocity, symbolic_density = collision(Q, D, symbolic_weights, symbolic_velocities, rho_value, lTaylor, lorder2)
    carleman_F1, carleman_F2, carleman_F3 = get_coeff_LBM_Fi_ngrid(poly_order, Q, symbolic_state, symbolic_collision, tau_value, nspatial)

    return (
        symbolic_weights=symbolic_weights,
        symbolic_velocities=symbolic_velocities,
        numeric_weights=numeric_weights,
        numeric_velocities=numeric_velocities,
        a_sym=a_sym,
        b_sym=b_sym,
        c_sym=c_sym,
        d_sym=d_sym,
        a_val=a_val,
        b_val=b_val,
        c_val=c_val,
        d_val=d_val,
        symbolic_state=symbolic_state,
        symbolic_collision=symbolic_collision,
        symbolic_velocity=symbolic_velocity,
        symbolic_density=symbolic_density,
        carleman_F1=carleman_F1,
        carleman_F2=carleman_F2,
        carleman_F3=carleman_F3,
    )
end

function select_d2q9_streaming_operator(nx, ny, hx, hy; boundary_setup=false)
    if boundary_setup
        println("ℹ️  CLBM uses a boundary-aware centered-difference D2Q9 streaming operator with bounce-back-inspired wall coupling.")
        return streaming_operator_D2Q9_interleaved_boundary_aware(nx, ny, hx, hy)
    end

    return streaming_operator_D2Q9_interleaved_periodic(nx, ny, hx, hy)
end

function main(; nx=3, ny=3, amplitude=0.05, rho_value=1.0, local_n_time=10, l_plot=false, boundary_setup=false)
    if nx < 3 || ny < 3
        error("Use nx >= 3 and ny >= 3 for non-degenerate periodic centered-difference TG streaming.")
    end

    global LX = nx
    global LY = ny
    global ngrid = nx * ny
    global Q = 9
    global D = 2
    global use_sparse = true
    global force_factor = 0.0
    global rho0 = rho_value
    global lTaylor = false
    global poly_order = 2
    global truncation_order = 2

    numerical_reference = build_numerical_tg_reference(
        nx=nx,
        ny=ny,
        amplitude=amplitude,
        rho_value=rho_value,
        tau_value=tau_value,
        local_n_time=local_n_time,
        boundary_setup=boundary_setup,
    )
    symbolic_setup = build_symbolic_carleman_setup(rho_value=rho_value, nspatial=ngrid)

    global w_value = symbolic_setup.numeric_weights
    global e_value = symbolic_setup.numeric_velocities
    global F1_ngrid = symbolic_setup.carleman_F1
    global F2_ngrid = symbolic_setup.carleman_F2
    global F3_ngrid = symbolic_setup.carleman_F3

    case_label = numerical_reference.case_label
    phi_ini = numerical_reference.reference_initial_state
    phiT_lbe = numerical_reference.reference_phi_history
    hx = 1.0
    hy = 1.0
    S_lbm, _ = select_d2q9_streaming_operator(nx, ny, hx, hy; boundary_setup=boundary_setup)

    phiT_clbm, VT = timeMarching_state_CLBM_sparse(symbolic_setup.symbolic_collision, symbolic_setup.symbolic_state, tau_value, Q, truncation_order, dt, phi_ini, local_n_time; S_lbm=S_lbm, nspatial=ngrid)

    dist_abs_err = abs.(phiT_clbm .- phiT_lbe)
    dist_rel_err = dist_abs_err ./ max.(abs.(phiT_lbe), eps(Float64))
    vel_abs_err, vel_rel_err = velocity_error_history(phiT_lbe, phiT_clbm, nx, ny, e_value)

    println("Running $(case_label) CLBM/LBM comparison")
    println("  grid = $(nx)×$(ny)")
    println("  Q = $Q, D = $D")
    println("  poly_order = $poly_order, truncation_order = $truncation_order")
    println("  n_time = $local_n_time")
    println("Max distribution absolute difference = ", maximum(dist_abs_err))
    println("Max distribution relative difference = ", maximum(dist_rel_err))
    println("Max velocity absolute error norm = ", maximum(vel_abs_err))
    println("Max velocity relative error norm = ", maximum(vel_rel_err))

    if l_plot
        plot_tg2d_comparison(phiT_lbe, phiT_clbm, nx, ny, e_value, local_n_time, truncation_order; case_label=case_label)
    end

    return phiT_lbe, phiT_clbm, VT, dist_abs_err, dist_rel_err, vel_abs_err, vel_rel_err
end

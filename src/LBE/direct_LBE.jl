using SparseArrays

# Semi-discrete n-point LBE (Li et al., Eq. eq:eq:lbm-carleman-n)
#
#   ∂_t φ(x)  =  −S φ(x)  +  F^(1) φ(x)
#                         +  F^(2) φ^[2](x)
#                         +  F^(3) φ^[3](x)
#
# Streaming `S` is a centered-difference operator (or whatever finite-difference
# stencil the caller chooses). The collision is the polynomial Taylor-expanded
# form obtained with `lTaylor = true` (so that 1/ρ ≈ 2 − ρ keeps the operator
# polynomial in φ).
#
# This module is intentionally independent of the Carleman lift: it is the
# direct nonlinear ODE that CLBE linearizes. Its role in the repository:
#   * primary self-consistency target for CLBE (the apples-to-apples reference),
#   * bridge between the discrete LBM baseline (src/LBM/, exact lattice shift,
#     integer iteration) and the Carleman-linearized system (src/CLBE/).
#
# Inputs are passed by caller — no globals, no fallback streaming. This keeps
# the module standalone and composable with any streaming stencil supplied
# elsewhere in the codebase.

function direct_lbe_rhs_ngrid(phi, S_lbm, F1_ngrid, F2_ngrid, F3_ngrid)
    rhs = -S_lbm * phi + F1_ngrid * phi
    if F2_ngrid !== nothing
        rhs += F2_ngrid * kron(phi, phi)
    end
    if F3_ngrid !== nothing
        rhs += F3_ngrid * kron(phi, kron(phi, phi))
    end
    return rhs
end

function timeMarching_direct_LBE_ngrid(phi_ini, dt, n_time, F1_ngrid, F2_ngrid, F3_ngrid; S_lbm)
    phiT = zeros(length(phi_ini), n_time)
    phiT[:, 1] = Float64.(phi_ini)

    for nt = 2:n_time
        rhs = direct_lbe_rhs_ngrid(phiT[:, nt - 1], S_lbm, F1_ngrid, F2_ngrid, F3_ngrid)
        phiT[:, nt] = phiT[:, nt - 1] + dt * rhs
    end

    return phiT
end

QCFD_SRC = ENV["QCFD_SRC"]  
QCFD_HOME = ENV["QCFD_HOME"]  

include(QCFD_SRC * "CLBM/carleman_transferA.jl")


function cal_delta_alpha(which_alpha, ngrid)
    delta_alpha = zeros(ngrid)
    delta_alpha[which_alpha] = 1.0
    return delta_alpha
end

function local_selector_matrix(which_alpha, Q, ngrid)
    selector = zeros(Q, Q * ngrid)
    ind_s = (which_alpha - 1) * Q + 1
    ind_e = which_alpha * Q
    selector[:, ind_s:ind_e] = Matrix(1.0I, Q, Q)
    return selector
end

function coeff_LBM_Fi_xalpha(Fj, which_alpha, ngrid, order)
    # F^(k)(x_alpha) acts on phi^[k] with dimension (nQ)^k, while remaining local in x.
    selector_alpha = local_selector_matrix(which_alpha, size(Fj, 1), ngrid)
    selector_alpha_k = Kron_kth(selector_alpha, order)
    F_xalpha = Fj * selector_alpha_k
    return F_xalpha
end

function coeff_LBM_Fi_ngrid(Q, j, f, omega, tau_value, ngrid)
    row_dim = ngrid * Q
    col_dim = (ngrid * Q)^j
    Fj_ngrid = zeros(row_dim, col_dim)
    # F^(k)(x_alpha) is a Q × (nQ)^k matrix selecting only local monomials at x_alpha.
    # F^(k)(x) is the n-point collision operator with size nQ × (nQ)^k.
    Fj = F_carlemanOrder_collision(Q, j, f, omega, tau_value)
    for i = 1 : ngrid
        F_xalpha = coeff_LBM_Fi_xalpha(Fj, i, ngrid, j)
        ind_s = (i - 1) * Q + 1
        ind_e = i * Q
        Fj_ngrid[ind_s:ind_e,:] = F_xalpha
    end
    return Fj, Fj_ngrid
end

function get_coeff_LBM_Fi_ngrid(poly_order, Q, f, omega, tau_value, ngrid) 
    upper_order = 3
    if poly_order != upper_order
        error("poly_order must be equal to upper_order")
    else    
        F1, F1_ngrid = coeff_LBM_Fi_ngrid(Q, upper_order - 2, f, omega, tau_value, ngrid)
        F2, F2_ngrid = coeff_LBM_Fi_ngrid(Q, upper_order - 1, f, omega, tau_value, ngrid)
        F3, F3_ngrid = coeff_LBM_Fi_ngrid(Q, upper_order, f, omega, tau_value, ngrid)
    end
    return F1_ngrid, F2_ngrid, F3_ngrid
end

function transferA_ngrid(i, j, Q, ngrid)
    # dim(A_ij) = (ngrid*Q)^(i-1) * Q^j
    if j == 1
        Fj_ngrid = F1_ngrid
    elseif j == 2
        Fj_ngrid = F2_ngrid
    elseif j == 3
        Fj_ngrid = F3_ngrid
    else
        error("j of F^{j} must be 1, 2, 3, ..., poly_order")
    end
    #
    A_ij = sum_Kron_kth_identity(Fj_ngrid, i, Q * ngrid)
    #
    return A_ij
end

function transferA_S(i, Q, ngrid, S_Fj)
    #
    A_ij = sum_Kron_kth_identity(S_Fj, i, Q * ngrid)
    #
    return A_ij
end


function get_phi(f, ngrid)
    return vcat([collect(f) for _ = 1:ngrid]...)
end

function get_S_Fj(S, ngrid)
    # Streaming is already the full nonlocal n-point linear operator S on phi(x).
    return S
end

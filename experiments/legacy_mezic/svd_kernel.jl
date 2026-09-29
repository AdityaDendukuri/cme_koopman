# Verbatim kernel from b007201:src/koopman.jl; provenance in README.md
function svd_dmd_K(Ψ_X, Ψ_Y, r)
    F = svd(Ψ_X)
    U_r = F.U[:, 1:r]
    Σ_r = Diagonal(F.S[1:r])
    V_r = F.Vt'[:, 1:r]
    Ã = U_r' * Ψ_Y * V_r / Σ_r
    return U_r * Ã * U_r'
end

# Spectral diagnostics for the fitted Koopman matrices in the lag study.
include("siads_brusselator_lag_regimes.jl")
using Serialization
using LinearAlgebra
using Statistics

"""Smallest singular-value distance from K to the principal-log branch cut."""
function branch_margin(K)
    eigenvalues = eigvals(K)
    any(z -> real(z) <= 0 && abs(imag(z)) < 1e-10, eigenvalues) && return 0.0
    left = -max(2.0, 2opnorm(K, 2))
    grid = range(left, 0.0; length=4001)
    values = [minimum(svdvals(K - x * I)) for x in grid]
    j = argmin(values)
    lo = grid[max(1, j - 1)]
    hi = grid[min(length(grid), j + 1)]
    ratio = (sqrt(5) - 1) / 2
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1 = minimum(svdvals(K - x1 * I))
    f2 = minimum(svdvals(K - x2 * I))
    for _ = 1:60
        if f1 > f2
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = minimum(svdvals(K - x2 * I))
        else
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = minimum(svdvals(K - x1 * I))
        end
    end
    min(minimum(values), f1, f2)
end

function projected_matrix(a, b)
    X = Matrix(cholesky(Symmetric(a)).L)
    Y = b / transpose(X)
    K = svd_dmd_K(X, Y, 10)
    norm(K - b / a) / max(1, norm(K)) < 1e-7 || error("SVD/Gram mismatch")
    K
end

function run_spectral_diagnostics()
    cache = joinpath(OUT, "grams_1000_10000.jls")
    isfile(cache) || error("missing $cache; run the lag experiment first")
    a, b, ar, br = deserialize(cache)
    means = (("1000", dropdims(mean(a; dims=3); dims=3), b),
             ("10000", dropdims(mean(ar; dims=3); dims=3), br))

    open(joinpath(OUT, "spectral_diagnostics.csv"), "w") do summary
        open(joinpath(OUT, "spectrum.csv"), "w") do spectrum
            println(summary, "lag,ensemble,branch_margin,min_eigenvalue_real,max_eigenvalue_modulus,admissible")
            println(spectrum, "lag,ensemble,eigenvalue_real,eigenvalue_imag")
            for (label, gram, cross) in means, (j, h) in enumerate(LAGS)
                cross_mean = dropdims(mean(cross[:, :, :, j]; dims=3); dims=3)
                K = projected_matrix(gram, cross_mean)
                eigenvalues = eigvals(K)
                margin = branch_margin(K)
                admissible = !any(z -> real(z) <= 0 && abs(imag(z)) < 1e-10, eigenvalues)
                println(summary, join((h, label, margin, minimum(real.(eigenvalues)),
                                       maximum(abs.(eigenvalues)), admissible), ','))
                for z in eigenvalues
                    println(spectrum, join((h, label, real(z), imag(z)), ','))
                end
                println("ensemble=$label lag=$h margin=$margin admissible=$admissible")
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_spectral_diagnostics()
end

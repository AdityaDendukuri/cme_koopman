module KoopmanCMEPaperTools

using LinearAlgebra

export checked_principal_log
export fit_defect_corrected_edmd, extract_stoichiometric_rates

"""Solve a small nonnegative least-squares problem by an active-set method."""
function nnls_solve(A::AbstractMatrix, b::AbstractVector;
                    max_iter::Int=100, tol::Float64=1e-10)
    _, n = size(A)
    x = zeros(n)
    passive = falses(n)

    for _ in 1:max_iter
        gradient = A' * (A * x - b)
        candidates = findall(.!passive)
        isempty(candidates) && break
        values = gradient[candidates]
        minimum(values) >= -tol && break
        passive[candidates[argmin(values)]] = true

        for _ in 1:max_iter
            indices = findall(passive)
            isempty(indices) && break
            proposed = A[:, indices] \ b
            if all(proposed .>= -tol)
                fill!(x, 0)
                x[indices] .= max.(proposed, 0)
                break
            end
            step = minimum(x[i] / (x[i] - proposed[j])
                           for (j, i) in enumerate(indices) if proposed[j] < 0)
            trial = copy(x)
            trial[indices] .= proposed
            x .= x .+ step .* (trial .- x)
            for i in indices
                if x[i] <= tol
                    x[i] = 0
                    passive[i] = false
                end
            end
        end
    end
    x
end

include("defect_corrected_edmd.jl")

end

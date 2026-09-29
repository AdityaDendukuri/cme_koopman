"""
Two-lag EDMD generator recovery with a semigroup-defect correction.

All matrices use the EDMD row convention

    psi(X(t + h)) ≈ K_h * psi(X(t)).

The three Gram matrices must use exactly the same current-state samples.  This
ensures that `K_half` and `K_lag` are compressions under the same empirical
projection, which is essential for interpreting `K_lag - K_half^2` as a
semigroup defect.
"""

struct DefectCorrectedEDMDFit
    gram::Matrix{Float64}
    cross_half::Matrix{Float64}
    cross_lag::Matrix{Float64}
    K_half::Matrix{Float64}
    K_lag::Matrix{Float64}
    log_half::Matrix{Float64}
    log_lag::Matrix{Float64}
    raw_generator::Matrix{Float64}
    corrected_generator::Matrix{Float64}
    richardson_generator::Matrix{Float64}
    semigroup_defect::Matrix{Float64}
    lag::Float64
    gram_condition::Float64
    branch_distance_half::Float64
    branch_distance_lag::Float64
    relative_imaginary_half::Float64
    relative_imaginary_lag::Float64
end

function _square_real_matrix(A, name)
    matrix = Matrix{Float64}(A)
    size(matrix, 1) == size(matrix, 2) ||
        throw(ArgumentError("$name must be square"))
    all(isfinite, matrix) || throw(ArgumentError("$name contains nonfinite entries"))
    matrix
end

function _distance_to_principal_log_cut(z)
    real(z) <= 0 ? abs(imag(z)) : abs(z)
end

"""
    checked_principal_log(K; branch_rtol=1e-10, real_rtol=1e-8)

Compute the real principal matrix logarithm after checking distance to the
closed nonpositive real axis.  The returned diagnostics distinguish proximity
to the branch cut from loss of reality in the computed logarithm.
"""
function checked_principal_log(K; branch_rtol=1e-10, real_rtol=1e-8)
    matrix = _square_real_matrix(K, "K")
    eigenvalues = ComplexF64.(eigvals(matrix))
    scale = max(opnorm(matrix), 1.0)
    branch_distance = minimum(_distance_to_principal_log_cut, eigenvalues)
    branch_distance > branch_rtol * scale ||
        throw(DomainError(branch_distance,
            "matrix spectrum meets or is too close to the principal-logarithm cut"))

    complex_log = log(complex.(matrix))
    relative_imaginary = norm(imag.(complex_log)) /
                         max(norm(real.(complex_log)), eps(Float64))
    relative_imaginary <= real_rtol ||
        throw(DomainError(relative_imaginary,
            "principal logarithm is not numerically real"))

    (value=real.(complex_log), eigenvalues=eigenvalues,
     branch_distance=branch_distance,
     relative_imaginary=relative_imaginary)
end

"""
    matrix_log_frechet(K, E; real_rtol=1e-8)

Evaluate Higham's Frechet derivative `D log(K)[E]` through the standard block
matrix identity.  `K` must lie in the real principal-logarithm domain.
"""
function matrix_log_frechet(K, E; real_rtol=1e-8)
    matrix = _square_real_matrix(K, "K")
    direction = _square_real_matrix(E, "E")
    size(direction) == size(matrix) ||
        throw(ArgumentError("K and E must have the same size"))
    checked_principal_log(matrix; real_rtol)

    d = size(matrix, 1)
    block = zeros(ComplexF64, 2 * d, 2 * d)
    block[1:d, 1:d] .= matrix
    block[1:d, (d + 1):(2 * d)] .= direction
    block[(d + 1):(2 * d), (d + 1):(2 * d)] .= matrix
    derivative = log(block)[1:d, (d + 1):(2 * d)]
    relative_imaginary = norm(imag.(derivative)) /
                         max(norm(real.(derivative)), eps(Float64))
    relative_imaginary <= real_rtol ||
        throw(DomainError(relative_imaginary,
            "logarithm Frechet derivative is not numerically real"))
    real.(derivative)
end

"""
    protected_innovation_svd(G0, protected; relative_tolerance=1e-2,
                             max_rank=size(G0, 1)-protected)

Construct a rank-revealing observable transform without altering the first
`protected` coordinates.  If `z = [p; e]`, first residualize the enrichment,

    r = e - G_ep * G_pp^(-1) * p,

then retain only well-populated eigenvectors of the conditional covariance
`G_ee - G_ep G_pp^(-1) G_pe`.  The returned rectangular matrix `T` defines
the reduced observables `w = T*z`; its first `protected` rows are exactly
`[I 0]`.  Consequently, target generator rows lift back unambiguously as
`B_w[1:protected, :] * T`.

Unlike full whitening, truncating weak conditional innovations changes the
observable subspace and can reduce finite-sample amplification.  Protected
species and propensity coordinates are never discarded.
"""
function protected_innovation_svd(G0, protected;
                                  relative_tolerance=1e-2,
                                  max_rank=size(G0, 1)-protected)
    gram = _square_real_matrix(G0, "G0")
    n = size(gram, 1)
    p = Int(protected)
    1 <= p <= n || throw(ArgumentError("protected must lie between 1 and size(G0,1)"))
    tolerance = Float64(relative_tolerance)
    tolerance >= 0 || throw(ArgumentError("relative_tolerance must be nonnegative"))
    maximum_rank = Int(max_rank)
    0 <= maximum_rank <= n - p ||
        throw(ArgumentError("max_rank must lie between 0 and the enrichment dimension"))

    if p == n
        return (transform=Matrix{Float64}(I, n, n), singular_values=Float64[],
                retained_rank=0, protected_dimension=p)
    end

    protected_gram = Symmetric(gram[1:p, 1:p])
    cholesky(protected_gram) # fail explicitly if protected coordinates are dependent
    enrichment = (p + 1):n
    regression = gram[enrichment, 1:p] / Matrix(protected_gram)
    innovation = Symmetric(gram[enrichment, enrichment] -
                           regression * gram[1:p, enrichment])
    decomposition = eigen(innovation)
    order = sortperm(decomposition.values; rev=true)
    values = decomposition.values[order]
    scale = max(first(values), eps(Float64))
    minimum(values) >= -100 * eps(Float64) * scale ||
        throw(ArgumentError("conditional enrichment covariance is indefinite"))
    values = max.(values, 0.0)
    retained = min(count(>(tolerance * scale), values), maximum_rank)

    residualize = [Matrix{Float64}(I, p, p) zeros(p, n-p);
                   -regression Matrix{Float64}(I, n-p, n-p)]
    if retained == 0
        transform = residualize[1:p, :]
    else
        modes = decomposition.vectors[:, order[1:retained]]'
        selector = [Matrix{Float64}(I, p, p) zeros(p, n-p);
                    zeros(retained, p) modes]
        transform = selector * residualize
    end
    (transform=transform, singular_values=values, retained_rank=retained,
     protected_dimension=p)
end

"""
    fit_defect_corrected_edmd(G0, Ghalf, Glag, h; ...)

Fit EDMD matrices at lags `h/2` and `h` from a common current-state Gram
matrix.  Besides the ordinary logarithmic generator, return

    B_dc = log(K_h)/h + 4/(3h) * (K_h - K_{h/2}^2),

whose selected species rows have `O(h^3)` population bias when their first
generator images lie in the retained dictionary.  A two-lag Richardson
generator is returned as an independent check.
"""
function fit_defect_corrected_edmd(G0, Ghalf, Glag, h;
                                   gram_rtol=nothing,
                                   branch_rtol=1e-10,
                                   real_rtol=1e-8)
    lag = Float64(h)
    lag > 0 || throw(ArgumentError("h must be positive"))
    gram = _square_real_matrix(G0, "G0")
    cross_half = _square_real_matrix(Ghalf, "Ghalf")
    cross_lag = _square_real_matrix(Glag, "Glag")
    size(cross_half) == size(gram) == size(cross_lag) ||
        throw(ArgumentError("G0, Ghalf, and Glag must have the same size"))

    singular_values = svdvals(gram)
    tolerance = gram_rtol === nothing ?
        size(gram, 1) * eps(Float64) * first(singular_values) :
        Float64(gram_rtol) * first(singular_values)
    last(singular_values) > tolerance ||
        throw(ArgumentError("current-state Gram matrix is rank deficient"))

    K_half = cross_half / gram
    K_lag = cross_lag / gram
    half_log = checked_principal_log(K_half; branch_rtol, real_rtol)
    lag_log = checked_principal_log(K_lag; branch_rtol, real_rtol)

    raw_generator = lag_log.value / lag
    half_generator = half_log.value / (lag / 2)
    semigroup_defect = K_lag - K_half * K_half
    corrected_generator = raw_generator + (4 / (3 * lag)) * semigroup_defect
    richardson_generator = (4 * half_generator - raw_generator) / 3

    DefectCorrectedEDMDFit(
        gram, cross_half, cross_lag, K_half, K_lag,
        half_log.value, lag_log.value, raw_generator, corrected_generator,
        richardson_generator, semigroup_defect, lag, cond(gram),
        half_log.branch_distance, lag_log.branch_distance,
        half_log.relative_imaginary, lag_log.relative_imaginary,
    )
end

"""
    defect_corrected_edmd_pushforward(fit, dG0, dGhalf, dGlag)

Differentiate the raw, defect-corrected, and Richardson generators through
both Gram solves and the matrix logarithm.  This is the linear map needed for
trajectory-cluster covariance estimates and bootstrap diagnostics.
"""
function defect_corrected_edmd_pushforward(fit::DefectCorrectedEDMDFit,
                                           dG0, dGhalf, dGlag)
    gram_direction = _square_real_matrix(dG0, "dG0")
    half_direction = _square_real_matrix(dGhalf, "dGhalf")
    lag_direction = _square_real_matrix(dGlag, "dGlag")
    size(gram_direction) == size(fit.gram) == size(half_direction) ==
        size(lag_direction) || throw(ArgumentError("Gram perturbations have the wrong size"))

    dK_half = (half_direction - fit.K_half * gram_direction) / fit.gram
    dK_lag = (lag_direction - fit.K_lag * gram_direction) / fit.gram
    dB_lag = matrix_log_frechet(fit.K_lag, dK_lag) / fit.lag
    dB_half = matrix_log_frechet(fit.K_half, dK_half) / (fit.lag / 2)
    d_defect = dK_lag - dK_half * fit.K_half - fit.K_half * dK_half
    d_corrected = dB_lag + (4 / (3 * fit.lag)) * d_defect
    d_richardson = (4 * dB_half - dB_lag) / 3
    (raw=dB_lag, corrected=d_corrected, richardson=d_richardson,
     defect=d_defect, K_half=dK_half, K_lag=dK_lag)
end

"""
    extract_stoichiometric_rates(B, species_rows, propensity_columns, stoichiometry;
                                 nonnegative=true)

Extract reaction rates from selected rows of an EDMD-row generator.  Reactions
may share a propensity column; their separation is then decided by the rank of
the corresponding stoichiometric block.  NNLS enforces physical nonnegativity.
"""
function extract_stoichiometric_rates(B, species_rows, propensity_columns,
                                      stoichiometry; nonnegative=true)
    generator = _square_real_matrix(B, "B")
    rows = Int.(collect(species_rows))
    columns = Int.(collect(propensity_columns))
    Z = Matrix{Float64}(stoichiometry)
    M = length(columns)
    size(Z) == (length(rows), M) ||
        throw(ArgumentError("stoichiometry must have one row per species and one column per reaction"))
    all(r -> 1 <= r <= size(generator, 1), rows) ||
        throw(ArgumentError("species row is outside the generator"))
    all(c -> 1 <= c <= size(generator, 2), columns) ||
        throw(ArgumentError("propensity column is outside the generator"))

    distinct_columns = sort(unique(columns))
    design = zeros(length(rows) * length(distinct_columns), M)
    response = zeros(size(design, 1))
    index = 0
    for column in distinct_columns, s in eachindex(rows)
        index += 1
        response[index] = generator[rows[s], column]
        for k in 1:M
            columns[k] == column && (design[index, k] = Z[s, k])
        end
    end
    rank(design) == M ||
        throw(ArgumentError("the selected species/propensity block does not identify all rates"))
    unconstrained = design \ response
    rates = nonnegative ? nnls_solve(design, response) : unconstrained
    (rates=rates, unconstrained=unconstrained, design=design,
     response=response, fitted=design * rates,
     residual=response - design * rates)
end

"""
    assemble_conservative_generator(templates, rates; atol=1e-10)

Combine unit-rate, column-convention reaction-generator templates with
nonnegative rates.  Each template must already be closed on the supplied state
set: nonnegative off-diagonals and zero column sums.  The result is therefore a
conservative generator with no sink state, and its negative is a singular
column-diagonally-dominant M-matrix.
"""
function assemble_conservative_generator(templates, rates; atol=1e-10)
    matrices = [_square_real_matrix(template, "generator template")
                for template in templates]
    coefficients = Float64.(collect(rates))
    length(matrices) == length(coefficients) ||
        throw(ArgumentError("there must be one rate per generator template"))
    isempty(matrices) && throw(ArgumentError("at least one template is required"))
    all(size(matrix) == size(first(matrices)) for matrix in matrices) ||
        throw(ArgumentError("generator templates must have the same size"))
    minimum(coefficients) >= -atol ||
        throw(ArgumentError("conservative reconstruction requires nonnegative rates"))
    coefficients = max.(coefficients, 0.0)

    for matrix in matrices
        maximum(abs.(vec(sum(matrix; dims=1)))) <= atol ||
            throw(ArgumentError("a generator template has nonzero column sum"))
        for j in axes(matrix, 2), i in axes(matrix, 1)
            i == j && continue
            matrix[i, j] >= -atol ||
                throw(ArgumentError("a generator template has a negative off-diagonal entry"))
        end
    end

    generator = zeros(size(first(matrices)))
    for (coefficient, matrix) in zip(coefficients, matrices)
        generator .+= coefficient .* matrix
    end
    generator
end

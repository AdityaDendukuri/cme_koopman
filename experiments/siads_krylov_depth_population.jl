# Exact population experiment for generator-image dictionary enrichment.
#
# The reversible dimerization has a finite state space, so this script can
# separate projection bias from sampling variance without truncation or Monte
# Carlo error.  Starting from the rate-extraction dictionary
#     [1, Y, binomial(X,2)],
# it adds the weighted-orthogonal residuals of A^2 Y, A^3 Y, and A^4 Y.
# The first generator image A Y is already in the base dictionary.
using LinearAlgebra
using CSV, DataFrames, Plots
using Plots.PlotMeasures
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULT_DIR = joinpath(ROOT, "results", "siads_log_diagnostics")
const CSV_OUTPUT = joinpath(RESULT_DIR, "krylov_depth_population.csv")
const MSE_OUTPUT = joinpath(RESULT_DIR, "krylov_depth_mse.csv")
const OPTIMAL_MSE_OUTPUT = joinpath(RESULT_DIR, "krylov_depth_optimal_mse.csv")
const PNG_OUTPUT = joinpath(ROOT, "figures", "krylov_depth_population.png")
const PDF_OUTPUT = joinpath(RESULT_DIR, "krylov_depth_population.pdf")

function model(::Type{T}=Float64; total=24, c1=T(1) / 40, c2=T(7) / 10) where {T}
    ys = collect(0:fld(total, 2))
    Q = zeros(T, length(ys), length(ys))
    Psi = zeros(T, 3, length(ys))
    for (j, y) in enumerate(ys)
        x = total - 2y
        a1 = c1 * x * (x - 1) / 2
        a2 = c2 * y
        if a1 > 0
            Q[j + 1, j] += a1
            Q[j, j] -= a1
        end
        if a2 > 0
            Q[j - 1, j] += a2
            Q[j, j] -= a2
        end
        Psi[:, j] .= (one(T), T(y), T(x * (x - 1)) / 2)
    end
    Q, Psi, [c1, c2]
end

function weighted_residual(v, basis, weights)
    gram = basis * Diagonal(weights) * basis'
    coefficients = (reshape(v, 1, :) * Diagonal(weights) * basis') / gram
    v - vec(coefficients * basis)
end

function depth_dictionary(Q, base, weights, depth)
    depth >= 1 || throw(ArgumentError("depth must be positive"))
    basis = copy(base)
    image = copy(vec(base[2, :]))
    for order in 1:depth
        image = vec(reshape(image, 1, :) * Q)
        order == 1 && continue # A Y already lies in the base span.
        residual = weighted_residual(image, basis, weights)
        residual_norm = sqrt(dot(weights, residual .^ 2))
        residual_norm > typeof(residual_norm)(1e-30) *
                        max(norm(image), one(residual_norm)) ||
            throw(ArgumentError("generator image at depth $order is dependent"))
        basis = vcat(basis, reshape(residual / residual_norm, 1, :))
    end
    basis
end

function matrix_exp_series(A::Matrix{BigFloat})
    scale = max(0, ceil(Int, log2(Float64(opnorm(A, Inf) / BigFloat("0.25")))))
    B = A / BigFloat(2)^scale
    result = Matrix{BigFloat}(I, size(A)...)
    term = copy(result)
    for order in 1:1000
        term = term * B / order
        result .+= term
        opnorm(term, Inf) < BigFloat("1e-70") && break
        order == 1000 && error("high-precision exponential did not converge")
    end
    for _ in 1:scale
        result = result * result
    end
    result
end

function matrix_log_series(K::Matrix{BigFloat})
    X = K - I
    maximum(abs.(eigvals(Float64.(X)))) < 1 ||
        error("Mercator series is not convergent")
    result = zeros(BigFloat, size(K))
    term = copy(X)
    for order in 1:20000
        result .+= (isodd(order) ? one(BigFloat) : -one(BigFloat)) .* term ./ order
        opnorm(term, Inf) / order < BigFloat("1e-60") && return result
        term = term * X
    end
    error("high-precision logarithm did not converge")
end

function compressed_operator(Q, Psi, weights, h)
    G0 = Psi * Diagonal(weights) * Psi'
    transition = eltype(Q) == BigFloat ? matrix_exp_series(h * Q) : exp(h * Q)
    Gh = (Psi * transition) * Diagonal(weights) * Psi'
    Gh / G0
end

rates(B) = [B[2, 3], -B[2, 2]]

function frechet_log(K, E)
    d = size(K, 1)
    Z = zeros(d, d)
    real.(log(complex.([K E; Z K]))[1:d, d+1:2d])
end

function population_statistics(Q, Psi, weights, truth, h)
    P = exp(h * Q)
    G0 = Psi * Diagonal(weights) * Psi'
    K = (Psi * P) * Diagonal(weights) * Psi' / G0
    B = real.(log(complex.(K))) / h
    estimate = rates(B)
    relative_bias = sqrt(sum(((estimate .- truth) ./ truth) .^ 2) / 2)

    V = zeros(2, 2)
    G0_inverse = inv(G0)
    for i in axes(Psi, 2), j in axes(Psi, 2)
        probability = weights[i] * P[j, i]
        probability == 0 && continue
        psi_i = Psi[:, i]
        E = (Psi[:, j] - K * psi_i) * (psi_i' * G0_inverse)
        dB = frechet_log(K, E) / h
        z = rates(dB)
        V .+= probability .* (z * z')
    end
    Dc_inverse = Diagonal(1 ./ truth)
    relative_variance = tr(Dc_inverse * V * Dc_inverse) / 2
    (; estimate, relative_bias, relative_variance, gram_condition=cond(G0))
end

function fitted_slope(lags, errors)
    # The exact-bias calculation uses 256-bit arithmetic. We fit a common
    # short-interval window rather than selecting a different range by depth.
    selected = (lags .>= 0.002) .& (lags .<= 0.06)
    count(selected) >= 4 || error("too few points for slope fit")
    h = log.(lags[selected])
    e = log.(errors[selected])
    hcat(ones(length(h)), h) \ e |> last
end

function main()
    Q, base, truth = model()
    weights = collect(1.0:size(Q, 1))
    weights ./= sum(weights)
    lags = 10.0 .^ range(log10(5e-4), log10(0.2), length=28)
    depths = 1:4

    rows = DataFrame(depth=Int[], dimension=Int[], lag=Float64[],
                     rate=String[], estimate=Float64[], truth=Float64[],
                     relative_error=Float64[], aggregate_relative_bias=Float64[],
                     relative_variance=Float64[], gram_condition=Float64[])
    dictionaries = Dict(depth => depth_dictionary(Q, base, weights, depth)
                        for depth in depths)
    Q_big, base_big, truth_big = setprecision(256) do
        model(BigFloat)
    end
    weights_big = BigFloat.(collect(1:size(Q_big, 1)))
    weights_big ./= sum(weights_big)
    dictionaries_big = Dict(depth => depth_dictionary(
        Q_big, base_big, weights_big, depth) for depth in depths)
    statistics = Dict{Tuple{Int, Float64}, NamedTuple}()
    for depth in depths, h in lags
        Psi = dictionaries[depth]
        stats = population_statistics(Q, Psi, weights, truth, h)
        estimate_big, relative_bias_big = setprecision(256) do
            K_big = compressed_operator(
                Q_big, dictionaries_big[depth], weights_big, BigFloat(h))
            B_big = matrix_log_series(K_big) / BigFloat(h)
            estimate = rates(B_big)
            bias = sqrt(sum(((estimate .- truth_big) ./ truth_big) .^ 2) / 2)
            estimate, bias
        end
        estimate = Float64.(estimate_big)
        relative_bias = Float64(relative_bias_big)
        statistics[(depth, h)] = merge(stats, (; estimate, relative_bias))
        for rate in eachindex(truth)
            push!(rows, (depth, size(Psi, 1), h, "c$rate", estimate[rate],
                         truth[rate], abs(estimate[rate] - truth[rate]) / truth[rate],
                         relative_bias, stats.relative_variance,
                         stats.gram_condition))
        end
    end

    slopes = Dict(depth => fitted_slope(
        lags, [statistics[(depth, h)].relative_bias for h in lags])
        for depth in depths)
    for depth in depths
        @printf("depth=%d dimension=%d slope=%.4f gram_condition=%.4e\n",
                depth, size(dictionaries[depth], 1), slopes[depth],
                statistics[(depth, first(lags))].gram_condition)
    end

    sample_sizes = 10.0 .^ range(2, 10, length=65)
    mse_lags = [0.03, 0.1]
    mse_rows = DataFrame(depth=Int[], dimension=Int[], lag=Float64[],
                          samples=Float64[], relative_bias=Float64[],
                          relative_sampling_rms=Float64[],
                          predicted_relative_mse=Float64[])
    for h in mse_lags, depth in depths
        stats = population_statistics(Q, dictionaries[depth], weights, truth, h)
        relative_bias = statistics[(depth, lags[argmin(abs.(lags .- h))])].relative_bias
        # Evaluate the exact bias at the requested MSE lag rather than using
        # the nearest plotting point.
        relative_bias = Float64(setprecision(256) do
            K_big = compressed_operator(
                Q_big, dictionaries_big[depth], weights_big, BigFloat(h))
            B_big = matrix_log_series(K_big) / BigFloat(h)
            estimate = rates(B_big)
            sqrt(sum(((estimate .- truth_big) ./ truth_big) .^ 2) / 2)
        end)
        for samples in sample_sizes
            sampling_rms = sqrt(stats.relative_variance / samples)
            mse = relative_bias^2 + sampling_rms^2
            push!(mse_rows, (depth, size(dictionaries[depth], 1), h, samples,
                              relative_bias, sampling_rms, mse))
        end
    end

    optimization_lags = 10.0 .^ range(log10(2e-4), log10(0.5), length=120)
    optimization_statistics = Dict{Tuple{Int, Float64}, NamedTuple}()
    for depth in depths, h in optimization_lags
        stats = population_statistics(Q, dictionaries[depth], weights, truth, h)
        relative_bias = Float64(setprecision(256) do
            K_big = compressed_operator(
                Q_big, dictionaries_big[depth], weights_big, BigFloat(h))
            B_big = matrix_log_series(K_big) / BigFloat(h)
            estimate = rates(B_big)
            sqrt(sum(((estimate .- truth_big) ./ truth_big) .^ 2) / 2)
        end)
        optimization_statistics[(depth, h)] =
            (; relative_bias, stats.relative_variance)
    end

    optimal_rows = DataFrame(depth=Int[], dimension=Int[], samples=Float64[],
                             optimal_lag=Float64[],
                             predicted_relative_mse=Float64[])
    for depth in depths, samples in sample_sizes
        values = [begin
            stats = optimization_statistics[(depth, h)]
            stats.relative_bias^2 + stats.relative_variance / samples
        end for h in optimization_lags]
        index = argmin(values)
        push!(optimal_rows, (depth, size(dictionaries[depth], 1), samples,
                             optimization_lags[index], values[index]))
    end

    mkpath(RESULT_DIR)
    CSV.write(CSV_OUTPUT, rows)
    CSV.write(MSE_OUTPUT, mse_rows)
    CSV.write(OPTIMAL_MSE_OUTPUT, optimal_rows)

    colors = [:gray25, :navy, :darkorange, :firebrick]
    markers = [:circle, :square, :diamond, :utriangle]
    p1 = plot(xscale=:log10, yscale=:log10, xlabel="observation interval h",
              ylabel="exact relative rate-bias norm", framestyle=:box,
              gridalpha=0.18, legend=:topleft, title="Projection bias")
    for (index, depth) in enumerate(depths)
        errors = [statistics[(depth, h)].relative_bias for h in lags]
        label = "depth $depth, slope $(round(slopes[depth], digits=2))"
        plot!(p1, lags, errors; label, color=colors[index], marker=markers[index],
              markersize=3, linewidth=1.5)
    end

    p2 = plot(xscale=:log10, yscale=:log10,
              xlabel="independent snapshot pairs",
              ylabel="minimum predicted relative MSE", framestyle=:box,
              gridalpha=0.18, legend=:bottomleft,
              title="MSE after optimizing h", xlims=(1e2, 1e10))
    for (index, depth) in enumerate(depths)
        subset = optimal_rows[optimal_rows.depth .== depth, :]
        plot!(p2, subset.samples, subset.predicted_relative_mse;
              label="depth $depth", color=colors[index], marker=markers[index],
              markevery=8, markersize=3, linewidth=1.5)
    end

    plot!(p1; left_margin=7mm, bottom_margin=5mm)
    plot!(p2; left_margin=6mm, bottom_margin=5mm)
    figure = plot(p1, p2; layout=(1, 2), size=(1050, 500), dpi=180)
    savefig(figure, PNG_OUTPUT)
    savefig(figure, PDF_OUTPUT)
    println("saved $CSV_OUTPUT")
    println("saved $MSE_OUTPUT")
    println("saved $OPTIMAL_MSE_OUTPUT")
    println("saved $PNG_OUTPUT")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

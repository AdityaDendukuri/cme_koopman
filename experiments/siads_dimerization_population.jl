# Exact population experiment for the finite, conservative CME theorem.
#
#   2X -> Y,  propensity c1 * binomial(X,2)
#    Y -> 2X, propensity c2 * Y
#
# The conserved quantity X + 2Y = N makes the state space finite without a
# sink.  The dictionary [1, Y, binomial(X,2)] contains the first generator
# image of Y but is not invariant under later generator images.
using LinearAlgebra
using CSV, DataFrames, Plots
using Plots.PlotMeasures

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULT_DIR = joinpath(ROOT, "results", "siads_log_diagnostics")
const CSV_OUTPUT = joinpath(RESULT_DIR, "dimerization_population.csv")
const PNG_OUTPUT = joinpath(ROOT, "figures", "dimerization_population.png")
const PDF_OUTPUT = joinpath(RESULT_DIR, "dimerization_population.pdf")

function model(; total=24, c1=0.025, c2=0.7)
    ys = collect(0:fld(total, 2))
    n = length(ys)
    Q = zeros(n, n)
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
    end
    Psi = zeros(3, n)
    for (j, y) in enumerate(ys)
        x = total - 2y
        Psi[:, j] .= (1.0, y, x * (x - 1) / 2)
    end
    return Q, Psi, [c1, c2]
end

function compressed_operator(Q, Psi, weights, h)
    G0 = Psi * Diagonal(weights) * transpose(Psi)
    Gh = (Psi * exp(h * Q)) * Diagonal(weights) * transpose(Psi)
    Gh / G0
end

function rates(B)
    [B[2, 3], -B[2, 2]]
end

function weak_rates(Q, Psi, weights, h, rule)
    C0 = Psi * Diagonal(weights) * transpose(Psi)
    Ch2 = (Psi * exp((h / 2) * Q)) * Diagonal(weights) * transpose(Psi)
    Ch = (Psi * exp(h * Q)) * Diagonal(weights) * transpose(Psi)
    M = if rule == :trapezoid
        (h / 2) * (C0 + Ch)
    elseif rule == :simpson
        (h / 6) * (C0 + 4Ch2 + Ch)
    else
        error("unknown quadrature rule $rule")
    end
    # The retained species is Y.  Its unit-rate generator rows are
    # +binomial(X,2) and -Y for the two reactions.
    R1 = reshape([0.0, 0.0, 1.0], 1, :)
    R2 = reshape([0.0, -1.0, 0.0], 1, :)
    design = hcat(vec(R1 * M), vec(R2 * M))
    response = vec((Ch - C0)[2:2, :])
    design \ response
end

function frechet_log(K, H)
    d = size(K, 1)
    Z = zeros(eltype(K), d, d)
    M = [K H; Z K]
    real.(log(complex.(M))[1:d, d+1:2d])
end

function rate_gain(K, h, W)
    norm(frechet_log(transpose(K), W)) / h
end

function main()
    Q, Psi, truth = model()
    n = size(Q, 1)
    # Full support gives a nonsingular population Gram matrix.  A mildly
    # nonuniform design avoids building symmetry into the projection.
    weights = collect(1.0:n)
    weights ./= sum(weights)
    lags = 10.0 .^ range(log10(0.003), log10(0.3), length=18)
    rows = DataFrame(lag=Float64[], method=String[], rate=String[],
                     estimate=Float64[], truth=Float64[], rel_error=Float64[],
                     log_gain=Float64[])
    W1 = zeros(3, 3); W1[2, 3] = 1
    W2 = zeros(3, 3); W2[2, 2] = -1
    for h in lags
        Kh = compressed_operator(Q, Psi, weights, h)
        Kh2 = compressed_operator(Q, Psi, weights, h / 2)
        B = real.(log(complex.(Kh))) / h
        Bdc = B + (4 / (3h)) * (Kh - Kh2^2)
        Bdiff = (Kh - I) / h
        Bnewton2 = (-3I + 4Kh2 - Kh) / h

        # Degree-three Newton interpolation of K_t at 0,h/3,2h/3,h,
        # differentiated at zero (Sechi--Sikorski--Weber comparator).
        delta = h / 3
        K1 = compressed_operator(Q, Psi, weights, delta)
        K2 = compressed_operator(Q, Psi, weights, 2delta)
        Bnewton = (-11I / 6 + 3K1 - 3K2 / 2 + Kh / 3) / delta

        gains = (rate_gain(Kh, h, W1), rate_gain(Kh, h, W2))
        for (name, estimate) in (("difference quotient", rates(Bdiff)),
                                 ("ordinary log", rates(B)),
                                 ("two-interval corrected log", rates(Bdc)),
                                 ("quadratic Newton", rates(Bnewton2)),
                                 ("weak trapezoid", weak_rates(Q, Psi, weights, h, :trapezoid)),
                                 ("weak Simpson", weak_rates(Q, Psi, weights, h, :simpson)),
                                 ("Newton extrapolation", rates(Bnewton)))
            for r in 1:2
                push!(rows, (h, name, "c$r", estimate[r], truth[r],
                             abs(estimate[r] - truth[r]) / truth[r], gains[r]))
            end
        end
    end
    mkpath(RESULT_DIR)
    CSV.write(CSV_OUTPUT, rows)

    # Aggregate the two rate errors so the reference slopes are unambiguous.
    p1 = plot(xscale=:log10, yscale=:log10, xlabel="observation interval h",
              ylabel="relative rate-error norm", framestyle=:box,
              gridalpha=0.18, legend=:topleft, legendfontsize=7,
              title="Exact projection error")
    styles = Dict("difference quotient" => (:diamond, :dot, :darkgreen),
                  "ordinary log" => (:circle, :solid, :gray30),
                  "two-interval corrected log" => (:square, :solid, :navy),
                  "quadratic Newton" => (:dtriangle, :dashdot, :purple),
                  "weak trapezoid" => (:hexagon, :dash, :darkorange),
                  "weak Simpson" => (:star5, :solid, :teal),
                  "Newton extrapolation" => (:utriangle, :dash, :firebrick))
    for method in keys(styles)
        err = Float64[]
        for h in lags
            sub = rows[(rows.lag .== h) .& (rows.method .== method), :]
            push!(err, norm(sub.estimate .- sub.truth) / norm(sub.truth))
        end
        marker, line, color = styles[method]
        plot!(p1, lags, err; label=method, marker, linestyle=line, color,
              markersize=3.2, linewidth=1.5)
    end
    anchor1 = 0.45 .* (lags ./ maximum(lags))
    anchor2 = 0.16 .* (lags ./ maximum(lags)).^2
    anchor3 = 0.045 .* (lags ./ maximum(lags)).^3
    anchor4 = 0.014 .* (lags ./ maximum(lags)).^4
    plot!(p1, lags, anchor1; label="slope 1", color=:gray55, linestyle=:dash)
    plot!(p1, lags, anchor2; label="slope 2", color=:gray55, linestyle=:dot)
    plot!(p1, lags, anchor3; label="slope 3", color=:gray55, linestyle=:dashdot)
    plot!(p1, lags, anchor4; label="slope 4", color=:gray55, linestyle=:dot)

    ordinary = rows[rows.method .== "ordinary log", :]
    p2 = plot(xscale=:log10, yscale=:log10, xlabel="observation interval h",
              ylabel="parameter sensitivity to operator error", framestyle=:box,
              gridalpha=0.18, legend=:topright,
              title="Measured parameter sensitivity")
    for (r, marker, color) in (("c1", :circle, :navy), ("c2", :square, :firebrick))
        sub = ordinary[ordinary.rate .== r, :]
        plot!(p2, sub.lag, sub.log_gain; label=r, marker, color,
              markersize=3.2, linewidth=1.5)
    end

    plot!(p1; left_margin=7mm, bottom_margin=5mm)
    plot!(p2; left_margin=5mm, bottom_margin=5mm)
    fig = plot(p1, p2; layout=(1, 2), size=(1050, 500), dpi=180)
    savefig(fig, PNG_OUTPUT)
    savefig(fig, PDF_OUTPUT)
    println("saved $CSV_OUTPUT")
    println("saved $PNG_OUTPUT")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

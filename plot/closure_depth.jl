# Plot the exact finite-state closure-depth calculation from retained CSV data.
using CSV, DataFrames, Plots
using Plots.PlotMeasures

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULT_DIR = joinpath(ROOT, "results", "siads_log_diagnostics")
const OUTPUT = joinpath(ROOT, "figures", "krylov_depth_population.png")

function fitted_slope(lags, errors)
    selected = (lags .>= 0.002) .& (lags .<= 0.06)
    h = log.(lags[selected])
    e = log.(errors[selected])
    last(hcat(ones(length(h)), h) \ e)
end

function main()
    rows = CSV.read(joinpath(RESULT_DIR, "krylov_depth_population.csv"), DataFrame)
    optimal_rows = CSV.read(joinpath(RESULT_DIR, "krylov_depth_optimal_mse.csv"),
                            DataFrame)
    depths = 1:4
    colors = [:gray25, :navy, :darkorange, :firebrick]
    markers = [:circle, :square, :diamond, :utriangle]

    p1 = plot(xscale=:log10, yscale=:log10, xlabel="observation interval h",
              ylabel="exact relative rate-bias norm", framestyle=:box,
              gridalpha=0.18, legend=:topleft, title="Projection bias")
    for (index, depth) in enumerate(depths)
        subset = sort(rows[(rows.depth .== depth) .& (rows.rate .== "c1"), :], :lag)
        slope = fitted_slope(subset.lag, subset.aggregate_relative_bias)
        label = "depth $depth, slope $(round(slope, digits=2))"
        plot!(p1, subset.lag, subset.aggregate_relative_bias;
              label, color=colors[index], marker=markers[index],
              markersize=3, linewidth=1.5)
    end

    p2 = plot(xscale=:log10, yscale=:log10,
              xlabel="independent snapshot pairs",
              ylabel="minimum predicted relative MSE", framestyle=:box,
              gridalpha=0.18, legend=:bottomleft,
              title="MSE after optimizing h", xlims=(1e2, 1e10))
    for (index, depth) in enumerate(depths)
        subset = sort(optimal_rows[optimal_rows.depth .== depth, :], :samples)
        plot!(p2, subset.samples, subset.predicted_relative_mse;
              label="depth $depth", color=colors[index], marker=markers[index],
              markevery=8, markersize=3, linewidth=1.5)
    end

    plot!(p1; left_margin=7mm, bottom_margin=5mm)
    plot!(p2; left_margin=6mm, bottom_margin=5mm)
    figure = plot(p1, p2; layout=(1, 2), size=(1050, 500), dpi=180)
    mkpath(dirname(OUTPUT))
    savefig(figure, OUTPUT)
    println("saved $OUTPUT")
end

main()

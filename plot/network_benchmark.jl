# Plot the matched multi-network closure benchmark.
using CSV, DataFrames, Plots, Statistics
using Plots.PlotMeasures

const ROOT = normpath(joinpath(@__DIR__, ".."))
const INPUT = joinpath(ROOT, "results", "koopman_closure_benchmark", "selected",
                       "summary.csv")
const PER_SEED_INPUT = joinpath(ROOT, "results", "koopman_closure_benchmark", "selected",
                                "per_seed.csv")
const OUTPUT = joinpath(ROOT, "figures", "closure_network_benchmark.png")
const ORDER = ["Lotka--Volterra", "Michaelis--Menten", "Brusselator", "MAPK cascade"]
const SHORT = ["Lotka--\nVolterra", "Michaelis--\nMenten", "Brusselator", "MAPK\ncascade"]
const COLORS = [:steelblue, :indianred, :gray60]

function ordered_rows(table, estimator)
    subset = table[table.estimator .== estimator, :]
    subset[[findfirst(==(name), subset.model) for name in ORDER], :]
end

function plot_benchmark()
    table = CSV.read(INPUT, DataFrame)
    per_seed = CSV.read(PER_SEED_INPUT, DataFrame)
    x = collect(1:4)
    offsets = [-0.14, 0.14]
    labels = ["Base", "Selected closure"]

    accuracy = plot(; yscale=:log10, ylabel="Mean absolute relative error (%)",
                    xticks=(x, SHORT), title="(a) Rate recovery",
                    titleloc=:left, framestyle=:box, gridalpha=0.18,
                    legend=:topleft, left_margin=8mm, bottom_margin=5mm)
    for (j, estimator) in enumerate(("base", "closure"))
        rows = ordered_rows(table, estimator)
        center = 100 .* rows.median_error
        lower = center .- 100 .* rows.q25_error
        upper = 100 .* rows.q75_error .- center
        scatter!(accuracy, x .+ offsets[j], center;
                 yerror=(lower, upper), marker=:circle, markersize=4,
                 markerstrokewidth=0, markercolor=COLORS[j],
                 linecolor=:gray25, linewidth=1.2, label=labels[j])
    end

    dimensions = plot(; ylabel="Dictionary dimension", xticks=(x, ["LV","MM","Bru","MAPK"]),
                      title="(b) Closure size", titleloc=:left,
                      framestyle=:box, gridalpha=0.18, legend=:topleft,
                      bottom_margin=5mm)
    base_rows = ordered_rows(table, "base")
    closure_rows = ordered_rows(table, "closure")
    dimension_offsets = [-0.22, 0.0, 0.22]
    for (j, values, label) in
        ((1, base_rows.dimension, "base"),
         (2, closure_rows.dimension, "selected"),
         (3, closure_rows.full_dimension, "full depth two"))
        bar!(dimensions, x .+ dimension_offsets[j], values;
             bar_width=0.21, color=COLORS[j], linecolor=COLORS[j], label)
    end

    paired = plot(; ylabel="Closure error / base error",
                  xticks=(x, ["LV","MM","Bru","MAPK"]),
                  title="(c) Paired error ratio", titleloc=:left,
                  framestyle=:box, gridalpha=0.18, legend=:topright,
                  ylims=(0, 1.9), left_margin=7mm, bottom_margin=5mm)
    hline!(paired, [1.0]; color=:gray45, linestyle=:dash,
           linewidth=1.2, label="equal error")
    for (i, model) in enumerate(ORDER)
        rows = per_seed[per_seed.model .== model, :]
        base = sort(rows[rows.estimator .== "base", :], :seed)
        closure = sort(rows[rows.estimator .== "closure", :], :seed)
        base.seed == closure.seed || error("unmatched seeds for $model")
        ratios = closure.mean_relative_error ./ base.mean_relative_error
        jitter = range(-0.13, 0.13, length=length(ratios))
        scatter!(paired, fill(x[i], length(ratios)) .+ jitter, ratios;
                 marker=:circle, markersize=3.2, markerstrokewidth=0,
                 color=:gray35, alpha=0.75, label="")
        scatter!(paired, [x[i]], [median(ratios)]; marker=:diamond,
                 markersize=5, markerstrokewidth=0, color=:indianred,
                 label=i == 1 ? "median" : "")
    end

    figure = plot(accuracy, dimensions, paired; layout=(1,3),
                  size=(1120,360), dpi=220, margin=2mm)
    mkpath(dirname(OUTPUT))
    savefig(figure, OUTPUT)
    println(OUTPUT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_benchmark()
end

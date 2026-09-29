# Plot the controlled Brusselator lag sweep from its retained CSV records.
using CSV, DataFrames, Plots
using Plots.PlotMeasures

const ROOT = normpath(get(ENV, "KOOPMAN_RATE_ROOT", joinpath(@__DIR__, "..")))
const INPUT = joinpath(ROOT, "results", "siads_lag_regimes", "summary.csv")
const FIGURE_DIR = joinpath(ROOT, "figures")

function rate_rows(table, rate)
    sort(table[table.rate .== rate, :], :lag)
end

function plot_lag_regimes()
    table = CSV.read(INPUT, DataFrame)
    mkpath(FIGURE_DIR)
    colors = [:navy, :darkorange, :seagreen, :purple]
    panels = Any[]
    for rate in 1:4
        data = rate_rows(table, "k$rate")
        estimate = data.estimate ./ data.truth
        bootstrap = data.bootstrap_sd ./ data.truth
        reference = data.reference ./ data.truth
        reference_se = data.reference_se ./ data.truth
        panel = plot(data.lag, estimate; yerror=bootstrap, xscale=:log10,
                     marker=:circle, markersize=3, linewidth=1.4,
                     color=colors[rate], label="1,000 trajectories ± 1 bootstrap SD",
                     xlabel=rate > 2 ? "observation interval h" : "",
                     ylabel="estimate / true rate", title=("c₁", "c₂", "c₃", "c₄")[rate],
                     xlims=(4e-4, 0.5), framestyle=:box, gridalpha=0.15,
                     left_margin=7mm, bottom_margin=5mm,
                     legend=rate == 1 ? :outertop : false)
        plot!(panel, data.lag, reference; marker=:square, markersize=3,
              linewidth=1, linestyle=:dash, color=:black,
              ribbon=2 .* reference_se, fillalpha=0.18,
              label="10,000-trajectory reference ± 2 MC SE")
        hline!(panel, [1.0]; color=:gray55, linestyle=:dot, label="")
        push!(panels, panel)
    end
    recovery = plot(panels...; layout=(2, 2), size=(1000, 680), dpi=180)
    savefig(recovery, joinpath(FIGURE_DIR, "brusselator_lag_regimes.png"))
    println("saved Brusselator lag-regime figure")
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_lag_regimes()
end

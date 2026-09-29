using CSV, DataFrames, Plots
using Plots.PlotMeasures

const ROOT = normpath(joinpath(@__DIR__, ".."))
const INPUT = joinpath(ROOT, "results", "siads_log_diagnostics",
                       "dimerization_modal_residues.csv")
const OUTPUT = joinpath(ROOT, "figures", "dimerization_modal_residues.pdf")

rows = CSV.read(INPUT, DataFrame)
p1 = scatter(rows.real_part, rows.residue_frobenius;
             xlabel="omitted decay rate Re(μ)", ylabel="residue norm",
             yscale=:log10, title="(a) Omitted-mode coupling",
             markercolor=:white, markerstrokecolor=:black,
             markerstrokewidth=1.1, markersize=5, legend=false,
             framestyle=:box, grid=false)

displayed = first(rows, min(6, nrow(rows)))
order = reverse(1:nrow(displayed))
labels = [displayed.imaginary_magnitude[i] < 1e-9 ?
          "$(round(displayed.real_part[i]; digits=2))" :
          "$(round(displayed.real_part[i]; digits=2)) ± " *
          "$(round(displayed.imaginary_magnitude[i]; digits=2))i"
          for i in order]
effects = displayed.relative_rate_change[order]
floor_value = minimum(effects) / 2
p2 = scatter(effects, 1:length(order);
             xlabel="Relative change in inferred rates", ylabel="Omitted mode μ",
             xscale=:log10, yticks=(1:length(order), labels),
             title="(b) Rate effect at h=0.1", markercolor=:white,
             markerstrokecolor=:black, markerstrokewidth=1.1, markersize=5,
             xlims=(floor_value, 2maximum(effects)), legend=false,
             framestyle=:box, grid=false)
for (position, effect) in enumerate(effects)
    plot!(p2, [floor_value, effect], [position, position];
          color=:black, linewidth=1, label=false)
end

figure = plot(p1, p2; layout=(1, 2), size=(920, 340),
              left_margin=5mm, bottom_margin=4mm)
mkpath(dirname(OUTPUT))
savefig(figure, OUTPUT)
println(OUTPUT)

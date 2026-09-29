using CairoMakie
using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULT_DIR = joinpath(ROOT, "results", "siads_lag_regimes")
const OUTPUT = joinpath(ROOT, "figures", "brusselator_spectral_diagnostic.png")

spectra = CSV.read(joinpath(RESULT_DIR, "spectrum.csv"), DataFrame)
margins = CSV.read(joinpath(RESULT_DIR, "spectral_diagnostics.csv"), DataFrame)
set_theme!(Theme(fontsize=11, Axis=(xgridvisible=false, ygridvisible=false,)))
fig = Figure(size=(850, 325))
spectral_axis = Axis(fig[1, 1], title="(a) Projected Koopman spectrum",
                     xlabel="Re λ(Kₕ)", ylabel="Im λ(Kₕ)",
                     aspect=DataAspect(), limits=((-1.25, 1.1), nothing))
margin_axis = Axis(fig[1, 2], title="(b) Distance to the logarithm cut",
                   xlabel="Observation interval h", ylabel="Branch margin δₕ",
                   xscale=log10, yscale=log10)

for (lag, color, marker) in zip([0.1, 0.4, 0.8],
                                [:dodgerblue3, :seagreen4, :firebrick3],
                                [:circle, :rect, :utriangle])
    selected = subset(spectra, :ensemble => ByRow(==(1000)),
                      :lag => ByRow(x -> isapprox(x, lag)))
    scatter!(spectral_axis, selected.eigenvalue_real, selected.eigenvalue_imag;
             color=:transparent, strokecolor=color, strokewidth=1.4,
             marker, markersize=10, label="h=$(lag)")
end
theta = range(0, 2pi; length=500)
lines!(spectral_axis, cos.(theta), sin.(theta); color=:gray75, linewidth=0.8)
lines!(spectral_axis, [-1.25, 0], [0, 0]; color=:firebrick4,
       linewidth=2.5, label="principal-log cut")
hlines!(spectral_axis, [0]; color=:gray75, linewidth=0.5)
vlines!(spectral_axis, [0]; color=:gray75, linewidth=0.5)
axislegend(spectral_axis; position=:lb, framevisible=false, labelsize=9)

for (ensemble, style, marker, label) in
    [(1000, :solid, :circle, "1,000 trajectories"),
     (10000, :dash, :rect, "10,000-trajectory reference")]
    selected = subset(margins, :ensemble => ByRow(==(ensemble)))
    sort!(selected, :lag)
    values = max.(selected.branch_margin, 1e-16)
    lines!(margin_axis, selected.lag, values; linestyle=style, color=:black,
           linewidth=1.2, label)
    scatter!(margin_axis, selected.lag, values;
             marker, color=:black, markersize=6)
end
axislegend(margin_axis; position=:lb, framevisible=false, labelsize=9)
colgap!(fig.layout, 28)
mkpath(dirname(OUTPUT))
save(OUTPUT, fig; px_per_unit=2.5)
println(OUTPUT)

"""
Modal attribution of projection bias in the finite dimerization example.

The retained dictionary is [1, Y, binomial(X,2)].  We construct a weighted
orthonormal retained/complement basis, diagonalize the omitted generator
block, and report for each real mode

    C_l = A_JJ' v_l w_l^* A_J'J.

The leave-one-mode-out diagnostic sets both coupling blocks for that mode to
zero, recomputes the projected finite-time operator, and measures the change
in the two inferred rates.  This is a finite-dimensional diagnostic, not an
additive decomposition of the nonlinear matrix logarithm.
"""

using LinearAlgebra
using CSV, DataFrames

include("siads_dimerization_population.jl")

const OUTPUT_DIR = joinpath(ROOT, "results", "siads_log_diagnostics")
const CSV_PATH = joinpath(OUTPUT_DIR, "dimerization_modal_residues.csv")
const FIGURE_PATH = joinpath(
    ROOT,
    "figures",
    "dimerization_modal_residues.pdf",
)
const H = 0.1

function weighted_block_generator(Q, Psi, weights)
    n = size(Q, 1)
    sqrt_weights = sqrt.(weights)
    inverse_sqrt_weights = 1.0 ./ sqrt_weights
    G = Symmetric(Psi * Diagonal(weights) * Psi')
    decomposition = eigen(G)
    T = Diagonal(1.0 ./ sqrt.(decomposition.values)) *
        transpose(decomposition.vectors)
    retained_rows = T * Psi * Diagonal(sqrt_weights)
    retained_columns = transpose(retained_rows)
    complement_columns = nullspace(transpose(retained_columns))
    V = hcat(retained_columns, complement_columns)
    L = Diagonal(inverse_sqrt_weights) * Q * Diagonal(sqrt_weights)
    B = transpose(V) * L * V
    d = size(Psi, 1)
    (; B, T, d)
end

physical_rates(Borth, T) = rates(inv(T) * Borth * T)

function paired_modes(values; tolerance=1e-9)
    unused = trues(length(values))
    groups = Vector{Vector{Int}}()
    for index in eachindex(values)
        unused[index] || continue
        value = values[index]
        if abs(imag(value)) <= tolerance
            push!(groups, [index])
            unused[index] = false
            continue
        end
        partner = argmin([
            unused[j] && j != index ? abs(values[j] - conj(value)) : Inf
            for j in eachindex(values)
        ])
        isfinite(abs(values[partner] - conj(value))) ||
            error("could not pair complex mode $value")
        push!(groups, sort([index, partner]))
        unused[index] = false
        unused[partner] = false
    end
    groups
end

function analyze()
    Q, Psi, truth = model()
    weights = collect(1.0:size(Q, 1))
    weights ./= sum(weights)
    blocks = weighted_block_generator(Q, Psi, weights)
    B, T, d = blocks.B, blocks.T, blocks.d
    B11 = B[1:d, 1:d]
    B12 = B[1:d, d+1:end]
    B21 = B[d+1:end, 1:d]
    B22 = B[d+1:end, d+1:end]

    eigendecomposition = eigen(B22)
    values = eigendecomposition.values
    right = eigendecomposition.vectors
    left = inv(right)
    groups = paired_modes(values)

    Bmodal = [B11 B12 * right; left * B21 Diagonal(values)]
    Kfull = exp(H * Bmodal)[1:d, 1:d]
    full_rates = physical_rates(real.(log(complex.(Kfull))) / H, T)
    full_relative_bias = norm((full_rates - truth) ./ truth)

    rows = NamedTuple[]
    for (group_index, group) in enumerate(groups)
        coupling = zeros(ComplexF64, d, d)
        for index in group
            coupling .+= B12 * right[:, index] *
                         transpose(left[index, :]) * B21
        end

        decoupled = copy(Bmodal)
        for index in group
            decoupled[1:d, d + index] .= 0
            decoupled[d + index, 1:d] .= 0
        end
        Kwithout = exp(H * decoupled)[1:d, 1:d]
        rates_without = physical_rates(
            real.(log(complex.(Kwithout))) / H,
            T,
        )
        relative_rate_change = norm((rates_without - full_rates) ./ truth)
        representative = values[first(group)]
        push!(rows, (
            mode_group=group_index,
            multiplicity=length(group),
            real_part=real(representative),
            imaginary_magnitude=abs(imag(representative)),
            residue_frobenius=norm(coupling),
            relative_rate_change=relative_rate_change,
            full_relative_bias=full_relative_bias,
            lag=H,
        ))
    end
    sort!(rows; by=row -> row.relative_rate_change, rev=true)
    DataFrame(rows)
end

function plot_results(rows)
    p1 = scatter(
        rows.real_part,
        rows.residue_frobenius,
        xlabel="omitted decay rate Re(μ)",
        ylabel="residue norm",
        yscale=:log10,
        title="(a) Omitted-mode coupling",
        markercolor=:white,
        markerstrokecolor=:black,
        markerstrokewidth=1.1,
        markersize=5,
        legend=false,
        framestyle=:box,
        grid=false,
        titlefontsize=11,
        guidefontsize=10,
        tickfontsize=9,
    )

    displayed = first(rows, min(6, nrow(rows)))
    order = reverse(1:nrow(displayed))
    labels = [
        abs(displayed.imaginary_magnitude[index]) < 1e-9 ?
            "$(round(displayed.real_part[index]; digits=2))" :
            "$(round(displayed.real_part[index]; digits=2)) ± " *
            "$(round(displayed.imaginary_magnitude[index]; digits=2))i"
        for index in order
    ]
    effects = displayed.relative_rate_change[order]
    floor_value = minimum(effects) / 2
    p2 = scatter(
        effects,
        1:length(order),
        xlabel="Relative change in inferred rates",
        ylabel="Omitted mode μ",
        xscale=:log10,
        yticks=(1:length(order), labels),
        title="(b) Rate effect at h=0.1",
        markercolor=:white,
        markerstrokecolor=:black,
        markerstrokewidth=1.1,
        markersize=5,
        xlims=(floor_value, 2maximum(effects)),
        legend=false,
        framestyle=:box,
        grid=false,
        titlefontsize=11,
        guidefontsize=10,
        tickfontsize=9,
    )
    for (position, effect) in enumerate(effects)
        plot!(p2, [floor_value, effect], [position, position];
              color=:black, linewidth=1, label=false)
    end
    figure = plot(
        p1,
        p2;
        layout=(1, 2),
        size=(920, 340),
        left_margin=5Plots.mm,
        bottom_margin=4Plots.mm,
    )
    savefig(figure, FIGURE_PATH)
end

mkpath(OUTPUT_DIR)
rows = analyze()
CSV.write(CSV_PATH, rows)
plot_results(rows)
show(stdout, MIME("text/plain"), rows)
println("\nsaved $CSV_PATH")
println("saved $FIGURE_PATH")

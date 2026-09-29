# Assemble the twenty-seed benchmark used by the paper.

using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
const BASE = joinpath(ROOT, "results", "koopman_closure_benchmark")
const OUTPUT = joinpath(BASE, "selected")

function selected_rows(path, model)
    table = CSV.read(path, DataFrame)
    table[table.model .== model, :]
end

function main()
    tables = [
        selected_rows(joinpath(BASE, "heldout20_lv", "summary.csv"),
                      "Lotka--Volterra"),
        selected_rows(joinpath(BASE, "heldout20_mm", "summary.csv"),
                      "Michaelis--Menten"),
        selected_rows(joinpath(BASE, "heldout20_bru", "summary.csv"),
                      "Brusselator"),
        selected_rows(joinpath(BASE, "heldout20_mapk", "summary.csv"),
                      "MAPK cascade"),
    ]
    output = vcat(tables...)
    output.estimator = ifelse.(output.estimator .== "base", "base", "closure")
    full_dimensions = Dict("Lotka--Volterra" => 7, "Michaelis--Menten" => 6,
                           "Brusselator" => 11, "MAPK cascade" => 21)
    output.full_dimension = [full_dimensions[model] for model in output.model]
    mkpath(OUTPUT)
    CSV.write(joinpath(OUTPUT, "summary.csv"), output)

    per_seed_tables = [
        selected_rows(joinpath(BASE, "heldout20_lv", "per_seed.csv"),
                      "Lotka--Volterra"),
        selected_rows(joinpath(BASE, "heldout20_mm", "per_seed.csv"),
                      "Michaelis--Menten"),
        selected_rows(joinpath(BASE, "heldout20_bru", "per_seed.csv"),
                      "Brusselator"),
        selected_rows(joinpath(BASE, "heldout20_mapk", "per_seed.csv"),
                      "MAPK cascade"),
    ]
    per_seed = vcat(per_seed_tables...)
    per_seed.estimator = ifelse.(per_seed.estimator .== "base", "base", "closure")
    CSV.write(joinpath(OUTPUT, "per_seed.csv"), per_seed)
    show(stdout, MIME("text/plain"), output); println()
end


main()

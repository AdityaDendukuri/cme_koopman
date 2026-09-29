# Matched finite-depth closure benchmark for the Koopman rate paper.
#
# Each model uses the same simulated trajectory pairs for the base dictionary
# and a rate-independent reaction-word closure. The optional MAX_ADDED setting
# retains a fixed prefix of the empirical residual singular vectors. The added
# candidate functions are
#
#     L_l alpha_k(x) = alpha_l(x) [alpha_k(x + zeta_l) - alpha_k(x)],
#
# where alpha_k is the propensity shape without its unknown rate constant.
# We remove zero and dependent functions using only the current observations.

include(joinpath(@__DIR__, "..", "src", "KoopmanCMEPaperTools.jl"))
using .KoopmanCMEPaperTools
using LinearAlgebra
using Printf
using Random
using Statistics
using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DESIGN = get(ENV, "CLOSURE_BENCHMARK_DESIGN", "baseline")
const RUN_TAG = get(ENV, "CLOSURE_BENCHMARK_TAG",
                    DESIGN == "baseline" ? "lag2" : DESIGN)
const OUT = joinpath(ROOT, "results", "koopman_closure_benchmark", RUN_TAG)
const SMOKE = get(ENV, "CLOSURE_BENCHMARK_SMOKE", "0") == "1"
const SEEDS = haskey(ENV, "CLOSURE_BENCHMARK_SEEDS") ?
    parse.(Int, split(ENV["CLOSURE_BENCHMARK_SEEDS"], ',')) :
    (SMOKE ? [7301] : [7301, 7401, 7501, 7601, 7701])
const LAG_FACTOR = parse(Float64, get(ENV, "CLOSURE_BENCHMARK_LAG_FACTOR", "2"))
const MODEL_FILTER = get(ENV, "CLOSURE_BENCHMARK_MODEL", "")
const INDEPENDENT_PAIRS = get(ENV, "CLOSURE_BENCHMARK_INDEPENDENT_PAIRS", "0") == "1"
const MAX_ADDED = parse(Int, get(ENV, "CLOSURE_BENCHMARK_MAX_ADDED", "0"))

env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_int(name, default) = parse(Int, get(ENV, name, string(default)))

struct ClosureModel
    name::String
    initials::Vector{Vector{Int}}
    stoichiometry::Matrix{Int}
    rates::Vector{Float64}
    propensity_shape::Function
    base_feature::Function
    base_scales::Vector{Float64}
    current_times::Vector{Float64}
    lag::Float64
    paths_per_initial::Int
    extract::Function
end

function simulate_states(model::ClosureModel, initial, times, rng)
    state = copy(initial)
    recorded = Matrix{Int}(undef, length(state), length(times))
    clock = 0.0
    event_time = 0.0
    propensities = model.rates .* model.propensity_shape(state)
    total = sum(propensities)
    event_time = total > 0 ? randexp(rng) / total : Inf
    for (column, sample_time) in enumerate(times)
        while event_time <= sample_time
            threshold = rand(rng) * total
            cumulative = 0.0
            reaction = length(propensities)
            for candidate in eachindex(propensities)
                cumulative += propensities[candidate]
                if threshold <= cumulative
                    reaction = candidate
                    break
                end
            end
            state .+= view(model.stoichiometry, :, reaction)
            any(state .< 0) && error("negative state in $(model.name)")
            clock = event_time
            propensities = model.rates .* model.propensity_shape(state)
            total = sum(propensities)
            event_time = total > 0 ? clock + randexp(rng) / total : Inf
        end
        recorded[:, column] .= state
    end
    recorded
end

function reaction_words(model::ClosureModel, state)
    alpha = model.propensity_shape(state)
    reactions = length(alpha)
    values = zeros(reactions * reactions)
    index = 1
    for l in 1:reactions, k in 1:reactions
        if alpha[l] != 0
            shifted = state .+ view(model.stoichiometry, :, l)
            values[index] = alpha[l] *
                (model.propensity_shape(shifted)[k] - alpha[k])
        end
        index += 1
    end
    values
end

function paired_features(model::ClosureModel, seed)
    current = model.current_times
    if INDEPENDENT_PAIRS
        base_dimension = length(model.base_scales)
        word_dimension = length(model.rates)^2
        pair_count = length(model.initials) * model.paths_per_initial
        Xbase = Matrix{Float64}(undef, base_dimension, pair_count)
        Ybase = similar(Xbase)
        Xword = Matrix{Float64}(undef, word_dimension, pair_count)
        Yword = similar(Xword)
        rng = MersenneTwister(seed)
        column = 1
        for initial in model.initials, pair in 1:model.paths_per_initial
            current_time = rand(rng, current)
            states = simulate_states(model, initial,
                                     [current_time, current_time + model.lag], rng)
            x = view(states, :, 1)
            y = view(states, :, 2)
            Xbase[:,column] .= model.base_feature(x)
            Ybase[:,column] .= model.base_feature(y)
            Xword[:,column] .= reaction_words(model, x)
            Yword[:,column] .= reaction_words(model, y)
            column += 1
        end
        return (; Xbase, Ybase, Xword, Yword)
    end
    future = current .+ model.lag
    times = sort(unique(vcat(current, future)))
    current_indices = searchsortedfirst.(Ref(times), current)
    future_indices = searchsortedfirst.(Ref(times), future)
    base_dimension = length(model.base_scales)
    word_dimension = length(model.rates)^2
    pair_count = length(model.initials) * model.paths_per_initial * length(current)
    Xbase = Matrix{Float64}(undef, base_dimension, pair_count)
    Ybase = similar(Xbase)
    Xword = Matrix{Float64}(undef, word_dimension, pair_count)
    Yword = similar(Xword)
    rng = MersenneTwister(seed)
    column = 1
    for initial in model.initials
        for path in 1:model.paths_per_initial
            states = simulate_states(model, initial, times, rng)
            for j in eachindex(current_indices)
                x = view(states, :, current_indices[j])
                y = view(states, :, future_indices[j])
                Xbase[:, column] .= model.base_feature(x)
                Ybase[:, column] .= model.base_feature(y)
                Xword[:, column] .= reaction_words(model, x)
                Yword[:, column] .= reaction_words(model, y)
                column += 1
            end
        end
    end
    (; Xbase, Ybase, Xword, Yword)
end

function checked_generator(X, Y, lag)
    samples = size(X, 2)
    G0 = X * X' / samples
    G1 = Y * X' / samples
    decomposition = eigen(Symmetric(G0))
    minimum(decomposition.values) > 100eps(Float64) * maximum(decomposition.values) ||
        error("rank-deficient Gram matrix")
    whitening = Diagonal(1 ./ sqrt.(decomposition.values)) * decomposition.vectors'
    Kwhite = whitening * G1 * whitening'
    logarithm = checked_principal_log(Kwhite)
    Bwhite = logarithm.value / lag
    B = whitening \ (Bwhite * whitening)
    (; B, condition=cond(G0), branch_distance=logarithm.branch_distance)
end

function closure_transform(Xbase, Xword; relative_tolerance=1e-8)
    base_gram = Xbase * Xbase'
    regression = (Xword * Xbase') / base_gram
    residual = Xword - regression * Xbase
    decomposition = svd(residual; full=false)
    isempty(decomposition.S) && error("empty closure SVD")
    retained = count(>(relative_tolerance * first(decomposition.S)), decomposition.S)
    MAX_ADDED > 0 && (retained = min(retained, MAX_ADDED))
    modes = decomposition.U[:, 1:retained]'
    base_dimension = size(Xbase, 1)
    word_dimension = size(Xword, 1)
    [Matrix{Float64}(I, base_dimension, base_dimension) zeros(base_dimension, word_dimension);
     -modes * regression modes]
end

function fit_pair(model::ClosureModel, seed)
    data = paired_features(model, seed)
    Xbase = data.Xbase ./ model.base_scales
    Ybase = data.Ybase ./ model.base_scales

    # Remove words that are zero on the sampled support, then scale the rest.
    word_rms = sqrt.(vec(mean(abs2, data.Xword; dims=2)))
    threshold = max(maximum(word_rms), 1.0) * 1e-12
    keep = findall(>(threshold), word_rms)
    Xword = data.Xword[keep, :] ./ word_rms[keep]
    Yword = data.Yword[keep, :] ./ word_rms[keep]

    base_fit = checked_generator(Xbase, Ybase, model.lag)
    transform = closure_transform(Xbase, Xword)
    Xclosure = transform * vcat(Xbase, Xword)
    Yclosure = transform * vcat(Ybase, Yword)
    closure_fit = checked_generator(Xclosure, Yclosure, model.lag)

    base_physical = Diagonal(model.base_scales) * base_fit.B *
                    Diagonal(1 ./ model.base_scales)
    # The first rows of the transform are the protected base coordinates.
    lifted_scaled = closure_fit.B[1:length(model.base_scales), :] * transform
    lifted_base_scaled = lifted_scaled[:, 1:length(model.base_scales)]
    closure_physical = Diagonal(model.base_scales) * lifted_base_scaled *
                       Diagonal(1 ./ model.base_scales)
    base_rates = model.extract(base_physical)
    closure_rates = model.extract(closure_physical)
    (; base_rates, closure_rates,
       base_condition=base_fit.condition,
       closure_condition=closure_fit.condition,
       closure_dimension=size(Xclosure, 1),
       candidate_words=length(keep),
       pair_count=size(Xbase, 2))
end

function models()
    lv = ClosureModel(
        "Lotka--Volterra", [[50, 100]],
        [1 -1 0; 0 1 -1], [1.0, 0.005, 0.6],
        u -> [u[1], u[1] * u[2], u[2]],
        u -> [1.0, u[1], u[2], u[1] * (u[1]-1) / 2,
              u[1] * u[2], u[2] * (u[2]-1) / 2],
        [1.0, 50.0, 100.0, 50.0^2, 50.0*100.0, 100.0^2],
        collect(0.0:0.1:(SMOKE ? 1.9 : 19.9)), 0.1*LAG_FACTOR,
        SMOKE ? 20 : 500,
        B -> [B[2,2], (-B[2,5] + B[3,5]) / 2, -B[3,3]])

    mm_initials = if DESIGN == "excited"
        [[50,0], [40,0], [30,0], [20,0], [10,0],
         [40,5], [25,5], [10,5]]
    else
        [[50,0]]
    end
    mm_dt = env_float("CLOSURE_BENCHMARK_MM_DT", DESIGN == "excited" ? 0.25 : 0.5)
    mm_tmax = env_float("CLOSURE_BENCHMARK_MM_TMAX", DESIGN == "excited" ? 49.75 : 199.5)
    mm_lag = env_float("CLOSURE_BENCHMARK_MM_LAG", 0.5 * LAG_FACTOR)
    mm_paths = env_int("CLOSURE_BENCHMARK_MM_PATHS",
                       SMOKE ? 20 : (DESIGN == "excited" ? 200 : 500))
    mm = ClosureModel(
        "Michaelis--Menten", mm_initials,
        [-1 1 0; 1 -1 -1], [0.01, 0.1, 0.1],
        u -> [u[1] * (10-u[2]), u[2], u[2]],
        u -> [1.0, u[1], u[2], u[1] * u[2]],
        [1.0, 50.0, 10.0, 500.0],
        collect(0.0:mm_dt:(SMOKE ? min(9.5, mm_tmax) : mm_tmax)), mm_lag,
        mm_paths,
        B -> begin
            k1 = mean((-B[2,2]/10, B[3,2]/10, B[2,4], -B[3,4]))
            k2 = B[2,3]
            [k1, k2, -B[3,3]-k2]
        end)

    bru = ClosureModel(
        "Brusselator", [[200, 600]],
        [1 1 -1 -1; 0 -1 1 0], [200.0, 2.5e-5, 3.0, 1.0],
        u -> [1.0, u[1]*(u[1]-1)*u[2]/2, u[1], u[1]],
        u -> [1.0, u[1], u[2], u[1]*(u[1]-1)/2, u[1]*u[2],
              u[2]*(u[2]-1)/2, u[1]*(u[1]-1)*(u[1]-2)/6,
              u[1]*(u[1]-1)*u[2]/2, u[1]*u[2]*(u[2]-1)/2,
              u[2]*(u[2]-1)*(u[2]-2)/6],
        [1.0, 200.0, 600.0, 200.0^2, 200.0*600.0, 600.0^2,
         200.0^3, 200.0^2*600.0, 200.0*600.0^2, 600.0^3],
        collect(0.0:0.1:(SMOKE ? 1.9 : 49.6)), 0.02*LAG_FACTOR,
        SMOKE ? 20 : 500,
        B -> [B[2,1], (B[2,8]-B[3,8])/2, B[3,2], -B[2,2]-B[3,2]])

    species = Dict("E"=>1, "K3"=>2, "K3p"=>3, "K2"=>4, "K2p"=>5,
                   "K2pp"=>6, "K1"=>7, "K1p"=>8, "K1pp"=>9)
    unit(name) = (v=zeros(Int,9); v[species[name]]=1; v)
    transfer(from,to) = unit(to) - unit(from)
    mapk_stoichiometry = hcat(
        transfer("K3","K3p"), transfer("K3p","K3"),
        transfer("K2","K2p"), transfer("K2p","K2pp"),
        transfer("K2p","K2"), transfer("K2pp","K2p"),
        transfer("K1","K1p"), transfer("K1p","K1pp"),
        transfer("K1p","K1"), transfer("K1pp","K1p"))
    mapk_initial(e,k3p,k2p,k2pp,k1p,k1pp) =
        [e, 40-k3p, k3p, 60-k2p-k2pp, k2p, k2pp,
         60-k1p-k1pp, k1p, k1pp]
    mapk_initials = if DESIGN == "excited"
        [mapk_initial(12,0,0,0,0,0), mapk_initial(4,0,0,0,0,0),
         mapk_initial(0,30,0,0,0,0), mapk_initial(12,20,0,0,0,0),
         mapk_initial(0,30,30,0,0,0), mapk_initial(0,30,20,20,0,0),
         mapk_initial(0,10,10,40,0,0), mapk_initial(0,0,0,40,0,0),
         mapk_initial(0,0,0,40,30,0), mapk_initial(0,0,0,40,20,20),
         mapk_initial(0,0,0,20,10,40), mapk_initial(12,10,10,5,10,5)]
    else
        [mapk_initial(12,0,0,0,0,0), mapk_initial(4,0,0,0,0,0),
         mapk_initial(0,30,0,0,0,0), mapk_initial(0,0,20,10,0,0),
         mapk_initial(0,0,0,0,25,15), mapk_initial(12,10,10,5,10,5)]
    end
    mapk_alpha = u -> [u[1]*u[2], u[3], u[3]*u[4], u[3]*u[5], u[5],
                       u[6], u[6]*u[7], u[6]*u[8], u[8], u[9]]
    mapk_feature = u -> [1.0, u[1], u[3], u[5], u[6], u[8], u[9],
                         u[1]*u[2], u[3]*u[4], u[3]*u[5],
                         u[6]*u[7], u[6]*u[8]]
    mapk_extract = B -> [B[3,8], -B[3,3], B[4,9],
                         (-B[4,10]+B[5,10])/2, -B[4,4],
                         (B[4,5]-B[5,5])/2, B[6,11],
                         (-B[6,12]+B[7,12])/2, -B[6,6],
                         (B[6,7]-B[7,7])/2]
    mapk_dt = env_float("CLOSURE_BENCHMARK_MAPK_DT", DESIGN == "excited" ? 0.1 : 0.2)
    mapk_tmax = env_float("CLOSURE_BENCHMARK_MAPK_TMAX", DESIGN == "excited" ? 4.9 : 9.8)
    mapk_lag = env_float("CLOSURE_BENCHMARK_MAPK_LAG", 0.2 * LAG_FACTOR)
    mapk_paths = env_int("CLOSURE_BENCHMARK_MAPK_PATHS",
                         SMOKE ? 10 : (DESIGN == "excited" ? 200 : 200))
    mapk = ClosureModel(
        "MAPK cascade", mapk_initials, mapk_stoichiometry,
        [0.025,0.30,0.020,0.020,0.25,0.25,0.015,0.015,0.20,0.20],
        mapk_alpha, mapk_feature,
        [1.0,12.0,40.0,60.0,60.0,60.0,60.0,
         12.0*40.0,40.0*60.0,40.0*60.0,60.0*60.0,60.0*60.0],
        collect(0.0:mapk_dt:(SMOKE ? min(1.8, mapk_tmax) : mapk_tmax)), mapk_lag,
        mapk_paths,
        mapk_extract)
    [lv, mm, bru, mapk]
end

function run_benchmark()
    mkpath(OUT)
    records = DataFrame(model=String[], seed=Int[], estimator=String[],
                        rate=Int[], estimate=Float64[], truth=Float64[],
                        relative_error=Float64[], dimension=Int[],
                        gram_condition=Float64[], pairs=Int[],
                        candidate_words=Int[])
    selected_models = isempty(MODEL_FILTER) ? models() :
        filter(model -> occursin(lowercase(MODEL_FILTER), lowercase(model.name)), models())
    isempty(selected_models) && error("no model matches CLOSURE_BENCHMARK_MODEL")
    for model in selected_models, seed in SEEDS
        @printf("%s seed %d\n", model.name, seed)
        result = fit_pair(model, seed)
        for (label, estimates, dimension, condition) in
            (("base", result.base_rates, length(model.base_scales), result.base_condition),
             ("depth two", result.closure_rates, result.closure_dimension,
              result.closure_condition))
            for rate in eachindex(model.rates)
                estimate = estimates[rate]
                truth = model.rates[rate]
                push!(records, (model.name, seed, label, rate, estimate, truth,
                                abs(estimate-truth)/truth, dimension, condition,
                                result.pair_count, result.candidate_words))
            end
        end
    end
    CSV.write(joinpath(OUT, "estimates.csv"), records)
    per_seed = combine(groupby(records, [:model,:seed,:estimator]),
                       :relative_error => mean => :mean_relative_error,
                       :dimension => first => :dimension,
                       :gram_condition => first => :gram_condition,
                       :pairs => first => :pairs,
                       :candidate_words => first => :candidate_words)
    CSV.write(joinpath(OUT, "per_seed.csv"), per_seed)
    summary = combine(groupby(per_seed, [:model,:estimator]),
                      :mean_relative_error => median => :median_error,
                      :mean_relative_error => (x -> quantile(x,0.25)) => :q25_error,
                      :mean_relative_error => (x -> quantile(x,0.75)) => :q75_error,
                      :dimension => first => :dimension,
                      :gram_condition => median => :median_condition,
                      :pairs => first => :pairs,
                      :candidate_words => first => :candidate_words)
    CSV.write(joinpath(OUT, "summary.csv"), summary)
    show(stdout, MIME("text/plain"), summary); println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end

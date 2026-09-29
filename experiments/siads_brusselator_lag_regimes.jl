# Reuses the controlled SSA and the recovered original full-rank SVD kernel.
include("siads_brusselator_log_diagnostics.jl")
include("legacy_mezic/svd_kernel.jl")
include(joinpath(@__DIR__, "..", "src", "KoopmanCMEPaperTools.jl"))
using Serialization
using .KoopmanCMEPaperTools
BLAS.set_num_threads(1)
const LAGS=[.0005,.001,.002,.005,.01,.02,.05,.1,.2,.4,.8,1.6,3.2,6.4]
const TRUTH=[200.,2.5e-5,3.,1.]
const OUT=joinpath(@__DIR__,"../results/siads_lag_regimes")
const BRU_STOICHIOMETRY=[1. 1. -1. -1.; 0. -1. 1. 0.]
const BRU_PROPENSITY_COLUMNS=[1,8,2,2]
function rate_weights(d=10)
    W=[zeros(d,d) for _=1:4]
    W[1][2,1]=200
    W[2][2,8]=scales[2]/scales[8]/2; W[2][3,8]=-scales[3]/scales[8]/2
    W[3][3,2]=3; W[4][2,2]=-1; W[4][3,2]=-3
    W
end
function extract_rates(B)
    # The cached features are diagonally scaled.  Return to physical factorial
    # coordinates before solving the shared-propensity stoichiometric system.
    physical=Diagonal(scales)*B*Diagonal(1 ./ scales)
    extract_stoichiometric_rates(
        physical,[2,3],BRU_PROPENSITY_COLUMNS,BRU_STOICHIOMETRY;
        nonnegative=true).rates
end
function regime_fit(a,b,h; derivative=false)
    # Synthetic snapshots retain exactly the empirical Gram pair, permitting
    # use of the original SVD kernel without materializing all trajectory columns.
    X=Matrix(cholesky(Symmetric(a)).L); Y=b/transpose(X)
    K=svd_dmd_K(X,Y,10)
    norm(K-b/a)/max(1,norm(K))<1e-7 || error("SVD/Gram mismatch")
    ev=eigvals(K)
    any(z->real(z)<=0 && abs(imag(z))<1e-10,ev) && throw(DomainError(h,"log cut"))
    B=log(complex.(K))/h
    norm(imag.(B))<1e-7*max(1,norm(B)) || throw(DomainError(h,"complex log"))
    W=rate_weights()
    rates=[sum(w.*real.(B)) for w in W]
    H=derivative ? [real.(log(complex.([transpose(K) w; zeros(10,10) transpose(K)]))[1:10,11:20])/h for w in W] : Matrix{Float64}[]
    conditions=[norm(Diagonal(1 ./ scales)*v*Diagonal(scales)) for v in H]
    (;K,rates,H,conditions)
end

"""Fit the ordinary, defect-corrected, and Richardson generators at a matched lag pair."""
function regime_defect_fit(a,bhalf,b,h)
    result=fit_defect_corrected_edmd(a,bhalf,b,h)
    rates=(raw=extract_rates(result.raw_generator),
           defect=extract_rates(result.corrected_generator),
           richardson=extract_rates(result.richardson_generator))
    (;result,rates)
end
function collect_grams(n,seed; groups=n)
    starts=collect(0:.1:49.6)
    ts=sort(unique(round.(vcat(starts,[starts .+ h for h in LAGS]...),digits=7)))
    ix=searchsortedfirst.(Ref(ts),round.(starts,digits=7))
    iy=[searchsortedfirst.(Ref(ts),round.(starts .+ h,digits=7)) for h in LAGS]
    a=zeros(10,10,groups); b=zeros(10,10,groups,length(LAGS)); rng=MersenneTwister(seed)
    for i=1:n
        z=trajectory(rng;times=ts); x=z[:,ix]; group=mod1(i,groups)
        a[:,:,group] .+= x*transpose(x)/length(starts)
        for l=eachindex(LAGS); b[:,:,group,l] .+= z[:,iy[l]]*transpose(x)/length(starts); end
        i%1000==0 && println("SSA seed=$seed trajectories=$i/$n")
    end
    a./(n/groups),b./(n/groups)
end
function run_regimes()
    mkpath(OUT); n=parse(Int,get(ENV,"SIADS_N","1000")); nr=parse(Int,get(ENV,"SIADS_REF","10000")); nb=100
    cache=joinpath(OUT,"grams_$(n)_$(nr).jls")
    if isfile(cache)
        a,b,ar,br=deserialize(cache)
    else
        a,b=collect_grams(n,714); ar,br=collect_grams(nr,2718;groups=20)
        serialize(cache,(a,b,ar,br))
    end
    am=dropdims(mean(a,dims=3),dims=3); arm=dropdims(mean(ar,dims=3),dims=3)
    rng=MersenneTwister(881); ids=[rand(rng,1:n,n) for _=1:nb] # paired across all lags
    open(joinpath(OUT,"summary.csv"),"w") do io
        println(io,"lag,rate,estimate,truth,bootstrap_sd,linearized_sd,relative_rmse_bootstrap,condition_physical,reference,reference_se,bootstrap_valid,bootstrap_total,reference_valid")
        for l=eachindex(LAGS)
            h=LAGS[l]; bm=dropdims(mean(b[:,:,:,l],dims=3),dims=3); brm=dropdims(mean(br[:,:,:,l],dims=3),dims=3)
            ref=try regime_fit(am,bm,h;derivative=true) catch err; err isa DomainError || rethrow(); nothing end
            pop=try regime_fit(arm,brm,h) catch err; err isa DomainError || rethrow(); nothing end
            samples=Vector{Float64}[]; predicted=Vector{Float64}[]
            for id=ids
                aa=dropdims(mean(a[:,:,id],dims=3),dims=3); bb=dropdims(mean(b[:,:,id,l],dims=3),dims=3)
                f=try regime_fit(aa,bb,h) catch err; err isa DomainError || rethrow(); nothing end
                f===nothing && continue
                push!(samples,f.rates)
                if ref!==nothing
                    E=(bb-bm-ref.K*(aa-am))/am
                    push!(predicted,[sum(H.*E) for H in ref.H])
                end
            end
            # Independent reference groups give a delta-method Monte Carlo SE.
            pr=pop===nothing ? nothing : regime_fit(arm,brm,h;derivative=true)
            rse=fill(NaN,4)
            if pr!==nothing
                vals=hcat([[sum(H.*((br[:,:,g,l]-brm-pr.K*(ar[:,:,g]-arm))/arm)) for H in pr.H] for g=1:20]...)
                rse=vec(std(vals,dims=2))/sqrt(20)
            end
            for k=1:4
                sd=length(samples)>1 ? std([s[k] for s in samples]) : NaN
                psd=length(predicted)>1 ? std([s[k] for s in predicted]) : NaN
                rmse=isempty(samples) ? NaN : sqrt(mean([(s[k]-TRUTH[k])^2 for s in samples]))/TRUTH[k]
                row=(h,"k$k",ref===nothing ? NaN : ref.rates[k],TRUTH[k],sd,psd,rmse,ref===nothing ? NaN : ref.conditions[k],pop===nothing ? NaN : pop.rates[k],rse[k],length(samples),nb,pop!==nothing)
                println(io,join(row,","))
            end
            flush(io); println("lag=$h valid=$(length(samples))/$nb rates=$(ref===nothing ? "invalid" : ref.rates)")
        end
    end
end

"""
Evaluate the new estimator only on lag pairs present in the original cached
experiment.  Every pair shares the same current-state Gram matrix, so its
semigroup defect has the projection interpretation required by the theory.
"""
function run_defect_correction()
    n=parse(Int,get(ENV,"SIADS_N","1000")); nr=parse(Int,get(ENV,"SIADS_REF","10000")); nb=100
    cache=joinpath(OUT,"grams_$(n)_$(nr).jls")
    isfile(cache) || error("run run_regimes() first to create $cache")
    a,b,ar,br=deserialize(cache)
    am=dropdims(mean(a,dims=3),dims=3); arm=dropdims(mean(ar,dims=3),dims=3)
    pairs=Tuple{Int,Int}[]
    for coarse in eachindex(LAGS)
        half=findfirst(x->isapprox(x,LAGS[coarse]/2;rtol=1e-12,atol=0),LAGS)
        half===nothing || push!(pairs,(coarse,half))
    end
    rng=MersenneTwister(1441); ids=[rand(rng,1:n,n) for _=1:nb]
    methods=(:raw,:defect,:richardson)
    open(joinpath(OUT,"defect_corrected.csv"),"w") do io
        println(io,"lag,rate,method,estimate,truth,bootstrap_sd,reference,reference_se,bootstrap_valid,bootstrap_total,reference_valid")
        for (coarse,half) in pairs
            h=LAGS[coarse]
            bh=dropdims(mean(b[:,:,:,half],dims=3),dims=3)
            bc=dropdims(mean(b[:,:,:,coarse],dims=3),dims=3)
            brh=dropdims(mean(br[:,:,:,half],dims=3),dims=3)
            brc=dropdims(mean(br[:,:,:,coarse],dims=3),dims=3)
            central=try regime_defect_fit(am,bh,bc,h) catch err; err isa DomainError || rethrow(); nothing end
            reference=try regime_defect_fit(arm,brh,brc,h) catch err; err isa DomainError || rethrow(); nothing end
            samples=Dict(method=>Vector{Vector{Float64}}() for method in methods)
            for id in ids
                aa=dropdims(mean(a[:,:,id],dims=3),dims=3)
                bhalf=dropdims(mean(b[:,:,id,half],dims=3),dims=3)
                bcoarse=dropdims(mean(b[:,:,id,coarse],dims=3),dims=3)
                fitted=try regime_defect_fit(aa,bhalf,bcoarse,h) catch err; err isa DomainError || rethrow(); nothing end
                fitted===nothing && continue
                for method in methods; push!(samples[method],getproperty(fitted.rates,method)); end
            end
            group_values=Dict(method=>Vector{Vector{Float64}}() for method in methods)
            if reference!==nothing
                for group=1:size(ar,3)
                    fitted=try regime_defect_fit(ar[:,:,group],br[:,:,group,half],br[:,:,group,coarse],h) catch err; err isa DomainError || rethrow(); nothing end
                    fitted===nothing && continue
                    for method in methods; push!(group_values[method],getproperty(fitted.rates,method)); end
                end
            end
            for method in methods, k=1:4
                values=samples[method]
                sd=length(values)>1 ? std([value[k] for value in values]) : NaN
                groups=group_values[method]
                reference_se=length(groups)>1 ? std([value[k] for value in groups])/sqrt(length(groups)) : NaN
                estimate=central===nothing ? NaN : getproperty(central.rates,method)[k]
                reference_value=reference===nothing ? NaN : getproperty(reference.rates,method)[k]
                row=(h,"k$k",String(method),estimate,TRUTH[k],sd,reference_value,
                     reference_se,length(values),nb,reference!==nothing)
                println(io,join(row,","))
            end
            flush(io)
            println("defect correction lag=$h valid=$(length(samples[:defect]))/$nb")
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    run_regimes()
    run_defect_correction()
end

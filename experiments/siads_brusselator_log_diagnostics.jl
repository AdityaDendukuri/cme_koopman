# Controlled finite-lag EDMD diagnostics; stdlib only.
using LinearAlgebra, Random, Statistics, Printf
const scales = [1.,200.,600.,200.0^2,200.0*600.,600.0^2,200.0^3,200.0^2*600.,200.0*600.0^2,600.0^3]
features(x,y) = [1.,x,y,x*(x-1)/2,x*y,y*(y-1)/2,x*(x-1)*(x-2)/6,x*(x-1)*y/2,x*y*(y-1)/2,y*(y-1)*(y-2)/6] ./ scales
function trajectory(rng; times=collect(0:.05:50))
    out=zeros(10,length(times)); x=200; y=600; t=0.; next=1
    while next<=length(times)
        a=(200.,2.5e-5*x*(x-1)*y/2,3.0*x,1.0*x)
        total=sum(a); event=t+randexp(rng)/total
        while next<=length(times) && times[next] < event
            out[:,next]=features(x,y); next+=1
        end
        next>length(times) && break
        u=rand(rng)*total
        if u<a[1]; x+=1
        elseif u<a[1]+a[2]; x+=1; y-=1
        elseif u<a[1]+a[2]+a[3]; x-=1; y+=1
        else; x-=1
        end
        t=event
    end
    out
end
function fit(G0,G1,h,affine)
    if affine
        C0=G0; C1=G1; ss=scales; ix=2; iy=3
    else
        C0=G0[2:end,2:end]-G0[2:end,1]*transpose(G0[2:end,1])
        C1=G1[2:end,2:end]-G1[2:end,1]*transpose(G0[2:end,1])
        ss=scales[2:end]; ix=1; iy=2
    end
    K=C1/C0; vals=eigvals(K)
    any(z->real(z)<=0 && abs(imag(z))<1e-10,vals) && error("principal-log domain failure")
    B=log(complex.(K))/h
    norm(imag.(B))<1e-7*max(1,norm(B)) || error("nonreal generator")
    B=real.(B); d=size(K,1)
    W3=zeros(d,d); W3[iy,ix]=ss[iy]/ss[ix]
    W4=zeros(d,d); W4[ix,ix]=-1; W4[iy,ix]=-ss[iy]/ss[ix]
    Ws=[W3,W4]
    Hs=[real.(log(complex.([transpose(K) W; zeros(d,d) transpose(K)]))[1:d,d+1:2d])/h for W in Ws]
    rates=[sum(W.*B) for W in Ws]
    # Conditions in declared physical feature coordinates, not scaled coordinates.
    conds=[norm(Diagonal(1 ./ ss)*H*Diagonal(ss)) for H in Hs]
    (;K,C0,C1,rates,conds,Hs)
end
function main()
    n=parse(Int,get(ENV,"SIADS_N","1000")); nb=parse(Int,get(ENV,"SIADS_BOOT","100"))
    rng=MersenneTwister(714); lags=[.05,.1,.2,.4]; steps=[1,2,4,8]
    G0=zeros(10,10,n); G1=zeros(10,10,n,4)
    starts=collect(1:2:993) # same current-time mixture at every lag, 0:0.1:49.6
    for i=1:n
        Z=trajectory(rng); X=Z[:,starts]
        G0[:,:,i]=X*transpose(X)/length(starts)
        for l=1:4; G1[:,:,i,l]=Z[:,starts .+ steps[l]]*transpose(X)/length(starts); end
    end
    mkpath("results/siads_log_diagnostics")
    open("results/siads_log_diagnostics/summary.csv","w") do io
        println(io,"n,bootstrap,lag,affine,rate,estimate,truth,condition_physical,bootstrap_sd,linearized_sd,linearization_rmse,bootstrap_failures")
        for l=1:4, affine in (true,false)
            h=lags[l]; a=dropdims(mean(G0,dims=3),dims=3); b=dropdims(mean(G1[:,:,:,l],dims=3),dims=3)
            ref=fit(a,b,h,affine); actual=Vector{Float64}[]; pred=Vector{Float64}[]; failures=0
            for rep=1:nb
                ids=rand(rng,1:n,n)
                aa=dropdims(mean(G0[:,:,ids],dims=3),dims=3); bb=dropdims(mean(G1[:,:,ids,l],dims=3),dims=3)
                try
                    f=fit(aa,bb,h,affine)
                    E=(f.C1-ref.C1-ref.K*(f.C0-ref.C0))/ref.C0
                    push!(actual,f.rates-ref.rates); push!(pred,[sum(H.*E) for H in ref.Hs])
                catch err
                    err isa InterruptException && rethrow(); failures+=1
                end
            end
            length(actual)>1 || error("too few valid bootstrap fits")
            A=hcat(actual...); P=hcat(pred...)
            for k=1:2
                row=(n,nb,h,affine,k==1 ? "k3" : "k4",ref.rates[k],k==1 ? 3. : 1.,ref.conds[k],std(A[k,:]),std(P[k,:]),sqrt(mean((A[k,:]-P[k,:]).^2)),failures)
                println(io,join(row,",")); println(join(row,","))
            end
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

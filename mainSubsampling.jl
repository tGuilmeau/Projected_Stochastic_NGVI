using LinearAlgebra
using Distributions
using DelimitedFiles
using LaTeXStrings
using CSV
using DataFrames
using Plots; pyplot()

include("methodGaussian.jl")




###------- Abstract target -------###


abstract type AbstractTarget end

function estimatorTarget(target::AbstractTarget, mean, cov, N) end

function evidenceLowerBound(target::AbstractTarget, mean, cov, N_ELBO) end


###------- Gaussian target with subsampling estimator -------###

mutable struct GaussianTargetSubSampling <: AbstractTarget
    name::String #Name of the target - to write the results
    dimension::Int64 #Dimension of the mean
    mean::Array{Float64,1} #Mean of the target
    cov::Array{Float64,2} #covariance of the target
    prec::Array{Float64,2} #inverse of the covariance - stored directly for convenience
    distr::FullNormal #distribution associated to the target
    logdetCov::Float64 #logdet of the target covariance
    isGaussian::Bool
    covPrior::Array{Float64}
    sigmaSqLk::Float64
    Z
    y
    M
end



function turbineTarget(name, covPrior, sigmaSqLk)
    df2015 = CSV.read("./gas+turbine+co+and+nox+emission+data+set/gt_2015.csv", DataFrame)
    df2014 = CSV.read("./gas+turbine+co+and+nox+emission+data+set/gt_2014.csv", DataFrame)
    df2013 = CSV.read("./gas+turbine+co+and+nox+emission+data+set/gt_2013.csv", DataFrame)
    df2012 = CSV.read("./gas+turbine+co+and+nox+emission+data+set/gt_2012.csv", DataFrame)
    df2011 = CSV.read("./gas+turbine+co+and+nox+emission+data+set/gt_2011.csv", DataFrame)

    df = vcat(df2015, df2014, df2013, df2012, df2011, cols=:union)

    Z = Matrix(df[:,1:9])
    y = df[:,:NOX] #predict NOX level from ambient and turbine info

    M = size(df, 1)
    d = 9

    theta_pi_1 = zeros(d)
    theta_pi_2 = zeros(d,d)

    for m in 1:M
        theta_pi_1 += y[m]*Z[m,:]
        theta_pi_2 += Z[m,:]*transpose(Z[m,:])
    end

    

    theta_pi_1 = (1/sigmaSqLk)*theta_pi_1
    theta_pi_2 = -0.5*inv(covPrior) - 0.5*(1/sigmaSqLk)*theta_pi_2


    invTheta_pi_2 = inv(theta_pi_2)
    invTheta_pi_2 = 0.5*(invTheta_pi_2 + transpose(invTheta_pi_2))

    covTarget = -0.5*invTheta_pi_2
    meanTarget = covTarget*theta_pi_1

    return GaussianTargetSubSampling(name, d, meanTarget, covTarget, -2.0*theta_pi_2, MvNormal(meanTarget,covTarget), log(det(covTarget)), true, covPrior, sigmaSqLk, Z,y,M)

end



function estimatorTarget(target::GaussianTargetSubSampling, mean, cov, N)
    d = target.dimension

    theta_1 = zeros(d)
    theta_2 = zeros(d,d)

    for n in 1:N
        n = rand(1:target.M)
        theta_1 += target.y[n] * target.Z[n,:]
        theta_2 += target.Z[n,:]*transpose(target.Z[n,:])
    end

    theta_1 = (target.M / N)*(1/target.sigmaSqLk)*theta_1
    theta_2 = -0.5*(inv(target.covPrior) + (target.M / N)*(1/target.sigmaSqLk)*theta_2)


    return theta_1, theta_2

end



function evidenceLowerBound(target::GaussianTargetSubSampling, mean, cov, N_ELBO)
    
    q = MvNormal(mean, cov)
    d = target.dimension
    pop = Array{Float64}(undef, d, N_ELBO)
    rand!(q,pop)
    
    return (1/N_ELBO) * sum([logpdf(target.distr,pop[:,j]) for j in 1:N_ELBO])
end


function targetKL(target::GaussianTargetSubSampling, mean, cov)
    prec=inv(cov)
    return 0.5* ( log( det(cov)) - target.logdetCov - target.dimension + transpose(mean - target.mean)*prec*(mean - target.mean) + tr(prec*target.cov) )
end




covPrior = Diagonal([5.0 for i in 1:9])
sigmaSqLk = 1.0
d = 9
targetName = "turbine"
target = turbineTarget(targetName, covPrior, sigmaSqLk)




eta = 0.001
N=1000
K=20000
proj=false
alpha=0.1
beta=100.0
N_ELBO = 100
nbRuns = 100





###########
# Running #
###########





# ###################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "decreasing"
# sampleSizeSchedule = "increasing"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([10.0 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

#     global avgKLtoTarget += KLtoTarget
#     global avgELBO += ELBO
#     global avgComputationEffort += computationEffort

# end

# avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
# avgELBO = (1/nbRuns)*avgELBO
# avgComputationEffort = (1/nbRuns)*avgComputationEffort

# directory = "./"*target.name*"/"*methodName
# mkpath(directory)
# writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
# writedlm(directory*"/ELBO.txt", avgELBO)
# writedlm(directory*"/computationEffort.txt", avgComputationEffort)




# ###################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "constant"
# sampleSizeSchedule = "increasing"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([10.0 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

#     global avgKLtoTarget += KLtoTarget
#     global avgELBO += ELBO
#     global avgComputationEffort += computationEffort

# end

# avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
# avgELBO = (1/nbRuns)*avgELBO
# avgComputationEffort = (1/nbRuns)*avgComputationEffort

# directory = "./"*target.name*"/"*methodName
# mkpath(directory)
# writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
# writedlm(directory*"/ELBO.txt", avgELBO)
# writedlm(directory*"/computationEffort.txt", avgComputationEffort)


###################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "decreasing"
# sampleSizeSchedule = "constant"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([10.0 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

#     global avgKLtoTarget += KLtoTarget
#     global avgELBO += ELBO
#     global avgComputationEffort += computationEffort

# end

# avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
# avgELBO = (1/nbRuns)*avgELBO
# avgComputationEffort = (1/nbRuns)*avgComputationEffort

# directory = "./"*target.name*"/"*methodName
# mkpath(directory)
# writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
# writedlm(directory*"/ELBO.txt", avgELBO)
# writedlm(directory*"/computationEffort.txt", avgComputationEffort)


# ###################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "constant"
# sampleSizeSchedule = "constant"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([10.0 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

#     global avgKLtoTarget += KLtoTarget
#     global avgELBO += ELBO
#     global avgComputationEffort += computationEffort

# end

# avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
# avgELBO = (1/nbRuns)*avgELBO
# avgComputationEffort = (1/nbRuns)*avgComputationEffort

# directory = "./"*target.name*"/"*methodName
# mkpath(directory)
# writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
# writedlm(directory*"/ELBO.txt", avgELBO)
# writedlm(directory*"/computationEffort.txt", avgComputationEffort)





############
# Plotting #
############




# p_computational = plot(yscale=:log, xlims=(0,K*N),xlabel="number of cumulated data points",ylabel="\$d_{A^*}(\\omega_*,\\omega_t)\$",xticks=[10000000,20000000],labelfontsize = 17, tickfontsize = 17, legendfontsize = 17)
# p_iterative = plot(yscale=:log,xlabel="\$ t \$",ylabel="\$d_{A^*}(\\omega_*,\\omega_t)\$", xticks=[10000,20000],labelfontsize = 17, tickfontsize = 17, legendfontsize = 17)

# directory = "./"*targetName

# methodName = "constanteta_increasingN"
# methodLabel = "\$ \\eta_t = 10^{-3},\\,N_t=t+1\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)

# methodName = "decreasingeta_increasingN"
# methodLabel = "\$ \\eta_t = 2/(t+2),\\,N_t=t+1\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)

# methodName = "constanteta_constantN"
# methodLabel = "\$ \\eta_t = 10^{-3},\\,N_t= 10^3\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)

# methodName = "decreasingeta_constantN"
# methodLabel = "\$ \\eta_t = 2/(t+2),\\,N_t= 10^3\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)


# savefig(p_computational, directory*"/"*targetName*"KLtoTargetComputational.pdf")
# savefig(p_iterative, directory*"/"*targetName*"KLtoTargetIteration.pdf")



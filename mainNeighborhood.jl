using LinearAlgebra
using Distributions
using DelimitedFiles
using LaTeXStrings
using Plots; pyplot()

include("methodGaussian.jl")



function conditionMatrix(d::Int64, kappa::Float64)
    #Constructs a matrix d*d whose condition number is kappa

    y = rand(Uniform(-1.0,1.0),d)
    Y = I - (2/(norm(y)^2))*y*transpose(y)
    D = Diagonal([exp( ((i-1)/(d-1))*log(kappa)) for i in 1:d])
    
    M = Y*D*Y
    
    return 0.5*(M + transpose(M))
end



function conditionMatrixStep(d::Int64, kappa::Float64)
    #Constructs a matrix d*d whose condition number is kappa

    
    y = rand(Uniform(-1.0,1.0),d)
    Y = I - (2/(norm(y)^2))*y*transpose(y)
    u = Array{Float64}(undef,d)
    for i in 1:d
        if i < d/2
            u[i] = 1.0
        else
            u[i] = kappa
        end
    end
    D = Diagonal(u)
    
    M = Y*D*Y
    
    return 0.5*(M + transpose(M))
end






###------- Abstract target -------###


abstract type AbstractTarget end

function estimatorTarget(target::AbstractTarget, mean, cov, N) end

function evidenceLowerBound(target::AbstractTarget, mean, cov, N_ELBO) end






###------- Gaussian target with Bonnet and Price estimator -------###

mutable struct GaussianTargetBP <: AbstractTarget
    name::String #Name of the target - to write the results
    dimension::Int64 #Dimension of the mean
    mean::Array{Float64,1} #Mean of the target
    cov::Array{Float64,2} #covariance of the target
    prec::Array{Float64,2} #inverse of the covariance - stored directly for convenience
    distr::FullNormal #distribution associated to the target
    logdetCov::Float64 #logdet of the target covariance
    isGaussian::Bool
end

function constructGBPTarget(name, mean, cov)
    # constructor that uses only the mean and covariance
    return GaussianTargetBP(name, length(mean), mean, cov, inv(cov), MvNormal(mean,cov), log(det(cov)), true)
end


function estimatorTarget(target::GaussianTargetBP, mean, cov, N)
    d = target.dimension

    q = MvNormal(mean, cov)
    pop = Array{Float64}(undef, d, N)
    rand!(q,pop)

    return target.prec*(target.mean + mean - (1/N)*sum([pop[:,j] for j in 1:N])), -0.5*target.prec
end



function evidenceLowerBound(target::GaussianTargetBP, mean, cov, N_ELBO)
    
    q = MvNormal(mean, cov)
    d = target.dimension
    pop = Array{Float64}(undef, d, N_ELBO)
    rand!(q,pop)
    
    return (1/N_ELBO) * sum([logpdf(target.distr,pop[:,j]) for j in 1:N_ELBO])
end


function targetKL(target::GaussianTargetBP, mean, cov)
    prec=inv(cov)
    return 0.5* ( log( det(cov)) - target.logdetCov - target.dimension + transpose(mean - target.mean)*prec*(mean - target.mean) + tr(prec*target.cov) )
end







targetName = "neighborhoodSizeN_d10_kappa100_expEigen"

d=10
meanTarget = rand(Uniform(-5.0, 5.0),d)
covTarget = conditionMatrix(d, 100.0)
precTarget = inv(covTarget)
directory = "./"*targetName
mkpath(directory)

target = constructGBPTarget(targetName, meanTarget,covTarget)

N = 100
K=5000
proj=false
alpha=0.1
beta=100.0
N_ELBO = 10
nbRuns = 100




###########
# Running #
###########

############################# different values of eta

# ###################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# N=100
# eta = 0.1
# stepSizeSchedule = "constant"
# sampleSizeSchedule = "constant"
# methodName = "constant_eta_N_eta01"
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

# N=100
# eta = 0.05
# stepSizeSchedule = "constant"
# sampleSizeSchedule = "constant"
# methodName = "constant_eta_N_eta005"
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

# N=100
# eta = 0.01
# stepSizeSchedule = "constant"
# sampleSizeSchedule = "constant"
# methodName = "constant_eta_N_eta001"
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

# N=100
# eta = 0.005
# stepSizeSchedule = "constant"
# sampleSizeSchedule = "constant"
# methodName = "constant_eta_N_eta0005"
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




############################# different values of eta

# p_computational = plot(yscale=:log, xlims=(0,K*N),xlabel="number of cumulated samples",ylabel="\$d_{A^*}(\\omega_*,\\omega_t)\$",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13)
# p_iterative = plot(yscale=:log,xlabel="\$ t \$",ylabel="\$d_{A^*}(\\omega_*,\\omega_t)\$",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13)

# directory = "./"*targetName




# methodName ="constant_eta_N_eta01"
# methodLabel = "\$ \\eta = 10^{-1},\\,N=10^2\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)


# methodName ="constant_eta_N_eta005"
# methodLabel = "\$ \\eta = 5 \\times 10^{-2},\\,N=10^2\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)


# methodName ="constant_eta_N_eta001"
# methodLabel = "\$ \\eta = 10^{-2},\\,N=10^2\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)


# methodName ="constant_eta_N_eta0005"
# methodLabel = "\$ \\eta = 5 \\times 10^{-3},\\,N=10^2\$"

# KLtoTarget = readdlm(directory*"/"*methodName*"/KLtoTarget.txt")
# computationEffort = readdlm(directory*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, KLtoTarget, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, KLtoTarget, label=methodLabel, linewidth=2)



# savefig(p_computational, directory*"/"*targetName*"KLtoTargetComputational.pdf")
# savefig(p_iterative, directory*"/"*targetName*"KLtoTargetIteration.pdf")

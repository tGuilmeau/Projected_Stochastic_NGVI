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


###------- Logistic regression posterior with Bonnet and Price estimator -------###

mutable struct logRegressionTarget <: AbstractTarget
    name::String #Name of the target - to write the results
    dimension::Int64 #Dimension of the mean
    isGaussian::Bool
    sigmaSqPrior::Float64
    prior
    Z
    y
    M
end


function logistic(s)
    return 1.0 / (1.0 + exp(-s))
end





d=5
global x_data = [5.0 for i in 1:d]

function syntheticTarget(name, sigmaPrior, d, M)

    prior = MvNormal(zeros(d), sigmaSqPrior)

    # x = rand(prior)
    # println(x)

    Z = Array{Float64}(undef, M, d)
    for m in 1:M
        Z[m,:] = (rand(d)-0.5*ones(d))*10.0
    end

    y = Array{Float64}(undef, M)
    for m in 1:M
        y[m] = rand(Binomial(1,logistic(transpose(x_data)*Z[m,:])))
    end

    return logRegressionTarget(name, d, false, sigmaSqPrior, prior, Z, y, M)
end




function estimatorTarget(target::logRegressionTarget, mean, cov, N)


    q = MvNormal(mean, cov)
    d = target.dimension
    pop = Array{Float64}(undef, d, N)
    rand!(q,pop)

    g_1= (1/target.sigmaSqPrior)*(mean - (1/N)*sum([pop[:,n] for n in 1:N]))
    g_2 = -1/(2*target.sigmaSqPrior)*I

    vectorData = zeros(d)
    matrixData = zeros(d,d)

    for n in 1:N
        for m in 1:target.M

            z = target.Z[m,:]
            prob = logistic(transpose(z)*pop[:,n])

            vectorData = vectorData + (target.y[m] - prob)*z
            matrixData = matrixData + prob*(1-prob)*z*transpose(z)

        end
    end

    vectorData = (1/N)*vectorData
    matrixData = (1/N)*matrixData

    return g_1 + vectorData + matrixData*mean, g_2 - 0.5*matrixData

end



function estimatorTargetDiag(target::logRegressionTarget, mean, cov, N)
    # Bonnet and Price estimator adapted to the case of variational family being the diagonal Gaussians


    q = MvNormal(mean, cov)
    d = target.dimension
    pop = Array{Float64}(undef, d, N)
    rand!(q,pop)

    g_1= (1/target.sigmaSqPrior)*(mean - (1/N)*sum([pop[:,n] for n in 1:N]))
    g_2 = -1/(2*target.sigmaSqPrior)*ones(d)

    meanData = zeros(d)
    diagCovData = zeros(d)

    for n in 1:N
        for m in 1:target.M

            z = target.Z[m,:]
            prob = logistic(transpose(z)*pop[:,n])

            meanData = meanData + (target.y[m] - prob)*z
            diagCovData = diagCovData + prob*(1-prob)*(z .* z)

        end
    end

    meanData = (1/N)*meanData
    diagCovData = (1/N)*diagCovData

    return g_1 + meanData + Diagonal(diagCovData)*mean, g_2 - 0.5*diagCovData

end


function logpdfTarg(target::logRegressionTarget, x)


    res = 0.0

    for m in 1:target.M

        z = target.Z[m,:]
        prob = logistic(transpose(z)*x)

        # res = res + target.y[m]*transpose(x)*z - log(1.0 + exp(transpose(x)*z))
        res = res + target.y[m]*transpose(x)*z - abs(transpose(x)*z) - log(exp(-abs(transpose(x)*z))+exp(transpose(x)*z-abs(transpose(x)*z)) )
    end

    return res + logpdf(target.prior, x)


end


function evidenceLowerBound(target::logRegressionTarget, mean, cov, N_ELBO)
    
    q = MvNormal(mean, cov)
    d = target.dimension
    pop = Array{Float64}(undef, d, N_ELBO)
    rand!(q,pop)
    
    return (1/N_ELBO) * sum([logpdfTarg(target,pop[:,j]) for j in 1:N_ELBO])
end






sigmaSqPrior = 1.0
targetName = "synthetic"
target = syntheticTarget(targetName, sigmaSqPrior, d, 100)






eta = 0.01
N=100
K=1000
proj=true
alpha=0.1
beta=100.0
N_ELBO = 100
nbRuns = 50

meanInit = rand(Uniform(-5.0, 5.0),d)
covInit = Array(Diagonal([0.5 for i in 1:d]))


#############################################
# Runs using Gaussians with full covariance #
#############################################



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


# ###################
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




#################################
# Runs using diagonal Gaussians #
#################################


# #################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "constant"
# sampleSizeSchedule = "constant"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_noProj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, false, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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


# #################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "constant"
# sampleSizeSchedule = "constant"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_proj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, true, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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







# ##################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "decreasing"
# sampleSizeSchedule = "constant"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_noProj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, false, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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


# ##################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "decreasing"
# sampleSizeSchedule = "constant"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_proj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, true, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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








# ##################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "constant"
# sampleSizeSchedule = "increasing"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_noProj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, false, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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


# ##################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "constant"
# sampleSizeSchedule = "increasing"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_proj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, true, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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






# ##################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "decreasing"
# sampleSizeSchedule = "increasing"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_noProj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, false, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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


# ##################
# avgKLtoTarget = zeros(K+1)
# avgELBO = zeros(K+1)
# avgComputationEffort = zeros(K+1)

# stepSizeSchedule = "decreasing"
# sampleSizeSchedule = "increasing"
# methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N"*"_proj"
# println(methodName)


# for i in 1:nbRuns

#     meanInit = rand(Uniform(-5.0, 5.0),d)
#     covInit = Array(Diagonal([0.5 for i in 1:d]))
#     KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, true, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

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



####################################################
# Plotting: comparing projection and no projection #
####################################################



# # constant eta and constant N

# p_computational = plot(ylims=(-100,0), xlims=(0,K*N),xlabel="number of cumulated data points",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13)
# p_iterative = plot(ylims=(-100,0), xlims=(0,K/2),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)


# methodName = "constanteta_constantN_proj"
# methodLabel = "projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)


# methodName = "constanteta_constantN_noProj"
# methodLabel = "no projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)

# savefig(p_computational, targetName*"/constantEta_constantN_ELBO_Computational.pdf")
# savefig(p_iterative, targetName*"/constantEta_constantN_ELBO_Iteration.pdf")


# # constant eta and increasingN

# p_computational = plot(ylims=(-100,0), xlims=(0,K*N),xlabel="number of cumulated data points",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13)
# p_iterative = plot(ylims=(-100,0), xlims=(0,K/2),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)


# methodName = "constanteta_increasingN_proj"
# methodLabel = "projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)


# methodName = "constanteta_increasingN_noProj"
# methodLabel = "no projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)

# savefig(p_computational, targetName*"/constantEta_increasingN_ELBO_Computational.pdf")
# savefig(p_iterative, targetName*"/constantEta_increasingN_ELBO_Iteration.pdf")


# # decreasing eta and increasingN

# p_computational = plot(ylims=(-100,0), xlims=(0,K*N),xlabel="number of cumulated data points",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13)
# p_iterative = plot(ylims=(-100,0), xlims=(0,K/2),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)


# methodName = "decreasingeta_increasingN_proj"
# methodLabel = "projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)


# methodName = "decreasingeta_increasingN_noProj"
# methodLabel = "no projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)

# savefig(p_computational, targetName*"/decreasingEta_increasingN_ELBO_Computational.pdf")
# savefig(p_iterative, targetName*"/decreasingEta_increasingN_ELBO_Iteration.pdf")




# # decreasing eta and constantN

# p_computational = plot(ylims=(-100,0), xlims=(0,K*N),xlabel="number of cumulated data points",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13)
# p_iterative = plot(ylims=(-100,0), xlims=(0,K/2),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)


# methodName = "decreasingeta_constantN_proj"
# methodLabel = "projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)


# methodName = "decreasingeta_constantN_noProj"
# methodLabel = "no projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)



# directory = "./"*targetName

# savefig(p_computational, targetName*"/decreasingEta_constantN_ELBO_Computational.pdf")
# savefig(p_iterative, targetName*"/decreasingEta_constantN_ELBO_Iteration.pdf")




#################################
# Plotting: comparing schedules #
#################################

# p_computational = plot(ylims=(-30,-10), xlims=(0,K*N),xlabel="number of cumulated data points", xticks=[50000,100000], ylabel="ELBO",labelfontsize = 17, tickfontsize = 17, legendfontsize = 17)
# p_iterative = plot(ylims=(-30,-10), xlims=(0,K),xlabel="\$ t \$",xticks=[500,1000],ylabel="ELBO",labelfontsize = 17, tickfontsize = 17, legendfontsize = 17)




# # constant eta and increasingN


# methodName = "constanteta_increasingN_noProj"
# methodLabel = "\$ \\eta_t = 10^{-2},\\,N_t=t+1 \$"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)



# # decreasing eta and increasingN


# methodName = "decreasingeta_increasingN_noProj"
# methodLabel = "\$ \\eta_t = 2/(t + 2),\\,N_t=t+1 \$"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)



# # constant eta and constant N



# methodName = "constanteta_constantN_noProj"
# methodLabel = "\$ \\eta_t = 10^{-2},\\,N_t=10^2 \$"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)




# # decreasing eta and constantN


# methodName = "decreasingeta_constantN_noProj"
# methodLabel = "\$ \\eta_t = 2/(t +2),\\,N_t=10^2 \$"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# computationEffort = readdlm(targetName*"/"*methodName*"/computationEffort.txt")

# plot!(p_computational, computationEffort, ELBO, label=methodLabel, linewidth=2)
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)





# savefig(p_computational, targetName*"/logReg_comparisonSchedulesProj_ELBO_Computational.pdf")
# savefig(p_iterative, targetName*"/logReg_comparisonSchedulesProj_ELBO_Iteration.pdf")




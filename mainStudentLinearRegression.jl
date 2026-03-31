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


###------- Robust linear regression posterior with Bonnet and Price estimator -------###

mutable struct StudentLinReg <: AbstractTarget
    name::String #Name of the target - to write the results
    dimension::Int64 #Dimension of the mean
    isGaussian::Bool
    covPrior
    invCovPrior
    meanPrior
    prior
    sigmaSqLk
    dof
    Z
    y
    M
end



function StudentLinRegTurbine(name, covPrior, meanPrior, sigmaSqLk, dof)
    
    df2013 = CSV.read("./gas+turbine+co+and+nox+emission+data+set/gt_2013.csv", DataFrame)
    
    df = hcat(df2013[1:715,1:7], df2013[1:715,9:11])

    Z = Matrix(df)
    M = size(df,1)


    for i in 1:10
        z = Z[:,i]
        z = (z - mean(z)*ones(M))/std(z)
        Z[:,i] = z
    end


    y = df2013[1:715,8] #TEY
    y = (y - mean(y)*ones(M))/std(y)
    
    d = 10



    return StudentLinReg(name, d, false, covPrior, inv(covPrior), meanPrior, MvNormal(meanPrior, covPrior), sigmaSqLk, dof, Z, y, M)

end



function derivativesLogPi(target::StudentLinReg, x)

    # nablaLogPi = - target.invCovPrior*(x - target.meanPrior)
    nablaLogPi = zeros(target.dimension)
    hessianLogPi = - target.invCovPrior

    for m in 1:target.M

        z = target.Z[m,:]
        qt = (target.y[m] - transpose(z)*x)

        nablaLogPi = nablaLogPi + (((target.dof + 1)*qt) / (target.dof*target.sigmaSqLk + qt^2))*z
        hessianLogPi = hessianLogPi - (((target.dof+1)*(target.dof*target.sigmaSqLk - qt^2)) / (target.dof*target.sigmaSqLk + qt^2)^2)*z*transpose(z)

    end

    return nablaLogPi, hessianLogPi
end

function gradientDescent(target, x_init, stepsize, K)
    logpdfArray = Array{Float64}(undef, K+1)
    logpdfArray[1] = logpdfTarg(target, x_init)
    x = x_init
    for k in 1:K
        nablaLogPi, hessianLogPi = derivativesLogPi(target, x)
        x = x + stepsize*nablaLogPi
        logpdfArray[k+1] = logpdfTarg(target, x)
    end
    return x, logpdfArray
end 




function estimatorTarget(target::StudentLinReg, mean, cov, N)

    d = target.dimension

    q = MvNormal(mean, cov)
    d = target.dimension
    pop = Array{Float64}(undef, d, N)
    rand!(q,pop)

    theta_1 = zeros(d)
    theta_2 = zeros(d,d)

    for n in 1:N
        nablaLogPi, hessianLogPi = derivativesLogPi(target, pop[:,n])
        # theta_1 = theta_1 + (nablaLogPi - hessianLogPi*mean)
        theta_1 = theta_1 + nablaLogPi
        theta_2 = theta_2 + 0.5*hessianLogPi
    end

    return (1/N)*theta_1, (1/N)*theta_2

end



function logpdfTarg(target::StudentLinReg, x)

    res = 0.0

    for m in 1:target.M

        z = target.Z[m,:]
        qLk = MvTDist(target.dof, [transpose(x)*z],  sigmaSqLk*Matrix{Float64}(I,1,1))
        
        res = res + logpdf(qLk, [target.y[m]])
    end

    return res + logpdf(target.prior, x)


end



function evidenceLowerBound(target::StudentLinReg, mean, cov, N_ELBO)
    
    q = MvNormal(mean, cov)
    d = target.dimension
    pop = Array{Float64}(undef, d, N_ELBO)
    rand!(q,pop)
    
    return (1/N_ELBO) * sum([logpdfTarg(target,pop[:,j]) - logpdf(q, pop[:,j]) for j in 1:N_ELBO])
end


function targetKL(target::StudentLinReg, mean, cov)
    prec=inv(cov)
    return 0.5* ( log( det(cov)) - target.logdetCov - target.dimension + transpose(mean - target.mean)*prec*(mean - target.mean) + tr(prec*target.cov) )
end







d = 10
covPrior = Diagonal([5.0 for i in 1:d])
meanPrior = zeros(d)
sigmaSqLk = 1.0
dof= 3.0  
targetName = "StudentLinRegTurbine_dof3"
target = StudentLinRegTurbine(targetName, covPrior, meanPrior, sigmaSqLk, dof)




eta = 0.005
N=250

K=1000
proj="clipping"
# proj="noProj"

alpha=0.0001
beta=10000
N_ELBO = 100
nbRuns = 50

meanInit = zeros(d)
covInit = Array(Diagonal([5.0 for i in 1:d]))




############################################
# Runs using Gaussians with full covariance #
############################################



###################
avgKLtoTarget = zeros(K+1)
avgELBO = zeros(K+1)
avgComputationEffort = zeros(K+1)

stepSizeSchedule = "constant"
sampleSizeSchedule = "constant"
proj = "noProj"
methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N_"*proj
directory = "./"*target.name*"/"*methodName

mkpath(directory)
println(methodName)


for i in 1:nbRuns

    KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

    writedlm(directory*"/ELBO_run"*string(i)*".txt", ELBO)

    global avgKLtoTarget += KLtoTarget
    global avgELBO += ELBO
    global avgComputationEffort += computationEffort

end

avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
avgELBO = (1/nbRuns)*avgELBO
avgComputationEffort = (1/nbRuns)*avgComputationEffort


writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
writedlm(directory*"/ELBO.txt", avgELBO)
writedlm(directory*"/computationEffort.txt", avgComputationEffort)




###################
avgKLtoTarget = zeros(K+1)
avgELBO = zeros(K+1)
avgComputationEffort = zeros(K+1)

stepSizeSchedule = "constant"
sampleSizeSchedule = "constant"
proj = "clipping"
methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N_"*proj
directory = "./"*target.name*"/"*methodName

mkpath(directory)
println(methodName)


for i in 1:nbRuns

    KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

    writedlm(directory*"/ELBO_run"*string(i)*".txt", ELBO)

    global avgKLtoTarget += KLtoTarget
    global avgELBO += ELBO
    global avgComputationEffort += computationEffort

end

avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
avgELBO = (1/nbRuns)*avgELBO
avgComputationEffort = (1/nbRuns)*avgComputationEffort


writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
writedlm(directory*"/ELBO.txt", avgELBO)
writedlm(directory*"/computationEffort.txt", avgComputationEffort)




###################
avgKLtoTarget = zeros(K+1)
avgELBO = zeros(K+1)
avgComputationEffort = zeros(K+1)

stepSizeSchedule = "decreasing"
sampleSizeSchedule = "constant"
proj = "noProj"
methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N_"*proj
directory = "./"*target.name*"/"*methodName

mkpath(directory)
println(methodName)


for i in 1:nbRuns

    KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

    writedlm(directory*"/ELBO_run"*string(i)*".txt", ELBO)

    global avgKLtoTarget += KLtoTarget
    global avgELBO += ELBO
    global avgComputationEffort += computationEffort

end

avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
avgELBO = (1/nbRuns)*avgELBO
avgComputationEffort = (1/nbRuns)*avgComputationEffort


writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
writedlm(directory*"/ELBO.txt", avgELBO)
writedlm(directory*"/computationEffort.txt", avgComputationEffort)




###################
avgKLtoTarget = zeros(K+1)
avgELBO = zeros(K+1)
avgComputationEffort = zeros(K+1)

stepSizeSchedule = "decreasing"
sampleSizeSchedule = "constant"
proj = "clipping"
methodName = stepSizeSchedule*"eta_"*sampleSizeSchedule*"N_"*proj
directory = "./"*target.name*"/"*methodName

mkpath(directory)
println(methodName)


for i in 1:nbRuns

    KLtoTarget, ELBO, computationEffort, mean, cov = GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSizeSchedule, sampleSizeSchedule, target, N_ELBO)

    writedlm(directory*"/ELBO_run"*string(i)*".txt", ELBO)

    global avgKLtoTarget += KLtoTarget
    global avgELBO += ELBO
    global avgComputationEffort += computationEffort

end

avgKLtoTarget = (1/nbRuns)*avgKLtoTarget
avgELBO = (1/nbRuns)*avgELBO
avgComputationEffort = (1/nbRuns)*avgComputationEffort


writedlm(directory*"/KLtoTarget.txt", avgKLtoTarget)
writedlm(directory*"/ELBO.txt", avgELBO)
writedlm(directory*"/computationEffort.txt", avgComputationEffort)









#####################################################################
# Plotting the averaged ELBO: Comparing with and without projection #
#####################################################################


# # constant eta and constant N

# p_iterative = plot(ylims=(-4000,-500), xlims=(0,K/2),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)

# methodName = "constanteta_constantN_clipping"
# methodLabel = "projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")
# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)


# methodName = "constanteta_constantN_noProj"
# methodLabel = "no projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")

# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)

# savefig(p_iterative, targetName*"/constantEta_constantN_ELBO_Iteration.pdf")



# # decreasing eta and constant N


# p_iterative = plot(ylims=(-4000,-500), xlims=(0,K/2),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)


# methodName = "decreasingeta_constantN_clipping"
# methodLabel = "projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")

# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)


# methodName = "decreasingeta_constantN_noProj"
# methodLabel = "no projection"

# ELBO = readdlm(targetName*"/"*methodName*"/ELBO.txt")

# plot!(p_iterative, 0:K, ELBO, label=methodLabel, linewidth=2)



# directory = "./"*targetName

# savefig(p_iterative, targetName*"/decreasingEta_constantN_ELBO_Iteration.pdf")



#################################################
# Plotting the median and quantiles of the ELBO #
#################################################



# decreasing eta constant N


p_iterative = plot(ylims=(-835,-830), xlims=(0,K),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)


methodName = "decreasingeta_constantN_clipping"
methodLabel = "projection"

directory = "./"*targetName*"/"*methodName

all_ELBO = Array{Float64}(undef, K+1, nbRuns)

for i in 1:nbRuns
    ELBO_run_i = readdlm(directory*"/ELBO_run"*string(i)*".txt")
    all_ELBO[:,i] = ELBO_run_i
end

medianELBO = [median(all_ELBO[k,:]) for k in 1:K]
q25_ELBO = [quantile(all_ELBO[k,:], 0.25) for k in 1:K]
q75_ELBO = [quantile(all_ELBO[k,:], 0.75) for k in 1:K]

plot!(p_iterative, 1:K, q25_ELBO, fillrange = q75_ELBO, fillalpha = 0.5, linealpha=0, label = methodLabel)
plot!(p_iterative, medianELBO, c=1,label=false, linewidth=2)

savefig(p_iterative, targetName*"/Student_decreasingEta_constantN_ELBO_confidence_Iteration_ProjOnly.pdf")



# constant eta constant N


p_iterative = plot(ylims=(-835,-830), xlims=(0,K),xlabel="\$ t \$",ylabel="ELBO",labelfontsize = 13, tickfontsize = 13, legendfontsize = 13, legend=:bottomright)


methodName = "constanteta_constantN_clipping"
methodLabel = "projection"

directory = "./"*targetName*"/"*methodName

all_ELBO = Array{Float64}(undef, K+1, nbRuns)

for i in 1:nbRuns
    ELBO_run_i = readdlm(directory*"/ELBO_run"*string(i)*".txt")
    all_ELBO[:,i] = ELBO_run_i
end

medianELBO = [median(all_ELBO[k,:]) for k in 1:K]
q25_ELBO = [quantile(all_ELBO[k,:], 0.25) for k in 1:K]
q75_ELBO = [quantile(all_ELBO[k,:], 0.75) for k in 1:K]

plot!(p_iterative, 1:K, q25_ELBO, fillrange = q75_ELBO, fillalpha = 0.5, linealpha=0, label = methodLabel)
plot!(p_iterative, medianELBO, c=1,label=false, linewidth=2)

savefig(p_iterative, targetName*"/Student_constantEta_constantN_ELBO_confidence_Iteration_ProjOnly.pdf")



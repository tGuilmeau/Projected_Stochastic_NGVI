using LinearAlgebra
using Distributions
using Random



# Implementation of projection operators

function clipping(theta1, cov, alpha,beta)
    #Projection onto the set of covariance matrices with bounded eigenvalues from below and above

    mean = cov * theta1

    decomposition = eigen(cov)
    diag = decomposition.values
    Q = decomposition.vectors
    d = length(diag)

    new_diag = Array{Float64}(undef, d)
    constraint = true
    i = 1
    while (i <= d)
        if alpha <= diag[i] <= beta
            new_diag[i] = diag[i]
        else
            constraint = false
            # new_diag[i] = max(max(alpha,lb),min(beta, diag[i]))
            new_diag[i] = max(alpha,min(beta, diag[i]))
        end
        i = i+1
    end
    
    #if the constraint are staisified by cov, cov is directly returned
    if constraint
        result = cov
    else
        result = Q*Diagonal(new_diag)*inv(Q)
        result = 0.5*(transpose(result) + result) #to correct small numerical errors
    end

    return mean, result
end



function positiveProj(mean)
    #Projection onto the set of mean vector with non-negative components
    return [max(m,0.0) for m in mean]
end





# Implementations of the algorithms




function GaussianProjNGVImodular(meanInit, covInit, eta, N, proj, alpha, beta, K, stepSchedule, sampleSchedule, target, N_ELBO)
    #
    #Projected NGVI algorithm when the approximating family is the family of Gaussian distributions with full covariance 
    #Several step size and sample/batch size schedules are possible
    #
    #meanInit / covInit is the initial mean / covariance matrix of the Gaussian approximation
    #eta is the step size
    #N is the sample size to be used when calling estimator
    #alpha / beta is the lower / beta bound on the eigenvalues of the covariance is projections are used (if proj = "clipping")
    #K is the maximal sample size
    #target represents the target with associated features and functions
    #N_ELBO is the number of samples to be used when computing the ELBO at each iteration

    mean = meanInit
    cov = covInit
    d = target.dimension

    KLtoTarget = Array{Float64}(undef, K+1) #array recording the KL divergence between the current proposal and the true target (when possible)
    ELBO = Array{Float64}(undef, K+1) #array recording the ELBO
    computationEffort = Array{Int64}(undef, K+1) #array recording the total number of samples used so far

    computationEffortCount = 0

    for k in 0:K

        #performance indicators
        if target.isGaussian
            KLtoTarget[k+1] = targetKL(target, mean,cov)
        else
            KLtoTarget[k+1] = -1.0
        end

        ELBO[k+1] = evidenceLowerBound(target, mean, cov, N_ELBO)

        computationEffort[k+1] = computationEffortCount

        #update
        if stepSchedule == "constant"
            stepSize = eta
        elseif stepSchedule == "decreasing"
            stepSize = 1/(0.5*k+1)
        else
            error("unknown step size schedule")
        end

        if sampleSchedule == "constant"
            sampleSize = N
        elseif sampleSchedule == "increasing"
            sampleSize = k+1#max(k+1,10)
        else
            error("unknown sample/batch size schedule")
        end

        
        (m,S) = estimatorTarget(target, mean, cov, sampleSize)

        prec = inv(cov)

        theta1 = prec*mean
        theta2 = -0.5*prec

        theta1_new = (1-stepSize)*theta1 + stepSize*m
        theta2_new = (1-stepSize)*theta2 + stepSize*S

        cov_new = -0.5*inv(theta2_new)

        computationEffortCount = computationEffortCount + sampleSize

        #eventual projection step
        if proj == "clipping"
            mean, cov = clipping(theta1_new, cov_new, alpha, beta)
        end

        cov = 0.5*(transpose(cov) + cov)
        

        

    end

    return KLtoTarget, ELBO, computationEffort, mean, cov


end












function GaussianProjNGVImodularDiag(meanInit, covInit, eta, N, proj, K, stepSchedule, sampleSchedule, target, N_ELBO)
    #
    #Projected NGVI algorithm when the approximating family is the family of Gaussian distributions with diagonal covariance 
    #Several step size and sample/batch size schedules are possible
    #
    #meanInit / covInit is the initial mean / covariance matrix of the Gaussian approximation
    #eta is the step size
    #N is the sample size to be used when calling estimator
    #proj is a Boolean indicating whether or not projection should be applied (by default, the mean is porjected on the non-negative orthant)
    #K is the maximal sample size
    #target represents the target with associated features and functions
    #N_ELBO is the number of samples to be used when computing the ELBO at each iteration

    mean = meanInit
    cov = Diagonal(covInit)
    d = target.dimension

    KLtoTarget = Array{Float64}(undef, K+1) #array recording the KL divergence between the current proposal and the true target (when possible)
    ELBO = Array{Float64}(undef, K+1) #array recording the ELBO
    computationEffort = Array{Int64}(undef, K+1) #array recording the total number of samples used so far

    computationEffortCount = 0

    for k in 0:K

        #performance indicators
        if target.isGaussian
            KLtoTarget[k+1] = targetKL(target, mean,cov)
        else
            KLtoTarget[k+1] = -1.0
        end

        ELBO[k+1] = evidenceLowerBound(target, mean, cov, N_ELBO)

        computationEffort[k+1] = computationEffortCount

        #update
        if stepSchedule == "constant"
            stepSize = eta
        elseif stepSchedule == "decreasing"
            stepSize = 1/(0.5*k+1)
        else
            error("unknown step size schedule")
        end

        if sampleSchedule == "constant"
            sampleSize = N
        elseif sampleSchedule == "increasing"
            sampleSize = k+1
        else
            error("unknown sample/batch size schedule")
        end

        
        (m,s) = estimatorTargetDiag(target, mean, cov, sampleSize)
        S = Diagonal(s)

        prec = inv(cov)

        theta1 = prec*mean
        theta2 = -0.5*prec

        theta1_new = (1-stepSize)*theta1 + stepSize*m
        theta2_new = (1-stepSize)*theta2 + stepSize*S

        cov = -0.5*inv(theta2_new)
        mean = cov*theta1_new 
        

        computationEffortCount = computationEffortCount + sampleSize

        #eventual projection step
        if proj
            mean = positiveProj(mean)
        end 

    end

    return KLtoTarget, ELBO, computationEffort, mean, cov


end






function GaussianProjNGVIadjustedSampleSize(meanInit, covInit, eta, N, gamma, proj, alpha, beta, K, stepSchedule, sampleSchedule, target, N_ELBO)
    #
    #Projected NGVI algorithm when the approximating family is the family of Gaussian distributions with full covariance 
    #Several step size and sample/batch size schedules are possible, with additional flexibility for the sample/batch size schedule
    #
    #meanInit / covInit is the initial mean / covariance matrix of the Gaussian approximation
    #eta is the step size
    #N is the sample size to be used when calling estimator
    #if increasing sample size schedule is chosen, gamma governs the increase as $N_t = (t+1)^\gamma$
    #alpha / beta is the lower / beta bound on the eigenvalues of the covariance is projections are used (if proj = "clipping")
    #K is the maximal sample size
    #target represents the target with associated features and functions
    #N_ELBO is the number of samples to be used when computing the ELBO at each iteration

    mean = meanInit
    cov = covInit
    d = target.dimension

    KLtoTarget = Array{Float64}(undef, K+1) #array recording the KL divergence between the current proposal and the true target (when possible)
    ELBO = Array{Float64}(undef, K+1) #array recording the ELBO
    computationEffort = Array{Int64}(undef, K+1) #array recording the total number of samples used so far


    computationEffortCount = 0



    for k in 0:K

        #performance indicators
        if target.isGaussian
            KLtoTarget[k+1] = targetKL(target, mean,cov)
        else
            KLtoTarget[k+1] = -1.0
        end

        ELBO[k+1] = evidenceLowerBound(target, mean, cov, N_ELBO)

        computationEffort[k+1] = computationEffortCount

        #update
        if stepSchedule == "constant"
            stepSize = eta
        elseif stepSchedule == "decreasing"
            stepSize = 1/(0.5*k+1)
        else
            error("unknown step size schedule")
        end

        if sampleSchedule == "constant"
            sampleSize = N
        elseif sampleSchedule == "increasing"
            sampleSize = Int64(floor((k+1)^gamma))+1
        else
            error("unknown sample/batch size schedule")
        end

        
        (m,S) = estimatorTarget(target, mean, cov, sampleSize)

        prec = inv(cov)

        theta1 = prec*mean
        theta2 = -0.5*prec

        theta1_new = (1-stepSize)*theta1 + stepSize*m
        theta2_new = (1-stepSize)*theta2 + stepSize*S

        cov_new = -0.5*inv(theta2_new)

        computationEffortCount = computationEffortCount + sampleSize

        #eventual projection step
        if proj=="clipping"
            mean, cov = clipping(theta1_new, cov_new, alpha,beta)
        end

        cov = 0.5*(cov + transpose(cov))

        

    end

    return KLtoTarget, ELBO, computationEffort, mean, cov


end

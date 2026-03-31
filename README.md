# Projected_Stochastic_NGVI
This repository contains the source code to reproduce the numerical experiments showcased in the paper "Convergence of projected stochastic natural gradient variational inference for various step size and sample or batch size schedules" at AISTATS 2026.

## Organization of the code
The file `methodGaussian.jl` contains the algorithms used for the experiments. Only algorithms with Gaussian variational families are considerd. Each file of the form `main_____.jl` corresponds to a specific experiment and a specific target dsitributions.

## Target distributions
Target distributions can be declared as `struct` objects, which must a minima (a minimal target being represented by the type `AbstractTarget`) be associated to a function returning the estimator $g^N$ and a function returning the ELBO for a given Gaussian approximation and a given number of samples. Then, the target distribution can be passed as a an argument of the algorithms.

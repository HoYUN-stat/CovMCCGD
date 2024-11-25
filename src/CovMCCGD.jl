module CovMCCGD
using LinearAlgebra
using Random: seed!

include("common.jl") # Include the common custom types and related operations

export BlockVec
export BlockDiag
export zerox

include("mat_cg.jl") # Matrized CG method

export LazyKhatri
export matrix_cg!
export matrix_cg

include("grid.jl") # Generate lococation grid and Gram matrix

export loc_grid
export ReproducingKernel
export GaussianKernel
export LaplacianKernel
export LinearKernel
export PolynomialKernel
export CustomKernel
export gram_matrix

include("process.jl") # Generate Stochastic Processes

export StochasticProcess
export BrownianMotion
export BrownianBridge
export IntegratedBM
export OrnsteinUhlenbeck
export true_sec_mom
export perturb_process!
export perturb_process
export color_range

include("cov_eval.jl") # Covariance Evalaution

export FrameMat
export frame_matrix
export eval_sec_mom

end
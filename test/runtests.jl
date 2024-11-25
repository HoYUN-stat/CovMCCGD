using Test
using LinearAlgebra
using Random: seed!

include("../src/CovMCCGD.jl")
using .CovMCCGD

@testset "CovMCCGD.jl" begin
    include("test_src/test_common.jl")
    include("test_src/test_mat_cg.jl")
    include("test_src/test_grid.jl")
    include("test_src/test_process.jl")
    include("test_src/test_cov_eval.jl")
end
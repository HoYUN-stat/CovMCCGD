@testset "Gram Matrix Tests" begin
    n, r = 10, 20
    x = loc_grid(n, r, λ=0)
    kernel = GaussianKernel(0.1)
    K = gram_matrix(x, kernel)
    @test size(K) == (n * r, n * r)
end


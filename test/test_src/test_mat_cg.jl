@testset "LazyKhatri Tests" begin
    # Create a BlockVec example
    n, r = 5, 4
    K = rand(n * r, n * r)
    khatK = LazyKhatri(K, 0.1, n, r)

    @test eltype(khatK) == Float64
    @test size(khatK) == (n * r^2, n * r^2)
end

@testset "MCCGD Tests" begin
    n, r = 200, 20
    A = rand(n * r, n * r)
    K = A * A' # s.p.d. matrix
    isposdef(K)
    khatK = LazyKhatri(K, 0.1, n, r)

    B = BlockDiag([ones(r, r) for _ in 1:n], n, r)
    Y = BlockDiag([ones(r, r) for _ in 1:n], n, r)
    @test size(B) == size(matrix_cg(khatK, Y; maxiter=5))
end
@testset "FrameMat Tests" begin
    # Create a BlockVec example
    n, r, m = 5, 10, 10
    block = [rand(r, m) for _ in 1:n]
    temp = rand(r, m)
    F = FrameMat(block, n, r, m, temp)
    @test size(F) == (n * r, m)
end


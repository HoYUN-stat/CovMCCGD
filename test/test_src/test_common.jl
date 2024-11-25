@testset "BlockVec Tests" begin
    # Create a BlockVec example
    n, r = 3, 2
    block = [rand(r) for _ in 1:n]
    y = BlockVec(block, n, r)

    # Check the size of the BlockVec
    @test size(y) == (n * r,)

    # Check that elements are stored correctly
    @test length(y.block) == n
    @test all(size(b) == (r,) for b in y.block)
end

@testset "BlockDiag Tests" begin
    # Create a BlockDiag example
    n, r = 3, 2
    block = [rand(r, r) for _ in 1:n]
    B = BlockDiag(block, n, r)

    # Check the size of the BlockDiag
    @test size(B) == (n * r, n * r)

    # Check if the temp buffer was correctly initialized
    @test size(B.temp) == (r, r)

    # Check iteration count for matrix_cg (should be 0 initially)
    @test B.iter == 0
end

@testset "zerox Tests" begin
    n, r = 3, 2
    A = BlockDiag([rand(r, r) for _ in 1:n], n, r, rand(r, r), 0)

    # Test for zeroing a BlockDiag
    zero_B = zerox(A, matrize=true)
    @test all(x -> all(y -> y == 0, x), zero_B.block)  # All elements should be zero

    # Test for zeroing a BlockVec
    zero_y = zerox(A, matrize=false)
    @test all(x -> all(y -> y == 0, x), zero_y.block)  # All elements should be zero
end


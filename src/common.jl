"""
        BlockVec{T}

    n  block vector with each vector block of size r
    
    #Fields
    block::Vector{Vector{T}}  # Saving the array of vector blocks
    n::Int64                     # Number of blocks
    r::Int64                     # Size of each block 

    # Examples
    ```julia-repl
    julia> n, r = 3, 2
       block = [rand(r) for _ in 1:n]
       y = BlockVec(block, n, r)
       size(y)
    (6,)
    ```
"""
struct BlockVec{T}
    block::Vector{Vector{T}}
    n::Int64
    r::Int64
end
Base.eltype(y::BlockVec) = eltype(y.block[1])
Base.size(y::BlockVec) = (y.n * y.r,)

"""
        BlockDiag{T}

    n × n block diagonal matrix B with each block of size r × r
    
    #Fields
    block::Vector{Matrix{T}}    # Saving the array of matrix blocks
    n::Int64                     # Number of blocks
    r::Int64                     # Size of each block
    temp::Matrix{T} #temp buffer for mul!, size: r x r (optional)
    iter::Int64     # Number of iterations for matrix_cg!, initialized to 0 

    # Examples
    ```julia-repl
    julia> n, r = 3, 2
       block = [rand(r, r) for _ in 1:n]
       @time B = BlockDiag(block, n, r);
    0.000005 seconds (2 allocations: 144 bytes)
    ```
"""
mutable struct BlockDiag{T}
    block::Vector{Matrix{T}}  # Saving the array of matrix blocks
    n::Int64                     # Number of blocks
    r::Int64                     # Size of each block
    temp::Matrix{T} #temp buffer for mul!, size: r x r
    iter::Int64     # Number of iterations for matrix_cg!, initialized to 0
end
Base.eltype(B::BlockDiag) = eltype(B.block[1])
Base.size(B::BlockDiag) = (B.n * B.r, B.n * B.r)
Base.size(B::BlockDiag, i::Int) = size(B)[i]

function BlockDiag(block::Vector{Matrix{T}}, n::Int64, r::Int64) where {T}
    BlockDiag(block, n, r, Matrix{T}(undef, r, r), 0)
end

#########################################
########### Custom Operations ###########
#########################################
"""
        blocktensor(y::BlockVec{T})::BlockDiag{T}

    Convert a block-vector y into a block matrix with each block representing y[i] * y[i]'
    # Examples
    ```julia-repl
    julia> n, r = 200, 40
       block = [2 .* ones(r) for _ in 1:n];
       y = BlockVec(block, n, r);
       @time blocktensor(y);
    0.000348 seconds (202 allocations: 2.468 MiB)
    ```
"""
function blocktensor(y::BlockVec{T})::BlockDiag{T} where {T}
    return BlockDiag([y.block[i] * y.block[i]' for i in 1:y.n], y.n, y.r, Matrix{T}(undef, y.r, y.r), 0)
end

"""
        frobenius(A::BlockDiag{T}, B::BlockDiag{T})::T
        frobenius(A::BlockDiag{T})::T 

    Compute the Frobenius inner product between block diagonal matrices A and B.
    When A and B are the same, compute the Frobenius squared norm of A.

    # Examples
    ```julia-repl
    julia> n, r = 200, 10
       block = [ones(r, r) for _ in 1:n];
       temp = rand(r, r);
       B = BlockDiag(block, n, r, temp, 0);
       A = BlockDiag(block, n, r, temp, 0);

    julia> @time frobenius(A, B) # Should be nr^2
    0.000023 seconds (1 allocation: 16 bytes)
    20000.0

    julia> @time frobenius(A)  # Should be nr^2
    0.000021 seconds (1 allocation: 16 bytes)
    20000.0
    ```
"""
function frobenius(A::BlockDiag{T}, B::BlockDiag{T})::T where {T}
    # Dimension check
    @assert B.n == A.n
    @assert B.r == A.r

    norm = zero(T)
    for i = 1:B.n
        norm += dot(A.block[i], B.block[i])
    end
    return norm
end
frobenius(A::BlockDiag) = frobenius(A, A)

"""
        axpby!(α::T, A::BlockDiag{T}, β::T, B::BlockDiag{T})::BlockDiag{T} -> B

    Overwrite B with α * A + β * B in-place and return B.

    # Examples
    ```julia-repl
    julia> n, r = 200, 20
       α, β = -1.5, 2.0
       temp = rand(r, r);
       A = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       B = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       
       @time axpby!(α, A, β, B);
       B.block[1] # Should be (α + β) .* ones(r, r)
    0.000085 seconds
    20×20 Matrix{Float64}:
    0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 
    0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 
    0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 
    0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 
    0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 
    ```

"""
function LinearAlgebra.axpby!(α::T, A::BlockDiag{T}, β::T, B::BlockDiag{T})::BlockDiag{T} where {T}
    # Dimension check
    @assert A.n == B.n
    @assert A.r == B.r

    for i = 1:A.n
        LinearAlgebra.axpby!(α, A.block[i], β, B.block[i])
    end
    return B
end

"""
        axpy!(α::T, A::BlockDiag{T}, B::BlockDiag{T})::BlockDiag{T} -> B

    Overwrite B with α * A + B in-place and return B.

    # Examples
    ```julia-repl
    julia> n, r = 3, 2
       α = -1.5
       temp = rand(r, r);
       A = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       B = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       
       @time LinearAlgebra.axpy!(α, A, B)# Each block should be (α + 1) .* ones(r, r)       
    0.000005 seconds
    BlockDiag{Float64}([[-0.5 -0.5; -0.5 -0.5], [-0.5 -0.5; -0.5 -0.5], [-0.5 -0.5; -0.5 -0.5]], 3, 2)
    ```

"""
function LinearAlgebra.axpy!(α::T, A::BlockDiag{T}, B::BlockDiag{T})::BlockDiag{T} where {T}
    # Dimension check
    @assert A.n == B.n
    @assert A.r == B.r

    for i = 1:A.n
        LinearAlgebra.axpy!(α, A.block[i], B.block[i])
    end
    return B
end

"""
        zerox(y::Union{BlockVec{T},BlockDiag{T}}; matrize=true)

    Return a zero block vector or block diagonal matrix with compatible size as y.

    #Arguments
    - `y::Union{BlockVec{T},BlockDiag{T}}`: Block vector or block diagonal matrix
    - `matrize::Bool=true`: If true, return a zero block diagonal matrix; otherwise, return a zero block vector

    # Examples
    ```julia-repl
    julia> n, r = 3, 2
       α = -1.5
       temp = rand(r, r);
       A = BlockDiag([ones(r, r) for _ in 1:n], n, r, temp, 0);
       @time zerox(A);
    0.000008 seconds (6 allocations: 512 bytes)
    ```

"""
function zerox(y::Union{BlockVec{T},BlockDiag{T}}; matrize::Bool=true) where {T}
    if matrize
        temp = rand(y.r, y.r)
        return BlockDiag([zeros(y.r, y.r) for _ in 1:y.n], y.n, y.r, temp, 0)
    else
        return BlockVec([zeros(T, y.r) for _ in 1:y.n], y.n, y.r)
    end
end

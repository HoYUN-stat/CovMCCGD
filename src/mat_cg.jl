"""
        LazyKhatri{T}

    Lazy Khatri-Rao product of nr x nr Gram matrix K with regularization η, i.e. (K ∗ K + η I)
    
    #Fields
    K::Matrix{T} # size: nr x nr
    η::T
    n::Int64
    r::Int64

    # Examples
    ```julia-repl
    julia> n, r = 5, 4
       K = rand(n*r, n*r);
       khatK = LazyKhatri(K, 0.1, n, r);

    julia> eltype(khatK)
    Float64

    julia> size(khatK) # Should be nr^2 x nr^2
    (80, 80)
    ```
"""
struct LazyKhatri{T}
    K::Matrix{T} # size: nr x nr
    η::T
    n::Int64
    r::Int64
end
Base.eltype(khatK::LazyKhatri) = eltype(khatK.K)
Base.size(khatK::LazyKhatri) = (khatK.n * khatK.r^2, khatK.n * khatK.r^2)
Base.size(khatK::LazyKhatri, i::Int) = size(khatK)[i]

"""
        mul!(B::BlockDiag{T}, khatK::LazyKhatri{T}, A::BlockDiag{T})::BlockDiag{T} -> B

    Compute the Khatri-Rao product dvec(C) = (K * K + η I) dvec(B), or equivalently,
        C.block[i] = K[i, :] * B * transpose(K[i, :]) + 2 * η * B.block[i]
    
    # Examples
    ```julia-repl
    julia> n, r = 200, 40;
       η = 0.1
       temp = rand(r, r);
       B = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       C = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       
       # Symmetrize blocks
       B.block = [(B.block[i] + B.block[i]') / 2 for i in 1:n];
       C.block = [(C.block[i] + C.block[i]') / 2 for i in 1:n];
       
       K = ones(n * r, n * r);
       K = (K + K') / 2;
       temp = rand(r, r);
       khatK = LazyKhatri(K, η, n, r);
       
       using BenchmarkTools
       @btime mul!(C, khatK, B).block; # Each block should be (nr^2+η) .* ones(r, r)
    321.988 ms (0 allocations: 0 bytes)
    ```
"""
function LinearAlgebra.mul!(B::BlockDiag{T}, khatK::LazyKhatri{T}, A::BlockDiag{T})::BlockDiag{T} where {T}
    # Dimension check
    @assert B.n == A.n == khatK.n
    @assert B.r == A.r == khatK.r

    @inbounds for i = 1:B.n
        LinearAlgebra.mul!(B.block[i], khatK.η, A.block[i])
        for j = 1:B.n
            K_block = view(khatK.K, (i-1)*khatK.r+1:i*khatK.r, (j-1)*khatK.r+1:j*khatK.r)
            LinearAlgebra.mul!(A.temp, K_block, A.block[j])
            LinearAlgebra.mul!(B.block[i], A.temp, transpose(K_block), one(T), one(T))
        end
    end
    return B
end

"""
        symdiagelim!(B::BlockDiag{T})::BlockDiag{T}

    In-place operation of eliminating diagonal elements after symmetrization.
    This is for the initial step of the conjugate gradient method.

    # Examples
    ```julia-repl
    julia> n, r = 2, 2
       block = [rand(r, r) for _ in 1:n];
       B = BlockDiag(block, n, r)
    BlockDiag{Float64}([[0.4830083738124016 0.282899062706514; 0.15038508402186845 0.6582764774976042], [0.8172937285095074 0.7613804451303424; 0.5719808760658703 0.8097984536047786]], 2, 2, [0.25730627884978985 0.16854783955205355; 0.13089055173216957 0.4200897845538295], 1)

    julia> @time symdiagelim!(B).block
    0.000015 seconds
    2-element Vector{Matrix{Float64}}:
    [0.0 0.6399732158743563; 0.6399732158743563 0.0]
    [0.0 0.7949930235183594; 0.7949930235183594 0.0]
    ```
"""
function symdiagelim!(B::BlockDiag{T})::BlockDiag{T} where {T}
    for i = 1:B.n
        A = B.block[i]
        @inbounds for j = 1:B.r
            A[j, j] = zero(T)
            for k = (j+1):B.r  # Iterate over the upper triangular part
                A[j, k] = (A[j, k] + A[k, j]) / 2
                A[k, j] = A[j, k]  # Ensure symmetry
            end
        end
    end
    return B
end

"""
        matrix_cg!(khatK::LazyKhatri{T}, y::Union{BlockVec{T},BlockDiag{T}}, B::BlockDiag{T};
    maxiter::Int64=500, tol=1e-10, trace_residual::Bool=false) -> B, residuals (optional)

    Solve the linear system in-place using MCCGD algorithm.

    #Arguments
    - `khatK::LazyKhatri{T}`: Lazy Khatri-Rao product of nr x nr Gram matrix K with regularization 2*η, i.e. (K ∗ K + 2*η I)
    - `y::Union{BlockVec{T}, BlockDiag{T}}`: Right-hand side of the linear system to solve, could be a block vector or block diagonal matrix
    - `B::BlockDiag{T}`: Initial guess
    - `maxiter::Int64=500`: Maximum number of iterations
    - `tol::Real=1e-10`: Tolerance for convergence
    - `trace_residual::Bool=false`: If true, return the array of residuals

    # Examples
    ```julia-repl
    julia> n, r = 200, 20
       A = rand(n * r, n * r);
       K = A * A'; # s.p.d. matrix
       isposdef(K)
       khatK = LazyKhatri(K, 0.1, n, r);
       
       B = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       symdiagelim!(B);
       B.iter
       Y = BlockDiag([ones(r, r) for _ in 1:n], n, r);
       @time mul!(Y, khatK, B);
       
       @time B0, residuals = matrix_cg(khatK, Y; maxiter = 500, tol=1e-1, trace_residual=true);
    0.071177 seconds
    Number of iterations: 177, residual: 0.07993722757996574
    10.006848 seconds (948 allocations: 2.642 MiB)

    julia> residuals[end-5:end]
    6-element Vector{Float64}:
    0.21421773691030677
    0.17063758715206914
    377.00245158647624
    0.13568660069461497
    0.10393609942192308
    0.07993722757996574

    julia> isapprox(B0.block[1], B.block[1])
    true
    ```

"""
function matrix_cg!(khatK::LazyKhatri{T}, y::Union{BlockVec{T},BlockDiag{T}}, B::BlockDiag{T};
    maxiter::Int64=500, tol=1e-10,
    trace_residual::Bool=false) where {T}

    # Dimension check
    @assert y.n == B.n == khatK.n
    @assert y.r == B.r == khatK.r

    if trace_residual
        #Pre-allocate an array to store the residual
        residuals = T[]
    end

    B.iter = 0
    symdiagelim!(B) # Project initial guess
    # Pre-allocate temp buffers
    V = deepcopy(B)
    δ_old = zero(T)
    δ_new = zero(T)
    α = zero(T)
    β = zero(T)

    mul!(V, khatK, B)
    if y isa BlockVec
        R = blocktensor(y) #This is what I actually use
    else
        R = deepcopy(y) #This is better for checking the convergence
    end
    # Initial residual
    axpy!(-one(T), V, R)
    # Project the residual
    symdiagelim!(R)
    # Initial conjugate gradient
    P = deepcopy(R)
    # Compute initial squared residual norm
    δ_old = frobenius(P)

    if trace_residual
        push!(residuals, δ_old)
    end

    # Iterate until convergence
    while δ_old > tol
        B.iter += 1
        # Compute V = khatK * P in-place
        mul!(V, khatK, P)
        # Compute step size
        α = δ_old / frobenius(P, V)
        # Update solution in-place
        axpy!(α, P, B)
        # Update the residual in-place
        axpy!(-α, V, R)
        # Project the residual
        R = symdiagelim!(R)
        # Compute new squared residual norm
        δ_new = frobenius(R)

        if trace_residual
            push!(residuals, δ_new)
        end

        # Check for convergence
        if δ_new < tol || B.iter >= maxiter
            break
        end

        # Compute step size
        β = δ_new / δ_old
        # Update the conjugate gradient in-place
        axpby!(one(T), R, β, P)
        # Update squared residual norm for next iteration
        δ_old = δ_new
    end

    # Return the solution
    println("Number of iterations: ", B.iter, ", residual: ", δ_new)
    if trace_residual
        return B, residuals
    else
        return B
    end
end

"""
        matrix_cg(khatK::LazyKhatri{T}, y::Union{BlockVec{T},BlockDiag{T}};
    maxiter::Int64=500, tol=1e-10, trace_residual::Bool=false)

    Solve the linear system in-place using MCCGD algorithm with the initial guess 0.

    #Arguments
    - `khatK::LazyKhatri{T}`: Lazy Khatri-Rao product of nr x nr Gram matrix K with regularization 2*η, i.e. (K ∗ K + 2*η I)
    - `y::Union{BlockVec{T}, BlockDiag{T}}`: Right-hand side of the linear system to solve, could be a block vector or block diagonal matrix
    - `maxiter::Int64=500`: Maximum number of iterations
    - `tol::Real=1e-10`: Tolerance for convergence
    - `trace_residual::Bool=false`: If true, return the array of residuals

    # Examples
    ```julia-repl
    julia> n, r = 200, 20
       A = rand(n * r, n * r);
       K = A * A'; # s.p.d. matrix
       isposdef(K)
       temp = rand(r, r);
       khatK = LazyKhatri(K, 0.1, n, r);
       
       B = BlockDiag([ones(r, r) for _ in 1:n], n, r, temp, 0);
       symdiagelim!(B);
       B.iter
       Y = BlockDiag([ones(r, r) for _ in 1:n], n, r, temp, 0);
       @time mul!(Y, khatK, B);
       
       @time B0, residuals = matrix_cg(khatK, Y; maxiter = 500, tol=1e-1, trace_residual=true);
    0.071177 seconds
    Number of iterations: 177, residual: 0.07993722757996574
    10.006848 seconds (948 allocations: 2.642 MiB)

    julia> residuals[end-5:end]
    6-element Vector{Float64}:
    0.21421773691030677
    0.17063758715206914
    377.00245158647624
    0.13568660069461497
    0.10393609942192308
    0.07993722757996574

    julia> isapprox(B0.block[1], B.block[1])
    true
    ```

"""
matrix_cg(khatK::LazyKhatri, y::Union{BlockVec,BlockDiag};
    maxiter::Int64=500, tol=1e-10, trace_residual::Bool=false) =
    matrix_cg!(khatK, y, zerox(y); maxiter=maxiter, tol=tol, trace_residual=trace_residual)



"""
        loc_grid(n::Int64, r::Int64; λ::Int64 = 0, seed::Int64=42)::BlockVec{Float64}

    Generate a grid of locations for n functions with r locations each.

    #Arguments
    - `n::Int64`: Number of functions
    - `r::Int64`: Number of locations for each function
    - `λ::Int64 = 0`: 0 for random locations, 1 for regular locations
    - `seed::Int64 = 42`: Seed for reproducibility

    # Examples
    ```julia-repl
    julia> n, r = 200, 40;
       @time loc_grid(n, r, λ=0);
    0.000270 seconds (207 allocations: 80.344 KiB)
    ```
"""
function loc_grid(n::Int64, r::Int64; λ::Int64=0, seed::Int64=42)::BlockVec{Float64}
    if λ == 0   #Random locations
        seed!(seed)
        loc = [sort!(rand(Float64, r)) for _ in 1:n]
    else        #Regular locations
        loc = [collect(range(start=0.0, step=1.0 / r, length=r)) for _ in 1:n]
    end
    return BlockVec(loc, n, r)
end

## Generate Gram Matrix##

"""
        ReproducingKernel

    Abstract type for reproducing kernels with 5 subtypes below.
    
    # Fields
    - `GaussianKernel`: Gaussian kernel with parameter γ
        - K(x, y) = exp(-γ * (x - y)^2)
    - `LaplacianKernel`: Laplacian kernel with parameter γ
        - K(x, y) = exp(-γ * |x - y|)
    - `LinearKernel`: Linear kernel
        - K(x, y) = x * y
    - `PolynomialKernel`: Polynomial kernel with degree d and coefficient c
        - K(x, y) = (x * y + c)^d
    - `CustomKernel`: Custom kernel with a s.p.d. bivariate function K(x, y)
    ```
"""
abstract type ReproducingKernel end

struct GaussianKernel <: ReproducingKernel
    γ::Float64  # Kernel parameter
    GaussianKernel(γ::Float64) = new(γ)
end

struct LaplacianKernel <: ReproducingKernel
    γ::Float64  # Kernel parameter
    LaplacianKernel(γ::Float64) = new(γ)
end

struct LinearKernel <: ReproducingKernel end

struct PolynomialKernel <: ReproducingKernel
    d::Int      # Degree of the polynomial
    c::Float64  # Coefficient for bias term
    PolynomialKernel(d::Int, c::Float64) = new(d, c)
end

struct CustomKernel <: ReproducingKernel
    K::Function # Custom kernel function, K(x, y)
end

"""
        compute_kernel(x::Float64, y::Float64, kernel::ReproducingKernel)::Float64

    Compute the kernel value given x, y, and kernel type.
    
    # Arguments
    - `x::Float64`: First input
    - `y::Float64`: Second input
    - `kernel::ReproducingKernel`: Kernel type

    # Examples
    ```julia-repl
    julia> x, y = 0.5, 0.7;
       kernel = GaussianKernel(0.1);
       compute_kernel(x, y, kernel)
    0.9960079893439915
    ```
"""
function compute_kernel(x::Float64, y::Float64, kernel::ReproducingKernel)::Float64
    if kernel isa GaussianKernel
        return exp(-kernel.γ * (x - y)^2)
    elseif kernel isa LaplacianKernel
        return exp(-kernel.γ * abs(x - y))
    elseif kernel isa LinearKernel
        return x * y
    elseif kernel isa PolynomialKernel
        return (x * y + kernel.c)^kernel.d
    elseif kernel isa CustomKernel
        return kernel.f(x, y)
    else
        error("Unsupported kernel type: $(typeof(kernel))")
    end
end

"""
        compute_kernel(x::AbstractVector{Float64}, y::AbstractVector{Float64}, kernel::ReproducingKernel)::Matrix{Float64}

    Compute the kernel value given x, y, and kernel type.
    
    # Arguments
    - `x::AbstractVector{Float64}`: First input
    - `y::AbstractVector{Float64}`: Second input
    - `kernel::ReproducingKernel`: Kernel type

    # Examples
    ```julia-repl
    julia> x = y = range(start=0.0, stop=1.0, length=5);

    julia> compute_kernel(x, y, GaussianKernel(5.0))
    5×5 Matrix{Float64}:
    1.0         0.731616   0.286505  0.0600547  0.00673795
    0.731616    1.0        0.731616  0.286505   0.0600547
    0.286505    0.731616   1.0       0.731616   0.286505
    0.0600547   0.286505   0.731616  1.0        0.731616
    0.00673795  0.0600547  0.286505  0.731616   1.0
    ```
"""
function compute_kernel(x::AbstractVector{Float64}, y::AbstractVector{Float64}, kernel::ReproducingKernel)::Matrix{Float64}
    K = Matrix{Float64}(undef, length(x), length(y))
    for i in eachindex(x)
        for j in eachindex(y)
            K[i, j] = compute_kernel(x[i], y[j], kernel)
        end
    end
    return K
end

"""
        gram_matrix(x::BlockVec{Float64}, kernel::ReproducingKernel)::Matrix{Float64}

    Generate Gram matrix given location grid x and kernel type.

    # Arguments
    - `x::BlockVec{Float64}`: Block vector of locations
    - `kernel::ReproducingKernel`: Kernel type

    # Examples
    ```julia-repl
    julia> n, r = 200, 20;
       x = loc_grid(n, r, λ=0);
       kernel = GaussianKernel(0.1);
       @time K = gram_matrix(x, kernel);
    0.083986 seconds (3 allocations: 122.074 MiB, 8.24% gc time)
    ```
"""
function gram_matrix(x::BlockVec{Float64}, kernel::ReproducingKernel)::Matrix{Float64}
    # Pre-allocate the result matrix
    K = Matrix{Float64}(undef, x.n * x.r, x.n * x.r)
    # Precompute block indices
    indices = [(i-1)*x.r+1:i*x.r for i in 1:x.n]

    for i1 in 1:x.n
        block1 = x.block[i1]
        idx1 = indices[i1]
        for i2 in i1:x.n  # Only compute for i2 >= i1
            block2 = x.block[i2]
            idx2 = indices[i2]

            @inbounds for row in 1:x.r
                for col in 1:x.r
                    value = compute_kernel(block1[row], block2[col], kernel)
                    K[idx1[row], idx2[col]] = value
                    if i1 != i2  # Fill symmetric entry only for off-diagonal blocks
                        K[idx2[col], idx1[row]] = value
                    end
                end
            end
        end
    end
    return K
end

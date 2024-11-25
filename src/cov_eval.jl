"""
        FrameMat{T}

    n  block matrix with each matrix block of size r x m
    
    #Fields
    block::Vector{Matrix{T}}  # Saving the array of matrix blocks
    n::Int64                     # Number of blocks
    r::Int64                     # Size of observation per block
    m::Int64                     # Size of grid

    # Examples
    ```julia-repl
    julia> n, r, m = 200, 20, 50;
       block = [rand(r, m) for _ in 1:n];
       temp = rand(r, m);
       @time F = FrameMat(block, n, r, m, temp);
    0.000004 seconds (1 allocation: 48 bytes)
    ```
"""
mutable struct FrameMat{T}
    block::Vector{Matrix{T}}    # Saving the array of matrix blocks
    n::Int64                 # Number of blocks
    r::Int64             # Size of observation per block
    m::Int64        # Size of grid
    temp::Matrix{T} #temp buffer for mul!, size: r x m
end
Base.eltype(F::FrameMat) = eltype(F.block[1])
Base.size(F::FrameMat) = (F.n * F.r, F.m)
Base.size(F::FrameMat, i::Int) = size(F)[i]

"""
        frame_matrix(x::BlockVec{Float64}, m::Int64, kernel::ReproducingKernel)::FrameMat{Float64}

    Generate frame matrix given location grid x, grid size m.

    #Arguments
    - `x::BlockVec{Float64}`: Block vector of locations
    - `m::Int64`: Resolution of the grid
    - `kernel::ReproducingKernel`: Kernel type

    # Examples
    ```julia-repl
    julia> n, r, m = 200, 20, 50;
       γ = 20.0;
       @time x = loc_grid(n, r, λ=0);
       kernel = GaussianKernel(γ);
       @time F = frame_matrix(x, m, kernel);
    0.000204 seconds (208 allocations: 46.000 KiB)
    0.001386 seconds (403 allocations: 3.110 MiB)
    ```
"""
function frame_matrix(x::BlockVec{Float64}, m::Int64, kernel::ReproducingKernel)::FrameMat{Float64}
    # Pre-allocate
    block = [Matrix{Float64}(undef, x.r, m) for _ = 1:x.n]
    temp = Matrix{Float64}(undef, x.r, m)
    z = range(start=0.0, step=1.0 / m, length=m)

    for i in 1:x.n
        block[i] = compute_kernel(x.block[i], z, kernel)

        # temp .= x.block[i] .- z'
        # block[i] .= exp.(-γ * temp .^ 2)  # F_block is now the size r x m
    end
    return FrameMat(block, x.n, x.r, m, temp)
end

"""
        eval_sec_mom(B::BlockDiag{Float64}, x::BlockVec{Float64}, m::Int64, kernel::ReproducingKernel)::Matrix{Float64}

    Evaluate the estimated second moment with Gaussian kernel parameter γ over the m x m regular grid.

    #Arguments
    - `B::BlockDiag{Float64}`: Solution of ridge regression
    - `x::BlockVec{Float64}`: Block vector of locations to compute the frame matrix
    - `m::Int64`: Resolution of the grid
    - `kernel::ReproducingKernel`: Kernel type

    # Examples
    ```julia-repl
    julia> n, r, m = 200, 20, 50;
       kernel = GaussianKernel(20.0);
       temp = rand(r, r);
       B = BlockDiag([ones(r, r) for _ in 1:n], n, r, temp, 1);
       @time x = loc_grid(n, r, λ=0);
       @time eval_sec_mom(B, x, m, kernel);
    0.000106 seconds (208 allocations: 46.000 KiB)
    0.001917 seconds (404 allocations: 3.135 MiB)
    ```
"""
function eval_sec_mom(B::BlockDiag{Float64}, x::BlockVec{Float64}, m::Int64, kernel::ReproducingKernel)::Matrix{Float64}
    # Dimension check
    @assert B.n == x.n
    @assert B.r == x.r

    Γ = zeros(m, m)   # Pre-allocate the covariance matrix
    F_block = Matrix{Float64}(undef, x.r, m)  # Pre-allocate F_block matrix
    z = range(start=0.0, step=1.0 / m, length=m)  # Pre-allocate the range

    # Pre-allocate a temporary array to store x.block[i] .- z'
    temp = Matrix{Float64}(undef, x.r, m)

    for i in 1:B.n
        # Calculate x.block[i] - z' in-place and reuse temp_diff
        # temp1 .= x.block[i] .- z'

        # # Compute F_block in-place (avoiding allocation of intermediate arrays)
        # F_block .= exp.(-γ * temp1 .^ 2)  # F_block is now the size r x m
        F_block = compute_kernel(x.block[i], z, kernel) # F_block is now the size r x m
        temp .= B.block[i] * F_block  # temp2 is now the size r x m
        # Perform matrix multiplication in-place for Σ
        mul!(Γ, F_block', temp, 1.0, 1.0)  # Σ += F_block' * temp2
    end
    return Γ
end

"""
        eval_sec_mom(B::BlockDiag{Float64}, F::FrameMat{Float64})::Matrix{Float64}

    Given the frame matrix, plot the estimated second moment over the m x m regular grid.

    #Arguments
    - `B::BlockDiag{Float64}`: Solution of ridge regression
    - `F::FrameMat{Float64}`: Frame matrix

    # Examples
    ```julia-repl
    julia> n, r, m = 200, 20, 50;
       γ = 20.0;
       temp = rand(r, r);
       B = BlockDiag([ones(r, r) for _ in 1:n], n, r, temp, 1);
       @time x = loc_grid(n, r, λ=0);
       @time Σ1 = eval_sec_mom(B, x, m, γ);
       
       @time F = frame_matrix(x, m, γ);
       @time Σ2 = eval_sec_mom(B, F);
       
       Σ1 == Σ2
    0.000094 seconds (208 allocations: 46.000 KiB)
    0.001810 seconds (605 allocations: 4.693 MiB)
    0.001644 seconds (603 allocations: 4.660 MiB)
    0.001106 seconds (202 allocations: 1.569 MiB)
    true
    ```
"""
function eval_sec_mom(B::BlockDiag{Float64}, F::FrameMat{Float64})::Matrix{Float64}
    # Dimension check
    @assert B.n == F.n
    @assert B.r == F.r

    Γ = zeros(F.m, F.m)   # Pre-allocate the covariance matrix

    for i in 1:B.n
        F.temp .= B.block[i] * F.block[i]
        # Perform matrix multiplication in-place for Σ
        mul!(Γ, F.block[i]', F.temp, 1.0, 1.0)  # Σ += F_block' * temp2
    end
    return Γ
end

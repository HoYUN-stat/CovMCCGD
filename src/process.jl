"""
        StochasticProcess

    Abstract type for stochastic processes with 4 concrete types below.
    
    # Fields
    - `BrownianMotion`: Simple Brownian motion from time 0 to 1
        - W_t ~ N(0, t)
        - Mean function: μ(t) = 0
        - Covariance function: Σ(s, t) = min(s, t)
    - `BrownianBridge`: Brownian bridge process from time 0 to 1 with fixed endpoints 0
        - B_t = W_t - t * W_1  ∼ N(0, t(1-t))
        - Mean function: μ(t) = 0
        - Covariance function: Σ(s, t) = min(s, t) - s * t
    - `IntegratedBM`: Integrated Brownian motion from time 0 to 1
        - I_t = ∫_{0}^{t} W_{t} dt ∼ N(0, t^3/3)
        - Mean function: μ(t) = 0
        - Covariance function: Σ(s, t) = max(s,t)*min(s,t)^2/2 - min(s, t)^3/6
    - `OrnsteinUhlenbeck`: Mean-reverting process with mean reversion rate θ and long-term mean μ from time 0 to 1 with starting value 0
        - dX_t = θ(μ - X_t)dt + σdW_t
        - Mean function: μ(t) = μ *(1-e^{-θ*t})
        - Covariance function: Σ(s, t) = σ^2 / (2θ) * (e^{-θ|t-s|} - e^{-θ(t+s)})
        - Second moment: Γ(s, t) = σ^2 / (2θ) * (e^{-θ|t-s|} - e^{-θ(t+s)}) + μ^2 * (1-e^{-θ*t}) * (1-e^{-θ*s})
    ```
"""
abstract type StochasticProcess end

struct BrownianMotion <: StochasticProcess end
struct BrownianBridge <: StochasticProcess end
struct IntegratedBM <: StochasticProcess end
struct OrnsteinUhlenbeck <: StochasticProcess
    θ::Float64  # Mean reversion rate
    μ::Float64  # Long-term mean
    σ::Float64  # Volatility

    # Add a keyword constructor
    OrnsteinUhlenbeck(; θ::Float64, μ::Float64, σ::Float64) = new(θ, μ, σ)
end

"""
        true_sec_mom(process::StochasticProcess, m::Int64)::Matrix{Float64}

    Compute the true second moment matrix for the given stochastic process.

    # Arguments
    - `process::StochasticProcess`: Type of stochastic process
    - `m::Int64`: Number of grid points

    # Examples
    ```julia-repl
    julia> process = BrownianBridge()
       m = 10
       true_sec_mom(process, m)
    10×10 Matrix{Float64}:
    0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    0.0  0.09  0.08  0.07  0.06  0.05  0.04  0.03  0.02  0.01
    0.0  0.08  0.16  0.14  0.12  0.1   0.08  0.06  0.04  0.02
    0.0  0.07  0.14  0.21  0.18  0.15  0.12  0.09  0.06  0.03
    0.0  0.06  0.12  0.18  0.24  0.2   0.16  0.12  0.08  0.04
    0.0  0.05  0.1   0.15  0.2   0.25  0.2   0.15  0.1   0.05
    0.0  0.04  0.08  0.12  0.16  0.2   0.24  0.18  0.12  0.06
    0.0  0.03  0.06  0.09  0.12  0.15  0.18  0.21  0.14  0.07
    0.0  0.02  0.04  0.06  0.08  0.1   0.12  0.14  0.16  0.08
    0.0  0.01  0.02  0.03  0.04  0.05  0.06  0.07  0.08  0.09
"""
function true_sec_mom(process::StochasticProcess, m::Int64)::Matrix{Float64}
    z = range(start=0.0, step=1.0 / m, length=m)  # Pre-allocate the range

    Γ = Matrix{Float64}(undef, m, m)
    for i in 1:m
        for j in 1:m
            if process isa BrownianMotion
                Γ[i, j] = min(z[i], z[j])
            elseif process isa BrownianBridge
                Γ[i, j] = min(z[i], z[j]) - z[i] * z[j]
            elseif process isa IntegratedBM
                Γ[i, j] = max(z[i], z[j]) * min(z[i], z[j])^2 / 2 - min(z[i], z[j])^3 / 6
            elseif process isa OrnsteinUhlenbeck
                Γ[i, j] = process.σ^2 / (2 * process.θ) * (exp(-process.θ * abs(z[i] - z[j])) - exp(-process.θ * (z[i] + z[j]))) +
                          process.μ^2 * (1 - exp(-process.θ * z[i])) * (1 - exp(-process.θ * z[j]))
            else
                error("Unknown process type: $process")
            end
        end
    end
    return Γ
end

"""
        perturb_process!(y::BlockVec{Float64}, x::BlockVec{Float64}, σ::Float64; seed::Int=142, process::StochasticProcess)::BlockVec{Float64}

    Given a location grid x, generate stochastic processeses n times with perturbation level σ.

    #Arguments
    - `y::BlockVec{Float64}`: Block vector to store the process values
    - `x::BlockVec{Float64}`: Block vector of locations
    - `σ::Float64`: Perturbation level, should not be confused with σ in OU process
    - `seed::Int=142`: Seed for reproducibility
    - `process::StochasticProcess`: Type of stochastic process to use. Default is BrownianMotion.

    # Examples
    ```julia-repl
    julia> n = 200;
       r = 20;
       λ = 1; # Random locations
       σ = 0.1; # Perturbation level
       
       @time x = loc_grid(n, r, λ=0);
       @time y = perturb_process(x, σ; process=OrnsteinUhlenbeck(θ=0.1, μ=0.5, σ=0.2));
       typeof(y)
    0.000220 seconds (208 allocations: 46.000 KiB)
    0.000237 seconds (611 allocations: 133.969 KiB)
    BlockVec{Float64}
    ```
"""
function perturb_process!(y::BlockVec{Float64}, x::BlockVec{Float64}, σ::Float64; seed::Int=142, process::StochasticProcess=BrownianMotion() # Stochastic process
)::BlockVec{Float64}
    # Dimension check
    @assert x.n == y.n
    @assert x.r == y.r

    seed!(seed)  # Set seed for reproducibility

    # Pre-allocate: r+1 to sample at time 1 for BrownianBridge
    Δ = Vector{Float64}(undef, x.r + 1)
    rand_inc = Vector{Float64}(undef, x.r + 1)

    for i in 1:x.n
        Δ[1] = sqrt(x.block[i][1])
        for j in 2:x.r
            Δ[j] = sqrt(x.block[i][j] - x.block[i][j-1])
        end
        Δ[x.r+1] = sqrt(1 - x.block[i][x.r])

        rand_inc .= randn(x.r + 1) # Increment between x[j] and x[j-1], where x[0] = 0 and x[r+1]=1

        # Process-specific logic
        if process isa BrownianMotion
            for j in 1:x.r
                rand_inc[j] *= Δ[j]
            end
            cumsum!(y.block[i], rand_inc[1:x.r])
        elseif process isa BrownianBridge
            for j in 1:x.r+1 #increment vector of length r+1 to sample at time 1
                rand_inc[j] *= Δ[j]
            end
            cumsum!(Δ, rand_inc) #Recycle Δ to store the BM. Now Δ[1:r+1] is the BM
            @. y.block[i] = Δ[1:x.r] - x.block[i] * Δ[x.r+1]
        elseif process isa IntegratedBM
            for j in 1:x.r
                rand_inc[j] *= Δ[j]
            end
            cumsum!(Δ, rand_inc) #Recycle Δ to store the BM. Now Δ[1:r] is the BM
            # Let Δ[0] = 0, x.block[i][0] = 0, y.block[i][0] = 0
            # y.block[i][j] = y.block[i][j-1] + (x.block[i][j] - x.block[i][j-1]) * (Δ[j] + Δ[j-1]) / 2
            y.block[i][1] = x.block[i][1] * Δ[1] / 2
            for j in 2:x.r
                y.block[i][j] = y.block[i][j-1] + (x.block[i][j] - x.block[i][j-1]) * (Δ[j] + Δ[j-1]) / 2
            end
        elseif process isa OrnsteinUhlenbeck
            y.block[i][1] = process.θ * process.μ * x.block[i][1] + process.σ * rand_inc[1] * Δ[1]
            for j in 2:x.r
                rand_inc[j] *= Δ[j] * process.σ
                y.block[i][j] = y.block[i][j-1] + process.θ * (process.μ - y.block[i][j-1]) * Δ[j]^2 + rand_inc[j]
            end
        else
            error("Unknown process type: $process")
        end

        # Add perturbation
        perturb = randn(x.r)
        @. y.block[i] += σ * perturb
    end

    return y
end
perturb_process(x::BlockVec{Float64}, σ::Float64; seed::Int=142, process::StochasticProcess=BrownianMotion()) =
    perturb_process!(zerox(x; matrize=false), x, σ; seed, process)


function color_range(process::StochasticProcess, σ::Float64)
    if process isa BrownianMotion
        return (0, 1 + σ^2)
    elseif process isa BrownianBridge
        return (0, 0.25 + σ^2)
    elseif process isa IntegratedBM
        return (0, 1 / 3 + σ^2)
    elseif process isa OrnsteinUhlenbeck
        return (0, process.σ^2 / (2 * process.θ) + process.μ^2 + σ^2)
    else
        error("Unknown process type: $process")
    end
end
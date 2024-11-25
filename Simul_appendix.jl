# Load the necessary packages
using CairoMakie #v0.10.12
using Random: seed!

# Include the necessary Julia files
include("src/main.jl")

# Use the custom modules
using .CovMCCGD


# Set the global parameters
n::Int64 = 20      #Number of random functions
r::Int64 = 100       #Number of random locations for each function
σ::Float64 = 0.3    #Perturbation level
λ::Int64 = 0        #0 for random locations, 1 for regular locations
m::Int64 = 500    #Resolution to plot the covariance

# Generate random locations
x = loc_grid(n, r, λ=λ);
# Generate Stochastic Process
process = BrownianMotion()
y = perturb_process(x, σ; process=process);

# Plot the Brownian motions
function plot_img(m::Int64=500)
    Γ_true = true_sec_mom(process, m)
    grid = range(start=0.0, step=1.0 / m, length=m)
    fig = Figure(resolution=(1800, 1000), fontsize=40)

    ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Value", title="$process, noise level: $σ")

    # Define an array of colors for each curve
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink]
    for i in 1:n
        # Use the colors array to select a color for each curve
        lines!(ax1, x.block[i], y.block[i], color=colors[i%length(colors)+1], linewidth=1)
    end

    lines!(ax1, [NaN, NaN], [NaN, NaN], label="n = $n, r = $r, σ = $σ", color=:black, linewidth=0)
    axislegend(ax1, position=:lb)

    ax2 = Axis(fig[1, 2], xlabel="Time", ylabel="Time", title="True Second Moment")
    p2 = heatmap!(ax2, grid, grid, Γ_true, colormap=:viridis)
    Colorbar(fig[1, 3], p2)
    fig
end

plot_img(m)

# Choose the kernel
η::Float64 = 0.05    #Regularization parameter

kernel_Poly = PolynomialKernel(10, 0.2);
K_Poly = gram_matrix(x, kernel_Poly);
khatK_Poly = LazyKhatri(K_Poly, η, n, r);

# Perform the covariance smoothing
time_Poly = @elapsed B_Poly, res_Poly = matrix_cg(khatK_Poly, y; tol=1e-50, trace_residual=true);
time_Poly = round(time_Poly; digits=2)
iter_Poly = B_Poly.iter
res_Poly

# Plot the smoothed covariance
function plot_cov(m::Int64=500)
    Γ_true = true_sec_mom(process, m)
    Γ_Poly = eval_sec_mom(B_Poly, x, m, kernel_Poly)

    frame_grid = range(start=0.0, step=1.0 / m, length=m)
    fig = Figure(resolution=(1800, 2000), fontsize=40)

    gproc = fig[1, 1:2] = GridLayout()
    gtrue = fig[2, 1] = GridLayout()
    gPoly = fig[2, 2] = GridLayout()

    rowsize!(fig.layout, 1, Auto(0.5))

    # Plot the Stochastic Process
    axproc = Axis(gproc[1, 1], xlabel="Time", ylabel="Value", title="$process, noise level: $σ")
    xlims!(axproc, 0, 1)
    axproc.xticks = 0:0.2:1
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink]
    for i in 1:n
        # Use the colors array to select a color for each curve
        lines!(axproc, x.block[i], y.block[i], color=colors[i%length(colors)+1], linewidth=1)
    end

    lines!(axproc, [NaN, NaN], [NaN, NaN], label="n = $n, r = $r, σ = $σ", color=:black, linewidth=0)
    axislegend(axproc, position=:lb)

    # Plot the true second moments
    axtrue = Axis(gtrue[1, 1], title="True Second Moment")
    ptrue = heatmap!(axtrue, frame_grid, frame_grid, Γ_true, colormap=:viridis)
    xlims!(axtrue, 0, 1)
    ylims!(axtrue, 0, 1)
    axtrue.xticks = 0:0.2:1
    axtrue.yticks = 0:0.2:1
    Colorbar(gtrue[1, 2], ptrue)
    rowsize!(gtrue, 1, Aspect(1, 1.0)) # Aspect ratio 1:1

    # Polynomial kernel
    axPoly = Axis(gPoly[1, 1], title="$kernel_Poly")
    pPoly = heatmap!(axPoly, frame_grid, frame_grid, Γ_Poly, colormap=:viridis)
    xlims!(axPoly, 0, 1)
    ylims!(axPoly, 0, 1)
    axPoly.xticks = 0:0.2:1
    axPoly.yticks = 0:0.2:1
    Colorbar(gPoly[1, 2], pPoly)
    rowsize!(gPoly, 1, Aspect(1, 1.0))

    # Residual plot
    gres = fig[3, 1:2] = GridLayout()
    ax_res_Poly = Axis(gres[1, 1], xlabel="Iteration", ylabel="Residual")
    lines!(ax_res_Poly, 1:iter_Poly, res_Poly[1:end-1], color=:blue, linewidth=2)
    lines!(ax_res_Poly, [NaN, NaN], [NaN, NaN], label="$time_Poly sec", color=:black, linewidth=0)
    axislegend(ax_res_Poly, position=:lb)

    rowsize!(fig.layout, 1, Auto(0.5))
    rowsize!(fig.layout, 3, Auto(0.5))

    fig

    # # Save the figure
    file_name = "$process,,$n,R,,$r"
    file_folder = "simul/"
    file_date = Dates.format(now(), "yymmddHHMM")
    CairoMakie.save(file_folder * file_name * "_" * file_date * ".png", fig)
end


plot_cov(m)

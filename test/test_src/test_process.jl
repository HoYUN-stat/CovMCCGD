@testset "Color Range Tests" begin
    process = BrownianMotion()
    σ = 0.1
    @test color_range(process, σ) == (0, 1 + σ^2)
end


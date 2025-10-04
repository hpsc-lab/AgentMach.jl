using Test
using CodexPar

@testset "CodexPar.jl" begin
    @test greet() == "Hello, world! Welcome to CodexPar."
    @test greet("team") == "Hello, team! Welcome to CodexPar."
end

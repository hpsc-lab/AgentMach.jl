using Test
using CodexPar

@testset "Compressible Euler" begin
    nx, ny = 32, 32
    lengths = (2.0, 2.0)
    origin = (-1.0, -1.0)
    prob = setup_compressible_euler_problem(nx, ny;
                                            lengths = lengths,
                                            origin = origin,
                                            gamma = 1.4)

    init(x, y) = (; rho = 1.0 + 0.2 * sinpi(x),
                   v1 = 0.1 * cospi(y),
                   v2 = 0.05 * sinpi(x + y),
                   p = 1.0 + 0.1 * cospi(x))

    state = CompressibleEulerState(prob; init = init, T = Float64)

    dt = stable_timestep(prob, state; cfl = 0.4)
    @test isfinite(dt) && dt > 0

    mass_before = sum(solution(state)[1, :, :])
    energy_before = sum(solution(state)[4, :, :])

    run_compressible_euler!(state, prob; steps = 4, dt = dt)

    u = solution(state)
    @test !any(isnan, u)
    @test minimum(u[1, :, :]) > 0

    total_mass = sum(u[1, :, :])
    total_energy = sum(u[4, :, :])

    tol_mass = 1e-12 * length(u)
    tol_energy = 1e-9 * length(u)
    @test abs(total_mass - mass_before) <= tol_mass
    @test abs(total_energy - energy_before) <= tol_energy
end

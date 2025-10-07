using Test
using CodexMach

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

    prim = primitive_variables(prob, state)
    @test size(prim.rho) == (nx, ny)
    @test size(prim.u) == (nx, ny)
    @test size(prim.v) == (nx, ny)
    @test size(prim.p) == (nx, ny)

    sample_i, sample_j = 5, 6
    @test isapprox(prim.rho[sample_i, sample_j], u[1, sample_i, sample_j]; atol = 1e-12)
    @test isapprox(prim.u[sample_i, sample_j], u[2, sample_i, sample_j] / u[1, sample_i, sample_j]; atol = 1e-10)
    @test isapprox(prim.v[sample_i, sample_j], u[3, sample_i, sample_j] / u[1, sample_i, sample_j]; atol = 1e-10)

    rho_buf = similar(prim.rho)
    u_buf = similar(prim.u)
    v_buf = similar(prim.v)
    p_buf = similar(prim.p)
    prim_buf = primitive_variables(prob, solution(state);
                                   rho_out = rho_buf,
                                   u_out = u_buf,
                                   v_out = v_buf,
                                   p_out = p_buf)
    @test prim_buf.rho === rho_buf
    @test prim_buf.u === u_buf
    @test prim_buf.v === v_buf
    @test prim_buf.p === p_buf
    @test all(isapprox.(prim_buf.rho, prim.rho; atol = 1e-12))
    @test all(isapprox.(prim_buf.u, prim.u; atol = 1e-12))
    @test all(isapprox.(prim_buf.v, prim.v; atol = 1e-12))
    @test all(isapprox.(prim_buf.p, prim.p; atol = 1e-12))
end

@testset "Compressible Euler KA backend" begin
    nx, ny = 16, 16
    prob = setup_compressible_euler_problem(nx, ny;
                                            lengths = (1.0, 1.0),
                                            origin = (0.0, 0.0),
                                            gamma = 1.4)

    init(x, y) = (; rho = 1.0 + 0.05 * sinpi(x),
                   v1 = 0.02 * cospi(y),
                   v2 = 0.01 * sinpi(x + y),
                   p = 1.0 + 0.02 * cospi(x))

    serial_state = CompressibleEulerState(prob; init = init, T = Float64)
    ka_state = CompressibleEulerState(prob; init = init, T = Float64,
                                      backend = KernelAbstractionsBackend(:cpu))

    steps = 3
    dt = stable_timestep(prob, serial_state; cfl = 0.4)
    for _ in 1:steps
        rk2_step!(serial_state, prob, dt)
        rk2_step!(ka_state, prob, dt)
    end

    serial_u = solution(serial_state)
    ka_u = solution(ka_state)
    @test all(isapprox.(ka_u, serial_u; atol = 1e-10))
end

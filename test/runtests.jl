using Test
using CodexPar

@testset "CodexPar.jl" begin
    @test greet() == "Hello, world! Welcome to CodexPar."
    @test greet("team") == "Hello, team! Welcome to CodexPar."
end

@testset "StructuredMesh" begin
    mesh = StructuredMesh(4, 2; lengths = (2.0, 1.0), origin = (-1.0, 0.5))
    @test size(mesh) == (4, 2)
    @test spacing(mesh) == (0.5, 0.5)
    @test origin(mesh) == (-1.0, 0.5)

    centers_x, centers_y = CodexPar.cell_centers(mesh)
    @test isapprox(first(centers_x), -0.75; atol = 1e-12)
    @test isapprox(last(centers_x), 0.75; atol = 1e-12)
    @test isapprox(first(centers_y), 0.75; atol = 1e-12)
    @test isapprox(last(centers_y), 1.25; atol = 1e-12)
end

@testset "BoundaryConditions" begin
    bc = PeriodicBoundaryConditions(; x = true, y = false)
    @test CodexPar.is_periodic(bc, 1)
    @test !CodexPar.is_periodic(bc, 2)

    @test_throws ArgumentError CodexPar.is_periodic(bc, 0)
    @test_throws ArgumentError CodexPar.is_periodic(bc, 3)
end

@testset "LinearAdvection setup" begin
    problem = setup_linear_advection_problem(8, 4;
                                             lengths = (2.0, 1.0),
                                             origin = (-1.0, -0.5),
                                             velocity = (0.5, -0.25))

    @test problem isa LinearAdvectionProblem
    mesh = CodexPar.mesh(problem)
    bc = CodexPar.boundary_conditions(problem)
    eq = CodexPar.pde(problem)

    @test spacing(mesh) == (0.25, 0.25)
    @test CodexPar.periodic_axes(bc) == (true, true)
    @test CodexPar.velocity(eq) == (0.5, -0.25)
end

@testset "LinearAdvection state" begin
    problem = setup_linear_advection_problem(8, 4; velocity = (1.0, 0.0))
    state = LinearAdvectionState(problem; init = 2.0)
    u = solution(state)
    ws = workspace(state)

    @test size(u) == (8, 4)
    @test all(u .== 2.0)
    @test size(ws.k1) == size(u)
    @test size(ws.k2) == size(u)
    @test size(ws.stage) == size(u)

    mesh = CodexPar.mesh(problem)
    init_fun(x, y) = x + y
    state_fun = LinearAdvectionState(problem; init = init_fun)
    u_fun = solution(state_fun)
    centers_x, centers_y = CodexPar.cell_centers(mesh)
    @inbounds for j in 1:size(u_fun, 2), i in 1:size(u_fun, 1)
        @test isapprox(u_fun[i, j], init_fun(centers_x[i], centers_y[j]); atol = 1e-12)
    end

    fill!(u, 1.0)
    compute_rhs!(ws.k1, u, problem)
    @test all(isapprox.(ws.k1, 0.0; atol = 1e-12))
end

@testset "LinearAdvection RK2" begin
    nx = 64
    problem = setup_linear_advection_problem(nx, 1; velocity = (1.0, 0.0))
    mesh = CodexPar.mesh(problem)
    centers_x, _ = CodexPar.cell_centers(mesh)
    init_fun(x, _) = sin(2pi * x)
    state = LinearAdvectionState(problem; init = init_fun)
    dx, _ = spacing(mesh)
    dt = 0.25 * dx

    rk2_step!(state, problem, dt)

    expected = [sin(2pi * (x - dt)) for x in centers_x]
    u = solution(state)
    @inbounds for i in 1:nx
        @test isapprox(u[i, 1], expected[i]; atol = 1e-3)
    end
end

@testset "CFL guidance" begin
    problem = setup_linear_advection_problem(16, 8; velocity = (2.0, -1.0))
    dt = stable_timestep(problem; cfl = 0.75)
    @test isapprox(cfl_number(problem, dt), 0.75; atol = 1e-12)

    half_dt = 0.5 * dt
    @test isapprox(cfl_number(problem, half_dt), 0.375; atol = 1e-12)

    zero_vel_problem = setup_linear_advection_problem(4, 4; velocity = (0.0, 0.0))
    @test isinf(stable_timestep(zero_vel_problem))

    @test_throws ArgumentError stable_timestep(problem; cfl = 0.0)
end

@testset "LinearAdvection driver" begin
    nx = 64
    problem = setup_linear_advection_problem(nx, 1; velocity = (1.0, 0.0))
    mesh = CodexPar.mesh(problem)
    centers_x, _ = CodexPar.cell_centers(mesh)
    init_fun(x, _) = sin(2pi * x)
    dx, _ = spacing(mesh)
    dt = 0.25 * dx

    state = LinearAdvectionState(problem; init = init_fun)
    result = run_linear_advection!(state, problem; steps = 4, dt = dt, record_cfl = true)
    expected = [sin(2pi * (x - 4 * dt)) for x in centers_x]
    u = solution(state)
    @inbounds for i in 1:nx
        @test isapprox(u[i, 1], expected[i]; atol = 1e-3)
    end
    @test length(result.cfl_history) == 4
    @test all(isapprox.(result.cfl_history, result.cfl; atol = 1e-12))

    state_auto = LinearAdvectionState(problem; init = init_fun)
    result_auto = run_linear_advection!(state_auto, problem; steps = 2, cfl_target = 0.5)
    @test isapprox(result_auto.dt, stable_timestep(problem; cfl = 0.5); atol = 1e-12)
    @test isapprox(result_auto.cfl, cfl_number(problem, result_auto.dt); atol = 1e-12)

    @test_throws ArgumentError run_linear_advection!(state_auto, problem; steps = 0, dt = dt)
end

@testset "Examples" begin
    include(joinpath(@__DIR__, "..", "examples", "linear_advection_demo.jl"))
    include(joinpath(@__DIR__, "..", "examples", "plot_linear_advection.jl"))

    result = run_linear_advection_demo(; nx = 16, ny = 1, steps = 2, cfl = 0.3, sample_every = 1)
    @test result.steps == 2
    @test result.dt > 0
    @test result.final_time ≈ 2 * result.dt
    @test !isempty(result.diagnostics)
    @test result.cfl_history == fill(result.cfl, 2)

    mktempdir() do dir
        diag_path = joinpath(dir, "diagnostics.csv")
        state_path = joinpath(dir, "state.csv")
        exported = run_linear_advection_demo(; nx = 8, ny = 1, steps = 3, cfl = 0.25,
                                             sample_every = 1,
                                             diagnostics_path = diag_path,
                                             state_path = state_path)
        @test isfile(diag_path)
        @test isfile(state_path)
        diag_lines = readlines(diag_path)
        @test first(diag_lines) == "step,time,rms,cfl"
        @test length(diag_lines) == length(exported.diagnostic_records) + 1
        state_lines = readlines(state_path)
        @test first(state_lines) == "i,j,x,y,u"
        @test length(state_lines) == length(exported.final_state) + 1

        plot_path = joinpath(dir, "plot.png")
        try
            out = plot_linear_advection_csv(diag_path; state_path = state_path, output_path = plot_path)
            @test out == plot_path
            @test isfile(plot_path)
        catch err
            @test err isa ArgumentError
        end
    end
end

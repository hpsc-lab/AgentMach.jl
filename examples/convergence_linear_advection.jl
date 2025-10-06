#!/usr/bin/env julia
using CodexMach
using Printf

"""
    run_convergence_study(; base_resolution=(16, 16), levels=5,
                             lengths=(1.0, 1.0), velocity=(1.0, 0.5),
                             final_time=0.5, cfl=0.45)

Execute a grid refinement study for periodic linear advection with a sinusoidal
initial condition on a square domain. Prints the L₂ error for each resolution,
experimental orders of convergence (EOC), and the average EOC across all levels.
"""
function run_convergence_study(; base_resolution::Tuple{Int,Int} = (16, 16),
                                levels::Int = 5,
                                lengths::Tuple{<:Real,<:Real} = (1.0, 1.0),
                                velocity::Tuple{<:Real,<:Real} = (1.0, 0.5),
                                final_time::Real = 0.5,
                                cfl::Real = 0.45)
    @assert levels >= 2 "Need at least two levels to measure convergence"

    Lx, Ly = float(lengths[1]), float(lengths[2])
    vel = (float(velocity[1]), float(velocity[2]))
    nx0, ny0 = base_resolution

    errors = Float64[]
    hs = Float64[]
    eocs = Float64[]

    println("Linear advection convergence study (sinusoidal initial condition)")
    println(" level    nx    ny        dt          L2 error        EOC")

    init = _sinusoidal_initializer(Lx, Ly)

    for level in 0:levels-1
        nx = nx0 * 2^level
        ny = ny0 * 2^level
        problem = setup_linear_advection_problem(nx, ny;
                                                 lengths = (Lx, Ly),
                                                 velocity = vel)
        state = LinearAdvectionState(problem; init = init)

        dt_stable = stable_timestep(problem; cfl = cfl)
        steps = max(1, ceil(Int, final_time / dt_stable))
        dt = final_time / steps

        run_linear_advection!(state, problem; steps = steps, dt = dt)

        u_num = solution(state)
        u_exact = _exact_solution(problem, vel, final_time, u_num)

        err = sqrt(sum(abs2, u_num .- u_exact) / length(u_num))
        push!(errors, err)

        dx = Lx / nx
        push!(hs, dx)

        if level == 0
            @printf(" %5d %5d %5d  %10.6f  %13.6e      %s\n", level, nx, ny, dt, err, "---")
        else
            eoc = log(errors[end-1] / err) / log(hs[end-1] / dx)
            push!(eocs, eoc)
            @printf(" %5d %5d %5d  %10.6f  %13.6e  %8.4f\n", level, nx, ny, dt, err, eoc)
        end
    end

    if !isempty(eocs)
        avg_eoc = sum(eocs) / length(eocs)
        @printf("Average EOC: %.4f\n", avg_eoc)
    end

    return (; lengths = (Lx, Ly), velocity = vel, final_time = final_time,
            resolutions = [(nx0 * 2^lvl, ny0 * 2^lvl) for lvl in 0:levels-1],
            errors = errors, eocs = eocs)
end

function _sinusoidal_initializer(Lx::Float64, Ly::Float64)
    function init(x, y)
        return sin(2pi * x / Lx) * sin(2pi * y / Ly)
    end
    return init
end

function _exact_solution(problem::LinearAdvectionProblem, velocity, t, template)
    mesh_obj = mesh(problem)
    centers_x, centers_y = cell_centers(mesh_obj)
    nx, ny = size(mesh_obj)
    dx, dy = spacing(mesh_obj)
    Lx = dx * nx
    Ly = dy * ny
    ax, ay = velocity

    u = similar(template)
    @inbounds for j in 1:ny
        y = centers_y[j]
        for i in 1:nx
            x = centers_x[i]
            phase_x = 2pi * (x - ax * t) / Lx
            phase_y = 2pi * (y - ay * t) / Ly
            u[i, j] = sin(phase_x) * sin(phase_y)
        end
    end
    return u
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_study()
end

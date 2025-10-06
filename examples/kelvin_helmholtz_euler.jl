#!/usr/bin/env julia
using CodexPar
using Printf
using Plots

"""
    run_kelvin_helmholtz(; nx=256, ny=256, gamma=1.4, final_time=1.5,
                           cfl=0.45, T=Float32, log_every=25,
                           diagnostics_path=nothing,
                           pdf_path=nothing)

Simulate the Kelvin-Helmholtz instability for the 2D compressible Euler system on
[-1, 1]² with periodic boundaries. Returns integration metadata and (optionally)
writes per-step diagnostics.
"""
function run_kelvin_helmholtz(; nx::Int = 256,
                               ny::Int = 256,
                               gamma::Real = 1.4,
                               final_time::Real = 1.5,
                               cfl::Real = 0.45,
                               T::Type = Float32,
                               log_every::Integer = 25,
                                 diagnostics_path::Union{Nothing,AbstractString} = nothing,
                                 pdf_path::Union{Nothing,AbstractString} = nothing)
    nx >= 3 || throw(ArgumentError("nx must be at least 3"))
    ny >= 3 || throw(ArgumentError("ny must be at least 3"))
    final_time > 0 || throw(ArgumentError("final_time must be positive"))

    problem = setup_compressible_euler_problem(nx, ny;
                                               lengths = (2.0, 2.0),
                                               origin = (-1.0, -1.0),
                                               gamma = gamma,
                                               boundary_conditions = PeriodicBoundaryConditions())

    init = _kelvin_helmholtz_initializer(T)
    state = CompressibleEulerState(problem; T = T, init = init)

    time = 0.0
    step = 0
    last_cfl = NaN
    records = diagnostics_path === nothing ? nothing : Vector{NamedTuple{(:step, :time, :cfl, :kinetic_energy),NTuple{4,Float64}}}()
    prim_buffers = primitive_variables(problem, solution(state))

    while time < final_time
        dt = stable_timestep(problem, state; cfl = cfl)
        if time + dt > final_time
            dt = final_time - time
        end
        dt > 0 || throw(ArgumentError("Encountered non-positive time step"))

        rk2_step!(state, problem, dt)
        time += dt
        step += 1

        cfl_val = cfl_number(problem, state, dt)
        kinetic = _volume_average_kinetic_energy(state, problem, prim_buffers)
        last_cfl = cfl_val

        if diagnostics_path !== nothing
            push!(records, (; step = float(step), time = time, cfl = cfl_val, kinetic_energy = kinetic))
        end

        if log_every > 0 && step % log_every == 0
            @info "Kelvin-Helmholtz" step=step time=time dt=dt cfl=cfl_val kinetic_energy=kinetic
        end
    end

    if diagnostics_path !== nothing
        _write_khi_diagnostics(diagnostics_path, records)
    end

    if pdf_path !== nothing
        _write_khi_pdf(pdf_path, problem, state, prim_buffers)
    end

    return (; final_time = time,
            steps = step,
            cfl_last = last_cfl,
            state = state,
            problem = problem,
            diagnostics = records,
            pdf_path = pdf_path)
end

function _kelvin_helmholtz_initializer(::Type{T}) where {T}
    slope = T(15)
    offset = T(7.5)
    function init(x, y)
        xx = T(x)
        yy = T(y)
        B = tanh(slope * yy + offset) - tanh(slope * yy - offset)
        rho = T(0.5) + T(0.75) * B
        v1 = T(0.5) * (B - T(1))
        v2 = T(0.1) * sinpi(T(2) * xx)
        p = T(1)
        return (; rho = rho, v1 = v1, v2 = v2, p = p)
    end
    return init
end

function _volume_average_kinetic_energy(state::CompressibleEulerState,
                                        problem::CompressibleEulerProblem,
                                        buffers)
    prim = primitive_variables(problem, solution(state);
                                rho_out = buffers.rho,
                                u_out = buffers.u,
                                v_out = buffers.v,
                                p_out = buffers.p)

    nx, ny = size(prim.rho)
    total = zero(eltype(prim.rho))
    half = convert(eltype(prim.rho), 0.5)
    @inbounds for j in 1:ny, i in 1:nx
        vel2 = prim.u[i, j]^2 + prim.v[i, j]^2
        total += half * prim.rho[i, j] * vel2
    end
    return float(total) / (nx * ny)
end

function _write_khi_diagnostics(path::AbstractString,
                                records::Vector{NamedTuple{(:step, :time, :cfl, :kinetic_energy),NTuple{4,Float64}}})
    open(path, "w") do io
        println(io, "step,time,cfl,kinetic_energy")
        for r in records
            @printf(io, "%d,%.8f,%.6f,%.8e\n", Int(r.step), r.time, r.cfl, r.kinetic_energy)
        end
    end
    return path
end

function _write_khi_pdf(path::AbstractString,
                        problem::CompressibleEulerProblem,
                        state::CompressibleEulerState,
                        buffers)
    mesh_obj = mesh(problem)
    centers_x, centers_y = cell_centers(mesh_obj)
    prim = primitive_variables(problem, solution(state);
                                rho_out = buffers.rho,
                                u_out = buffers.u,
                                v_out = buffers.v,
                                p_out = buffers.p)

    heatmap(centers_x,
            centers_y,
            prim.rho';
            xlabel = "x",
            ylabel = "y",
            title = "Kelvin-Helmholtz density",
            colorbar = true,
            aspect_ratio = 1,
            color = cgrad(:coolwarm))

    savefig(path)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_kelvin_helmholtz()
end

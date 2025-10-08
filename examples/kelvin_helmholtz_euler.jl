#!/usr/bin/env julia
using AgentMach
using Printf
using Plots

"""
    run_kelvin_helmholtz(; nx=256, ny=256, gamma=1.4, final_time=1.5,
                           cfl=0.45, T=Float32, backend=default_backend(), log_every=25,
                           diagnostics_path=nothing,
                           pdf_path=nothing,
                           animation_path=nothing,
                           animation_every=10,
                           animation_fps=24,
                           progress_interval=10.0)

Simulate the Kelvin-Helmholtz instability for the 2D compressible Euler system on
[-1, 1]² with periodic boundaries. Returns integration metadata and (optionally)
writes per-step diagnostics.
"""
function run_kelvin_helmholtz(; nx::Int = 256,
                               ny::Int = 256,
                               gamma::Real = 1.4,
                               final_time::Real = 1.5,
                               cfl::Real = 0.45,
                               T::Union{Type,Nothing} = Float32,
                               backend::Union{ExecutionBackend,Symbol} = default_backend(),
                               log_every::Integer = 25,
                               diagnostics_path::Union{Nothing,AbstractString} = nothing,
                               pdf_path::Union{Nothing,AbstractString} = nothing,
                               animation_path::Union{Nothing,AbstractString} = nothing,
                               animation_every::Integer = 10,
                               animation_fps::Integer = 24,
                               progress_interval::Real = 10.0)
    nx >= 3 || throw(ArgumentError("nx must be at least 3"))
    ny >= 3 || throw(ArgumentError("ny must be at least 3"))
    final_time > 0 || throw(ArgumentError("final_time must be positive"))
    animation_every > 0 || throw(ArgumentError("animation_every must be positive"))
    animation_fps > 0 || throw(ArgumentError("animation_fps must be positive"))

    problem = setup_compressible_euler_problem(nx, ny;
                                               lengths = (2.0, 2.0),
                                               origin = (-1.0, -1.0),
                                               gamma = gamma,
                                               boundary_conditions = PeriodicBoundaryConditions())

    backend_obj = _resolve_example_backend(backend)
    state_T = isnothing(T) ? _default_state_eltype(backend_obj) : T

    init = _kelvin_helmholtz_initializer(state_T)
    state = CompressibleEulerState(problem; T = state_T, init = init, backend = backend_obj)

    sim_time = 0.0
    step = 0
    last_cfl = NaN
    records = diagnostics_path === nothing ? nothing : Vector{NamedTuple{(:step, :time, :cfl, :kinetic_energy),NTuple{4,Float64}}}()
    prim_buffers = primitive_variables(problem, solution(state);
                                       backend = AgentMach.backend(state))
    last_progress = time()
    centers_x, centers_y = cell_centers(mesh(problem))
    animation_obj = animation_path === nothing ? nothing : Animation()

    while sim_time < final_time
        dt = stable_timestep(problem, state; cfl = cfl)
        if sim_time + dt > final_time
            dt = final_time - sim_time
        end
        dt > 0 || throw(ArgumentError("Encountered non-positive time step"))

        rk2_step!(state, problem, dt; t = sim_time)
        sim_time += dt
        step += 1

        cfl_val = cfl_number(problem, state, dt)
        kinetic, prim = _volume_average_kinetic_energy(state, problem, prim_buffers)
        last_cfl = cfl_val

        if diagnostics_path !== nothing
            push!(records, (; step = float(step), time = sim_time, cfl = cfl_val, kinetic_energy = kinetic))
        end

        if log_every > 0 && step % log_every == 0
            @info "Kelvin-Helmholtz" step=step time=sim_time dt=dt cfl=cfl_val kinetic_energy=kinetic
        end

        now = time()
        if progress_interval > 0 && (now - last_progress) >= progress_interval
            @info "Kelvin-Helmholtz progress" step=step sim_time=sim_time dt=dt cfl=cfl_val
            last_progress = now
        end

        if animation_obj !== nothing && (step % animation_every == 0 || isapprox(sim_time, final_time; atol = eps(sim_time)))
            density_plot = _density_plot(centers_x, centers_y, prim.rho;
                                         title = @sprintf("Density t = %.3f", sim_time))
            frame(animation_obj, density_plot)
        end
    end

    if diagnostics_path !== nothing
        _write_khi_diagnostics(diagnostics_path, records)
    end

    if pdf_path !== nothing
        _write_khi_pdf(pdf_path, problem, state, prim_buffers, centers_x, centers_y)
    end

    if animation_obj !== nothing
        _finalize_animation(animation_obj, animation_path, animation_fps)
    end

    return (; final_time = sim_time,
            steps = step,
            cfl_last = last_cfl,
            state = state,
            problem = problem,
            diagnostics = records,
            pdf_path = pdf_path,
            animation_path = animation_path)
end

function _resolve_example_backend(spec::ExecutionBackend)
    return spec
end

function _resolve_example_backend(spec::Symbol)
    return KernelAbstractionsBackend(spec)
end

function _default_state_eltype(::ExecutionBackend)
    return Float64
end

function _default_state_eltype(::KernelAbstractionsBackend)
    return Float32
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
                                backend = AgentMach.backend(state),
                                rho_out = buffers.rho,
                                u_out = buffers.u,
                                v_out = buffers.v,
                                p_out = buffers.p)

    ρ = prim.rho
    u = prim.u
    v = prim.v
    nx, ny = size(ρ)
    if ρ isa Array
        total = zero(eltype(ρ))
        half = convert(eltype(ρ), 0.5)
        @inbounds for j in 1:ny, i in 1:nx
            vel2 = u[i, j]^2 + v[i, j]^2
            total += half * ρ[i, j] * vel2
        end
        return float(total) / (nx * ny), prim
    else
        T = eltype(ρ)
        half = inv(convert(T, 2))
        vel2 = u .* u .+ v .* v
        total = sum(ρ .* vel2 .* half)
        return float(total) / (nx * ny), prim
    end
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
                        buffers,
                        centers_x,
                        centers_y)
    prim = primitive_variables(problem, solution(state);
                                backend = AgentMach.backend(state),
                                rho_out = buffers.rho,
                                u_out = buffers.u,
                                v_out = buffers.v,
                                p_out = buffers.p)

    fig = _density_plot(centers_x, centers_y, prim.rho; title = "Kelvin-Helmholtz density")
    savefig(fig, path)
    return path
end

function _density_plot(centers_x, centers_y, density; title::AbstractString)
    dens = density isa Array ? density : Array(density)
    heatmap(centers_x,
            centers_y,
            dens';
            xlabel = "x",
            ylabel = "y",
            title = title,
            colorbar = true,
            aspect_ratio = 1,
            color = cgrad(:coolwarm))
end

function _finalize_animation(anim::Animation, path::AbstractString, fps::Integer)
    ext = lowercase(splitext(path)[2])
    if ext == ".mp4"
        mp4(anim, path; fps = fps)
    elseif ext == ".gif"
        gif(anim, path; fps = fps)
    else
        @warn "Unknown animation extension; defaulting to GIF" path=path
        gif(anim, path * ".gif"; fps = fps)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_kelvin_helmholtz()
end

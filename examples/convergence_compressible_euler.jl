#!/usr/bin/env julia
using CodexMach
using Printf

const DEFAULT_MMS_PARAMS = (
    rho0 = 1.0,
    rho_amp = 0.1,
    u0 = 0.2,
    u_amp = 0.05,
    v0 = -0.15,
    v_amp = 0.04,
    p0 = 1.0,
    p_amp = 0.05,
    omega = 1.2,
)

"""
    run_euler_convergence_study(; base_resolution=(64, 64), levels=4,
                                   lengths=(1.0, 1.0), gamma=1.4,
                                   final_time=0.05, cfl=0.3,
                                   backend=default_backend(),
                                   state_eltype=nothing,
                                   params=DEFAULT_MMS_PARAMS)

Perform a grid refinement study for the compressible Euler equations using a
time-dependent manufactured solution with sinusoidal structure. The volumetric
source term is chosen so that the analytical state remains an exact solution,
allowing L₂ error estimates for each conserved component `(ρ, ρu, ρv, E)` at the
requested `final_time`.
"""
function run_euler_convergence_study(; base_resolution::Tuple{Int,Int} = (64, 64),
                                      levels::Int = 4,
                                      lengths::Tuple{<:Real,<:Real} = (1.0, 1.0),
                                      gamma::Real = 1.4,
                                      final_time::Real = 0.05,
                                      cfl::Real = 0.3,
                                      backend::Union{ExecutionBackend,Symbol} = default_backend(),
                                      state_eltype::Union{Nothing,Type} = nothing,
                                      params = DEFAULT_MMS_PARAMS)
    @assert levels >= 2 "Need at least two levels to measure convergence"

    Lx, Ly = float(lengths[1]), float(lengths[2])
    kx = 2π / Lx
    ky = 2π / Ly
    γ = float(gamma)

    backend_obj = _resolve_example_backend(backend)
    state_T = isnothing(state_eltype) ? _default_state_eltype(backend_obj) : state_eltype
    source_cb = _manufactured_source_factory(params, kx, ky, γ)

    nx0, ny0 = base_resolution
    hs = Float64[]
    var_labels = ("rho", "rhou", "rhov", "E")
    errors = [Float64[] for _ in var_labels]
    eocs = [Float64[] for _ in var_labels]

    println("Compressible Euler convergence study (manufactured sinusoid)")
    header = IOBuffer()
    print(header, " level    nx    ny        dt")
    for lbl in var_labels
        @printf(header, "    L2(%5s)    EOC", lbl)
    end
    println(String(take!(header)))

    for level in 0:levels-1
        nx = nx0 * 2^level
        ny = ny0 * 2^level

        problem = setup_compressible_euler_problem(nx, ny;
                                                   lengths = (Lx, Ly),
                                                   gamma = γ,
                                                   source = source_cb)

        init = _manufactured_initializer(params, kx, ky)
        state = CompressibleEulerState(problem;
                                       T = state_T,
                                       init = init,
                                       backend = backend_obj)

        dt_stable = stable_timestep(problem, state; cfl = cfl)
        steps = max(1, ceil(Int, final_time / dt_stable))
        dt = final_time / steps

        run_compressible_euler!(state, problem;
                                 steps = steps,
                                 dt = dt,
                                 adapt_dt = false,
                                 show_timers = false)

        sol = solution(state)
        rho_num = Array(component(sol, 1))
        rhou_num = Array(component(sol, 2))
        rhov_num = Array(component(sol, 3))
        E_num = Array(component(sol, 4))

        mesh_obj = mesh(problem)
        centers_x, centers_y = cell_centers(mesh_obj)
        nxm, nym = size(mesh_obj)
        @assert nxm == nx && nym == ny

        acc = zeros(Float64, length(var_labels))
        t_eval = dt * steps
        total_cells = nx * ny
        @inbounds for j in 1:ny
            y = centers_y[j]
            for i in 1:nx
                x = centers_x[i]
                ρ_exact, u_exact, v_exact, p_exact =
                    _manufactured_primitives(params, kx, ky, x, y, t_eval)
                ρu_exact = ρ_exact * u_exact
                ρv_exact = ρ_exact * v_exact
                E_exact = p_exact / (γ - 1) + 0.5 * ρ_exact * (u_exact^2 + v_exact^2)

                δρ = rho_num[i, j] - ρ_exact
                δρu = rhou_num[i, j] - ρu_exact
                δρv = rhov_num[i, j] - ρv_exact
                δE = E_num[i, j] - E_exact

                acc[1] += δρ^2
                acc[2] += δρu^2
                acc[3] += δρv^2
                acc[4] += δE^2
            end
        end

        errs = sqrt.(acc ./ total_cells)
        for (store, val) in zip(errors, errs)
            push!(store, val)
        end

        dx = Lx / nx
        push!(hs, dx)

        row_prefix = IOBuffer()
        @printf(row_prefix, " %5d %6d %6d  %10.6f", level, nx, ny, dt)
        if level == 0
            for err in errs
                @printf(row_prefix, "  %12.4e  %8s", err, "---")
            end
        else
            ratio = log(hs[end-1] / dx)
            for (err_hist, eoc_hist, err) in zip(errors, eocs, errs)
                prev_err = err_hist[end-1]
                curr_err = err_hist[end]
                eoc_val = log(prev_err / curr_err) / ratio
                push!(eoc_hist, eoc_val)
                @printf(row_prefix, "  %12.4e  %8.4f", err, eoc_val)
            end
        end
        println(String(take!(row_prefix)))
    end

    avg_row = IOBuffer()
    @printf(avg_row, " %5s %6s %6s  %10s", "avg", "", "", "")
    for vec in eocs
        if isempty(vec)
            @printf(avg_row, "  %12s  %8s", "", "---")
        else
            avg_val = sum(vec) / length(vec)
            @printf(avg_row, "  %12s  %8.4f", "", avg_val)
        end
    end
    println(String(take!(avg_row)))

    return (; lengths = (Lx, Ly),
            gamma = γ,
            final_time = final_time,
            resolutions = [(nx0 * 2^lvl, ny0 * 2^lvl) for lvl in 0:levels-1],
            errors = (rho = errors[1], rhou = errors[2], rhov = errors[3], energy = errors[4]),
            eocs = (rho = eocs[1], rhou = eocs[2], rhov = eocs[3], energy = eocs[4]))
end

function _manufactured_initializer(params, kx, ky)
    function init(x, y)
        ρ, u, v, p = _manufactured_primitives(params, kx, ky, x, y, 0.0)
        return (; rho = ρ, v1 = u, v2 = v, p = p)
    end
    return init
end

function _manufactured_source_factory(params, kx, ky, γ)
    function source!(du, _, problem, t)
        mesh_obj = mesh(problem)
        centers_x, centers_y = cell_centers(mesh_obj)
        nx, ny = size(mesh_obj)

        dρ = component(du, 1)
        drhou = component(du, 2)
        drhov = component(du, 3)
        dE = component(du, 4)

        @inbounds for j in 1:ny
            y = centers_y[j]
            for i in 1:nx
                x = centers_x[i]
                phase = kx * x + ky * y - params.omega * t
                s = sin(phase)
                c = cos(phase)
                ρ, u, v, p = _manufactured_state(params, s, c)
                Sρ, Sρu, Sρv, SE = _manufactured_sources(params, kx, ky, γ, s, c, ρ, u, v, p)
                dρ[i, j] += Sρ
                drhou[i, j] += Sρu
                drhov[i, j] += Sρv
                dE[i, j] += SE
            end
        end

        return du
    end
    return source!
end

@inline function _manufactured_state(params, s, c)
    ρ = params.rho0 + params.rho_amp * s
    u = params.u0 + params.u_amp * c
    v = params.v0 + params.v_amp * s
    p = params.p0 + params.p_amp * c
    return ρ, u, v, p
end

function _manufactured_primitives(params, kx, ky, x, y, t)
    phase = kx * x + ky * y - params.omega * t
    s = sin(phase)
    c = cos(phase)
    return _manufactured_state(params, s, c)
end

function _manufactured_sources(params, kx, ky, γ, s, c, ρ, u, v, p)
    ω = params.omega

    ρt = -params.rho_amp * ω * c
    ρx =  params.rho_amp * kx * c
    ρy =  params.rho_amp * ky * c

    ut =  params.u_amp * ω * s
    ux = -params.u_amp * kx * s
    uy = -params.u_amp * ky * s

    vt = -params.v_amp * ω * c
    vx =  params.v_amp * kx * c
    vy =  params.v_amp * ky * c

    pt =  params.p_amp * ω * s
    px = -params.p_amp * kx * s
    py = -params.p_amp * ky * s

    u2 = u^2
    v2 = v^2

    ρu_t = ρt * u + ρ * ut
    ρv_t = ρt * v + ρ * vt

    ρu_x = ρx * u + ρ * ux
    ρu_y = ρy * u + ρ * uy
    ρv_x = ρx * v + ρ * vx
    ρv_y = ρy * v + ρ * vy

    ρu2_x = ρx * u2 + ρ * 2u * ux
    ρv2_y = ρy * v2 + ρ * 2v * vy

    ρuv_x = ρx * u * v + ρ * (ux * v + u * vx)
    ρuv_y = ρy * u * v + ρ * (uy * v + u * vy)

    E = p / (γ - 1) + 0.5 * ρ * (u2 + v2)
    Et = pt / (γ - 1) + 0.5 * ρt * (u2 + v2) + ρ * (u * ut + v * vt)
    Ex = px / (γ - 1) + 0.5 * ρx * (u2 + v2) + ρ * (u * ux + v * vx)
    Ey = py / (γ - 1) + 0.5 * ρy * (u2 + v2) + ρ * (u * uy + v * vy)

    mass_res = ρt + ρu_x + ρv_y

    momx_res = ρu_t + (ρu2_x + px) + ρuv_y
    momy_res = ρv_t + ρuv_x + (ρv2_y + py)

    Eplusp = E + p
    energy_res = Et + ((Ex + px) * u + Eplusp * ux) + ((Ey + py) * v + Eplusp * vy)

    return mass_res, momx_res, momy_res, energy_res
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

if abspath(PROGRAM_FILE) == @__FILE__
    run_euler_convergence_study()
end

#!/usr/bin/env julia
using CodexMach

"""
    run_linear_advection_demo(; nx=256, ny=128, lengths=(2.0, 1.0),
                               velocity=(1.0, 0.0), cfl=0.4, steps=100,
                               backend=default_backend(), state_eltype=nothing,
                               sample_every=nothing,
                               diagnostics_path=nothing, state_path=nothing,
                               match_return_period=true)

Run a periodic linear advection problem on a rectangular structured mesh whose
cells are square (`Δx = Δy`) by allocating more cells along the longer x
exactly four oscillations across x and two across y, giving circular level sets.
The integration length is automatically increased so the advected field completes
an integer number of domain traversals, bringing the final state back to the
initial configuration. Optionally write diagnostics and the final state to disk;
visualization lives in `plot_linear_advection.jl`.
"""
function run_linear_advection_demo(; nx::Int = 256,
                                    ny::Int = 128,
                                    lengths::NTuple{2,<:Real} = (2.0, 1.0),
                                    velocity::Tuple{<:Real,<:Real} = (1.0, 0.0),
                                    cfl::Real = 0.4,
                                    steps::Int = 100,
                                    backend::Union{ExecutionBackend,Symbol} = default_backend(),
                                    state_eltype::Union{Nothing,Type} = nothing,
                                    sample_every::Union{Nothing,Int} = nothing,
                                    diagnostics_path::Union{Nothing,AbstractString} = nothing,
                                    state_path::Union{Nothing,AbstractString} = nothing,
                                    match_return_period::Bool = true)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    nx >= 1 || throw(ArgumentError("nx must be positive"))
    ny >= 1 || throw(ArgumentError("ny must be positive"))

    lengths_tuple = (float(lengths[1]), float(lengths[2]))

    nx_cells, ny_cells = _square_cell_counts(nx, ny, lengths_tuple)

    problem = setup_linear_advection_problem(nx_cells, ny_cells;
                                             velocity = velocity,
                                             lengths = lengths_tuple)
    backend_obj = _resolve_example_backend(backend)
    Tstate = isnothing(state_eltype) ? _default_state_eltype(backend_obj) : state_eltype

    init_field = _sine_blob_initializer(lengths_tuple)
    state = LinearAdvectionState(problem;
                                 init = init_field,
                                 T = Tstate,
                                 backend = backend_obj)

    dt_cfl = stable_timestep(problem; cfl = cfl)
    target_time = _return_period(lengths_tuple, velocity)

    if match_return_period && target_time > 0
        total_steps = _select_step_count(steps, dt_cfl, target_time)
        effective_dt = target_time / total_steps
    else
        total_steps = steps
        effective_dt = dt_cfl
    end

    sample_stride = isnothing(sample_every) ? max(1, total_steps ÷ 8) : sample_every
    sample_stride > 0 || throw(ArgumentError("sample_every must be positive when provided"))

    rms_history = Float64[]
    records = Vector{NamedTuple{(:step, :time, :rms, :cfl), NTuple{4,Float64}}}()
    function record_callback(step, state, problem, dt)
        if step % sample_stride == 0 || step == total_steps
            u = solution(state)
            rms = sqrt(sum(abs2, u) / length(u))
            time = step * dt
            cfl_val = cfl_number(problem, dt)
            push!(rms_history, rms)
            push!(records, (; step = float(step), time = time, rms = rms, cfl = cfl_val))
            @info "linear advection demo" step = step time = time rms = rms cfl = cfl_val
        end
    end

    summary = run_linear_advection!(state, problem;
                                    steps = total_steps,
                                    dt = effective_dt,
                                    callback = record_callback,
                                    record_cfl = true)

    if diagnostics_path !== nothing
        _write_diagnostics_csv(diagnostics_path, records)
    end

    final_field = copy(solution(state))
    final_state = scalar_component(final_field)
    if state_path !== nothing
        _write_state_csv(state_path, problem, final_state)
    end

    return merge(summary, (; diagnostics = rms_history,
                            diagnostic_records = records,
                            final_state = final_state,
                            nx = nx_cells,
                            ny = ny_cells,
                            target_time = target_time))
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

function _write_diagnostics_csv(path::AbstractString,
                                records::Vector{NamedTuple{(:step, :time, :rms, :cfl), NTuple{4,Float64}}})
    open(path, "w") do io
        println(io, "step,time,rms,cfl")
        for r in records
            println(io, "$(r.step),$(r.time),$(r.rms),$(r.cfl)")
        end
    end
    return path
end

function _sine_blob_initializer(lengths::NTuple{2,Float64})
    Lx, Ly = lengths
    kx = 4
    ky = 2
    function init(x, y)
        return sin(2pi * kx * x / Lx) * sin(2pi * ky * y / Ly)
    end
    return init
end

function _square_cell_counts(nx::Int, ny::Int, lengths::NTuple{2,Float64})
    nx >= 1 || throw(ArgumentError("nx must be positive"))
    ny >= 1 || throw(ArgumentError("ny must be positive"))
    Lx, Ly = lengths
    Lx > 0 || throw(ArgumentError("lengths[1] must be positive"))
    Ly > 0 || throw(ArgumentError("lengths[2] must be positive"))

    dx = Lx / nx
    dy = Ly / ny

    if isapprox(dx, dy; rtol = 1e-12, atol = 1e-12)
        return nx, ny
    end

    if dx > dy
        target_nx = max(nx, Int(round(Lx / dy)))
        dx_new = Lx / target_nx
        if !isapprox(dx_new, dy; rtol = 1e-12, atol = 1e-12)
            throw(ArgumentError("Unable to match square cells by refining nx (nx=$(target_nx), ny=$(ny))"))
        end
        return target_nx, ny
    else
        target_ny = max(ny, Int(round(Ly / dx)))
        dy_new = Ly / target_ny
        if !isapprox(dx, dy_new; rtol = 1e-12, atol = 1e-12)
            throw(ArgumentError("Unable to match square cells by refining ny (nx=$(nx), ny=$(target_ny))"))
        end
        return nx, target_ny
    end
end

function _return_period(lengths::NTuple{2,Float64}, velocity::Tuple{<:Real,<:Real})
    periods = Rational{Int}[]
    Lx, Ly = lengths
    vx, vy = velocity
    tol = eps(Float64)

    if !iszero(vx)
        push!(periods, rationalize(Lx / abs(vx); tol = tol))
    end
    if !iszero(vy)
        push!(periods, rationalize(Ly / abs(vy); tol = tol))
    end

    isempty(periods) && return 0.0

    common = periods[1]
    for p in Iterators.drop(periods, 1)
        common = _lcm_rational(common, p)
    end

    return float(common)
end

function _lcm_rational(a::Rational{T}, b::Rational{T}) where {T<:Integer}
    num = lcm(numerator(a), numerator(b))
    den = gcd(denominator(a), denominator(b))
    return num // den
end

function _select_step_count(requested_steps::Int, dt_cfl::Float64, target_time::Float64)
    requested_steps > 0 || throw(ArgumentError("steps must be positive"))
    if target_time <= 0
        return requested_steps
    end

    ratio = target_time / dt_cfl
    target_steps = Int(ceil(ratio - eps(Float64)))
    target_steps = max(target_steps, 1)
    return max(requested_steps, target_steps)
end

function _write_state_csv(path::AbstractString,
                          problem::LinearAdvectionProblem,
                          state::AbstractArray)
    mesh = CodexMach.mesh(problem)
    centers_x, centers_y = CodexMach.cell_centers(mesh)
    nx, ny = size(mesh)

    open(path, "w") do io
        println(io, "i,j,x,y,u")
        @inbounds for j in 1:ny, i in 1:nx
            println(io, "$(i),$(j),$(centers_x[i]),$(centers_y[j]),$(state[i, j])")
        end
    end
    return path
end

function main()
    diagnostics_path = length(ARGS) >= 1 ? ARGS[1] : nothing
    state_path = length(ARGS) >= 2 ? ARGS[2] : nothing

    summary = run_linear_advection_demo(; nx=64, ny=32, diagnostics_path = diagnostics_path,
                                         state_path = state_path)
    @info "Demo complete" final_time = summary.final_time dt = summary.dt cfl = summary.cfl samples = length(summary.diagnostics) nx = summary.nx ny = summary.ny target_time = summary.target_time

    if diagnostics_path !== nothing
        @info "Diagnostics written" path = diagnostics_path
    end
    if state_path !== nothing
        @info "Final state written" path = state_path
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

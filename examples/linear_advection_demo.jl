#!/usr/bin/env julia
using CodexPar

"""
    run_linear_advection_demo(; nx=256, ny=128, lengths=(2.0, 1.0),
                               velocity=(1.0, 0.0), cfl=0.4, steps=100,
                               sample_every=nothing,
                               diagnostics_path=nothing, state_path=nothing)

Run a periodic linear advection problem on a rectangular structured mesh whose
cells are square (`Δx = Δy`) by allocating more cells along the longer x
direction. The initial condition comprises a lattice of sine ``blobs`` that fits
exactly four oscillations across x and two across y, giving circular level sets.
Optionally write diagnostics and the final state to disk; visualization lives in
`plot_linear_advection.jl`.
"""
function run_linear_advection_demo(; nx::Int = 256,
                                    ny::Int = 128,
                                    lengths::NTuple{2,<:Real} = (2.0, 1.0),
                                    velocity::Tuple{<:Real,<:Real} = (1.0, 0.0),
                                    cfl::Real = 0.4,
                                    steps::Int = 100,
                                    sample_every::Union{Nothing,Int} = nothing,
                                    diagnostics_path::Union{Nothing,AbstractString} = nothing,
                                    state_path::Union{Nothing,AbstractString} = nothing)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    nx >= 1 || throw(ArgumentError("nx must be positive"))
    ny >= 1 || throw(ArgumentError("ny must be positive"))

    lengths_tuple = (float(lengths[1]), float(lengths[2]))

    nx_cells, ny_cells = _square_cell_counts(nx, ny, lengths_tuple)

    sample_stride = isnothing(sample_every) ? max(1, steps ÷ 8) : sample_every
    sample_stride > 0 || throw(ArgumentError("sample_every must be positive when provided"))

    problem = setup_linear_advection_problem(nx_cells, ny_cells;
                                             velocity = velocity,
                                             lengths = lengths_tuple)
    init_field = _sine_blob_initializer(lengths_tuple)
    state = LinearAdvectionState(problem; init = init_field)

    rms_history = Float64[]
    records = Vector{NamedTuple{(:step, :time, :rms, :cfl), NTuple{4,Float64}}}()
    function record_callback(step, state, problem, dt)
        if step % sample_stride == 0 || step == steps
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
                                    steps = steps,
                                    cfl_target = cfl,
                                    callback = record_callback,
                                    record_cfl = true)

    if diagnostics_path !== nothing
        _write_diagnostics_csv(diagnostics_path, records)
    end

    final_state = copy(solution(state))
    if state_path !== nothing
        _write_state_csv(state_path, problem, final_state)
    end

    return merge(summary, (; diagnostics = rms_history,
                            diagnostic_records = records,
                            final_state = final_state,
                            nx = nx_cells,
                            ny = ny_cells))
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

function _write_state_csv(path::AbstractString,
                          problem::LinearAdvectionProblem,
                          state::AbstractArray)
    mesh = CodexPar.mesh(problem)
    centers_x, centers_y = CodexPar.cell_centers(mesh)
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

    summary = run_linear_advection_demo(; diagnostics_path = diagnostics_path,
                                         state_path = state_path)
    @info "Demo complete" final_time = summary.final_time dt = summary.dt cfl = summary.cfl samples = length(summary.diagnostics) nx = summary.nx ny = summary.ny

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

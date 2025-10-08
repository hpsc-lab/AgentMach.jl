"""
    run_linear_advection!(state, problem; steps, dt=nothing, cfl_target=0.9,
                           log_every=0, callback=nothing, record_cfl=false,
                           show_timers=true)

Advance a `LinearAdvectionState` for a fixed number of RK2 steps. Either provide
`dt` explicitly or supply a `cfl_target`, which is passed to `stable_timestep` to
infer a step size. Optional logging and callback hooks support simple drivers.
Set `show_timers = false` to suppress the aggregated `TimerOutputs` summary.

The function returns a named tuple summarising integration details.
"""
function run_linear_advection!(state::LinearAdvectionState,
                               problem::LinearAdvectionProblem;
                               steps::Integer,
                               dt::Union{Nothing,Real} = nothing,
                               cfl_target::Real = 0.9,
                               log_every::Integer = 0,
                               callback = nothing,
                               record_cfl::Bool = false,
                               show_timers::Bool = true)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    log_every >= 0 || throw(ArgumentError("log_every must be non-negative"))

    timer = simulation_timers()
    reset_timer!(timer)

    effective_dt = if dt === nothing
        @timeit timer "Stable timestep" stable_timestep(problem; cfl = cfl_target)
    else
        float(dt)
    end
    isfinite(effective_dt) ||
        throw(ArgumentError("Time step is infinite; provide a finite dt explicitly"))
    effective_dt > 0 || throw(ArgumentError("time step must be positive"))

    cfl = @timeit timer "CFL number" cfl_number(problem, effective_dt)
    history = record_cfl ? Float64[] : nothing

    current_time = 0.0

    @timeit timer "Time integration" begin
        for step in 1:steps
            rk2_step!(state, problem, effective_dt; t = current_time)
            current_time += effective_dt

            if callback !== nothing
                callback(step, state, problem, effective_dt)
            end

            if record_cfl
                push!(history, float(cfl))
            end

            if log_every > 0 && step % log_every == 0
                @info "Linear advection step" step=step step_time=step * effective_dt cfl=cfl
            end
        end
    end

    result = (; dt = effective_dt,
               steps = steps,
               final_time = current_time,
               cfl = cfl,
               cfl_history = history)

    if show_timers
        print_timer(stdout, timer; allocations = true, sortby = :time)
    end

    return result
end

"""
    run_compressible_euler!(state, problem; steps, dt=nothing, cfl_target=0.45,
                             log_every=0, callback=nothing, record_cfl=false,
                             adapt_dt=true, show_timers=true)

Advance a `CompressibleEulerState` for a fixed number of RK2 steps. If `dt` is
omitted, a stable timestep is recomputed from the current state each iteration
using the requested `cfl_target`.
Set `show_timers = false` to suppress the aggregated `TimerOutputs` summary.
"""
function run_compressible_euler!(state::CompressibleEulerState,
                                 problem::CompressibleEulerProblem;
                                 steps::Integer,
                                 dt::Union{Nothing,Real} = nothing,
                                 cfl_target::Real = 0.45,
                                 log_every::Integer = 0,
                                 callback = nothing,
                                 record_cfl::Bool = false,
                                 adapt_dt::Bool = true,
                                 show_timers::Bool = true)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    log_every >= 0 || throw(ArgumentError("log_every must be non-negative"))

    timer = simulation_timers()
    reset_timer!(timer)

    base_dt = if dt === nothing
        @timeit timer "Stable timestep" stable_timestep(problem, state; cfl = cfl_target)
    else
        float(dt)
    end
    isfinite(base_dt) ||
        throw(ArgumentError("Time step is infinite; provide a finite dt explicitly"))
    base_dt > 0 || throw(ArgumentError("time step must be positive"))

    history = record_cfl ? Float64[] : nothing
    total_time = 0.0
    last_cfl = NaN

    @timeit timer "Time integration" begin
        for step in 1:steps
            current_dt = if dt === nothing
                if adapt_dt
                    @timeit timer "Stable timestep" stable_timestep(problem, state; cfl = cfl_target)
                else
                    base_dt
                end
            else
                base_dt
            end
            current_dt > 0 || throw(ArgumentError("Encountered non-positive time step during integration"))

            rk2_step!(state, problem, current_dt; t = total_time)
            total_time += current_dt

            last_cfl = @timeit timer "CFL number" cfl_number(problem, state, current_dt)

            if callback !== nothing
                callback(step, state, problem, current_dt)
            end

            if record_cfl
                push!(history, float(last_cfl))
            end

            if log_every > 0 && step % log_every == 0
                @info "Compressible Euler step" step=step step_time=total_time cfl=last_cfl dt=current_dt
            end
        end
    end

    result = (; dt = (dt === nothing && adapt_dt) ? nothing : base_dt,
               steps = steps,
               final_time = total_time,
               cfl = last_cfl,
               cfl_history = history)

    if show_timers
        print_timer(stdout, timer; allocations = true, sortby = :time)
    end

    return result
end

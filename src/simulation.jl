"""
    run_linear_advection!(state, problem; steps, dt=nothing, cfl_target=0.9,
                           log_every=0, callback=nothing, record_cfl=false)

Advance a `LinearAdvectionState` for a fixed number of RK2 steps. Either provide
`dt` explicitly or supply a `cfl_target`, which is passed to `stable_timestep` to
infer a step size. Optional logging and callback hooks support simple drivers.

The function returns a named tuple summarising integration details.
"""
function run_linear_advection!(state::LinearAdvectionState,
                               problem::LinearAdvectionProblem;
                               steps::Integer,
                               dt::Union{Nothing,Real} = nothing,
                               cfl_target::Real = 0.9,
                               log_every::Integer = 0,
                               callback = nothing,
                               record_cfl::Bool = false)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    log_every >= 0 || throw(ArgumentError("log_every must be non-negative"))

    effective_dt = dt === nothing ? stable_timestep(problem; cfl = cfl_target) : float(dt)
    isfinite(effective_dt) ||
        throw(ArgumentError("Time step is infinite; provide a finite dt explicitly"))
    effective_dt > 0 || throw(ArgumentError("time step must be positive"))

    cfl = cfl_number(problem, effective_dt)
    history = record_cfl ? Float64[] : nothing

    for step in 1:steps
        rk2_step!(state, problem, effective_dt)

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

    return (; dt = effective_dt,
            steps = steps,
            final_time = steps * effective_dt,
            cfl = cfl,
            cfl_history = history)
end

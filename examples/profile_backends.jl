using CodexMach
try
    using Metal
    CodexMach.register_backend!(:metal, () -> Metal.MetalBackend())
    println("Metal backend registered")
catch err
    println("Metal unavailable: ", err)
end

function profile(label, setup)
    println("\n== $label ==")
    for (name, state, prob, dt, steps) in setup()
        GC.gc()
        elapsed = @elapsed begin
            current_time = 0.0
            for _ in 1:steps
                rk2_step!(state, prob, dt; t = current_time)
                current_time += dt
            end
        end
        println(rpad(name, 10), ": ", round(elapsed, digits=3), " s")
    end
end

nx = 1024
ny = 1024
steps = 100
prob_la = setup_linear_advection_problem(nx, ny; velocity=(1.0, 0.5))
dt_la = 1e-3
profile_T = Float32

function advection_cases()
    cases = Tuple{String,Any,Any,Float64,Int}[]
    push!(cases, ("Serial", LinearAdvectionState(prob_la; init=1.0, T=profile_T), prob_la, dt_la, steps))
    push!(cases, ("KA CPU", LinearAdvectionState(prob_la; init=1.0, T=profile_T, backend=KernelAbstractionsBackend(:cpu)), prob_la, dt_la, steps))
    if :metal in available_backends()
        push!(cases, ("Metal", LinearAdvectionState(prob_la; init=1.0, T=profile_T, backend=KernelAbstractionsBackend(:metal)), prob_la, dt_la, steps))
    end
    return cases
end

profile("Linear Advection $(nx)x$(ny), steps=$(steps)", advection_cases)

prob_eu = setup_compressible_euler_problem(nx÷2, ny÷2; gamma=1.4)
init(x,y) = (; rho = 1.0 + 0.1 * sinpi(x), v1 = 0.05*cospi(y), v2 = 0.02*sinpi(x+y), p = 1.0 + 0.05*cospi(x))
serial_state = CompressibleEulerState(prob_eu; init=init)
dt_eu = stable_timestep(prob_eu, serial_state; cfl=0.3)
steps_eu = 100

function euler_cases()
    cases = Tuple{String,Any,Any,Float64,Int}[]
    push!(cases, ("Serial", CompressibleEulerState(prob_eu; init=init, T=profile_T), prob_eu, dt_eu, steps_eu))
    push!(cases, ("KA CPU", CompressibleEulerState(prob_eu; init=init, T=profile_T, backend=KernelAbstractionsBackend(:cpu)), prob_eu, dt_eu, steps_eu))
    if :metal in available_backends()
        push!(cases, ("Metal", CompressibleEulerState(prob_eu; init=init, T=profile_T, backend=KernelAbstractionsBackend(:metal)), prob_eu, dt_eu, steps_eu))
    end
    return cases
end

profile("Compressible Euler $(nx÷2)x$(ny÷2), steps=$(steps_eu)", euler_cases)

module AgentMach

using TimerOutputs

const _SIMULATION_TIMERS = TimerOutput("AgentMach Simulation")

simulation_timers() = _SIMULATION_TIMERS

include("mesh.jl")
include("boundary_conditions.jl")
include("fields.jl")
include("backends.jl")
include("limiters.jl")
include("kernel_abstractions_support.jl")
include("equations.jl")
include("time_integration.jl")
include("simulation.jl")

export StructuredMesh,
       cell_centers,
       spacing,
       origin,
       PeriodicBoundaryConditions,
       is_periodic,
       periodic_axes,
       CellField,
       cell_components,
       ncomponents,
       spatial_size,
       component,
       scalar_component,
       allocate_cellfield,
       allocate_like,
       map_components!,
       register_backend!,
        available_backends,
       ExecutionBackend,
        SerialBackend,
        KernelAbstractionsBackend,
        default_backend,
        describe,
       LinearAdvection,
       LinearAdvectionProblem,
       velocity,
       mesh,
       boundary_conditions,
       pde,
       source,
       limiter,
       setup_linear_advection_problem,
       LinearAdvectionState,
       RK2Workspace,
       solution,
       workspace,
        backend,
       compute_rhs!,
       rk2_step!,
       cfl_number,
       stable_timestep,
       run_linear_advection!,
       CompressibleEuler,
       CompressibleEulerProblem,
       setup_compressible_euler_problem,
       CompressibleEulerState,
       run_compressible_euler!,
       primitive_variables,
       AbstractLimiter,
       MinmodLimiter,
       UnlimitedLimiter,
       minmod_limiter,
       unlimited_limiter,
       apply_limiter

end # module

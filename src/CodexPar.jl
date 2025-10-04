module CodexPar

include("mesh.jl")
include("boundary_conditions.jl")
include("equations.jl")
include("time_integration.jl")

export greet,
       StructuredMesh,
       cell_centers,
       spacing,
       origin,
       PeriodicBoundaryConditions,
       is_periodic,
       periodic_axes,
       LinearAdvection,
       LinearAdvectionProblem,
       velocity,
       mesh,
       boundary_conditions,
       pde,
       setup_linear_advection_problem,
       LinearAdvectionState,
       RK2Workspace,
       solution,
       workspace,
       compute_rhs!,
       rk2_step!

"""
    greet(name::AbstractString = "world")

Return a friendly greeting so downstream users can smoke-test that the package is correctly installed.
"""
function greet(name::AbstractString = "world")
    return "Hello, $(name)! Welcome to CodexPar."
end

end # module

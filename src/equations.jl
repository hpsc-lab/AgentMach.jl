"""
    LinearAdvection(velocity)

Define the linear advection equation with a constant advection `velocity`
represented as a 2-tuple.
"""
struct LinearAdvection{T}
    velocity::NTuple{2,T}
    function LinearAdvection{T}(velocity::NTuple{2,T}) where {T}
        new{T}(velocity)
    end
end

function LinearAdvection(velocity::NTuple{2,<:Real})
    vel = (float(velocity[1]), float(velocity[2]))
    T = promote_type(typeof(vel[1]), typeof(vel[2]))
    return LinearAdvection{T}((T(vel[1]), T(vel[2])))
end

function LinearAdvection(velocity::AbstractVector{<:Real})
    length(velocity) == 2 ||
        throw(ArgumentError("LinearAdvection velocity must have length 2"))
    return LinearAdvection((velocity[1], velocity[2]))
end

velocity(eq::LinearAdvection) = eq.velocity

"""
    LinearAdvectionProblem(mesh, bc, equation)

Bundle the mesh, boundary conditions, and PDE description for the linear
advection equation.
"""
struct LinearAdvectionProblem{M,B,T}
    mesh::M
    boundary_conditions::B
    equation::LinearAdvection{T}
    function LinearAdvectionProblem(mesh::M,
                                    boundary_conditions::B,
                                    equation::LinearAdvection{T}) where {M,B,T}
        new{M,B,T}(mesh, boundary_conditions, equation)
    end
end

mesh(problem::LinearAdvectionProblem) = problem.mesh
boundary_conditions(problem::LinearAdvectionProblem) = problem.boundary_conditions
pde(problem::LinearAdvectionProblem) = problem.equation

"""
    setup_linear_advection_problem(nx, ny; lengths=(1.0, 1.0), origin=(0.0, 0.0), velocity=(1.0, 0.0))

Create a linear advection problem on a structured mesh with periodic boundary
conditions along both axes.
"""
function setup_linear_advection_problem(nx::Integer,
                                        ny::Integer;
                                        lengths::NTuple{2,<:Real} = (1.0, 1.0),
                                        origin::NTuple{2,<:Real} = (0.0, 0.0),
                                        velocity::NTuple{2,<:Real} = (1.0, 0.0))
    mesh = StructuredMesh(nx, ny; lengths = lengths, origin = origin)
    bc = PeriodicBoundaryConditions()
    eq = LinearAdvection(velocity)
    return LinearAdvectionProblem(mesh, bc, eq)
end

"""
    CompressibleEuler(; gamma=1.4)

Create a 2D compressible Euler equation set with ratio of specific heats `gamma`.
"""
struct CompressibleEuler{T}
    gamma::T
end

function CompressibleEuler(; gamma::Real = 1.4)
    g = float(gamma)
    g > 1 || throw(ArgumentError("gamma must exceed unity for physical gases"))
    T = typeof(g)
    return CompressibleEuler{T}(g)
end

gamma(eq::CompressibleEuler) = eq.gamma

"""
    CompressibleEulerProblem(mesh, bc, equation)

Bundle the mesh, boundary conditions, and PDE description for the 2D
compressible Euler equations.
"""
struct CompressibleEulerProblem{M,B,T}
    mesh::M
    boundary_conditions::B
    equation::CompressibleEuler{T}
    function CompressibleEulerProblem(mesh::M,
                                      boundary_conditions::B,
                                      equation::CompressibleEuler{T}) where {M,B,T}
        new{M,B,T}(mesh, boundary_conditions, equation)
    end
end

mesh(problem::CompressibleEulerProblem) = problem.mesh
boundary_conditions(problem::CompressibleEulerProblem) = problem.boundary_conditions
pde(problem::CompressibleEulerProblem) = problem.equation

"""
    setup_compressible_euler_problem(nx, ny; lengths=(1.0, 1.0), origin=(0.0, 0.0),
                                     gamma=1.4, boundary_conditions=PeriodicBoundaryConditions())

Create a compressible Euler problem on a structured mesh with configurable
boundary conditions (defaults to fully periodic).
"""
function setup_compressible_euler_problem(nx::Integer,
                                          ny::Integer;
                                          lengths::NTuple{2,<:Real} = (1.0, 1.0),
                                          origin::NTuple{2,<:Real} = (0.0, 0.0),
                                          gamma::Real = 1.4,
                                          boundary_conditions::AbstractBoundaryConditions = PeriodicBoundaryConditions())
    mesh = StructuredMesh(nx, ny; lengths = lengths, origin = origin)
    eq = CompressibleEuler(; gamma = gamma)
    return CompressibleEulerProblem(mesh, boundary_conditions, eq)
end

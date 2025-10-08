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

"""
    velocity(eq)

Return the constant advection velocity associated with `eq`.
"""
velocity(eq::LinearAdvection) = eq.velocity

"""
    LinearAdvectionProblem(mesh, bc, equation, [source])

Bundle the mesh, boundary conditions, and PDE description for the linear
advection equation. The optional `source` callback receives `(du, u, problem, t)`
and may mutate `du` in-place to add volumetric forcing.
"""
struct LinearAdvectionProblem{M,B,T,S}
    mesh::M
    boundary_conditions::B
    equation::LinearAdvection{T}
    source::S
    function LinearAdvectionProblem(mesh::M,
                                    boundary_conditions::B,
                                    equation::LinearAdvection{T},
                                    source::S) where {M,B,T,S}
        new{M,B,T,S}(mesh, boundary_conditions, equation, source)
    end
end

function LinearAdvectionProblem(mesh::M,
                                boundary_conditions::B,
                                equation::LinearAdvection{T}) where {M,B,T}
    return LinearAdvectionProblem(mesh, boundary_conditions, equation, nothing)
end

"""
    mesh(problem)

Return the mesh associated with the PDE problem `problem`.
"""
mesh(problem::LinearAdvectionProblem) = problem.mesh

"""
    boundary_conditions(problem)

Return the boundary-condition specification attached to `problem`.
"""
boundary_conditions(problem::LinearAdvectionProblem) = problem.boundary_conditions

"""
    pde(problem)

Return the equation-of-motion descriptor stored within `problem`.
"""
pde(problem::LinearAdvectionProblem) = problem.equation

"""
    source(problem)

Return the volumetric source callback associated with `problem`, or `nothing`
if no source is registered.
"""
source(problem::LinearAdvectionProblem) = problem.source

"""
    setup_linear_advection_problem(nx, ny; lengths=(1.0, 1.0), origin=(0.0, 0.0),
                                   velocity=(1.0, 0.0), source=nothing)

Create a linear advection problem on a structured mesh with periodic boundary
conditions along both axes. Supply `source` with a function of
`(du, u, problem, t)` to append volumetric forcing contributions.
"""
function setup_linear_advection_problem(nx::Integer,
                                        ny::Integer;
                                        lengths::NTuple{2,<:Real} = (1.0, 1.0),
                                        origin::NTuple{2,<:Real} = (0.0, 0.0),
                                        velocity::NTuple{2,<:Real} = (1.0, 0.0),
                                        source = nothing)
    mesh = StructuredMesh(nx, ny; lengths = lengths, origin = origin)
    bc = PeriodicBoundaryConditions()
    eq = LinearAdvection(velocity)
    return LinearAdvectionProblem(mesh, bc, eq, source)
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
    CompressibleEulerProblem(mesh, bc, equation, [source], [limiter])

Bundle the mesh, boundary conditions, and PDE description for the 2D
compressible Euler equations. The optional `source` callback is invoked as
`source(du, u, problem, t)` to accumulate volumetric forcing, while the
`limiter` controls the MUSCL slope limiting strategy.
"""
struct CompressibleEulerProblem{M,B,T,S,L}
    mesh::M
    boundary_conditions::B
    equation::CompressibleEuler{T}
    source::S
    limiter::L
    function CompressibleEulerProblem(mesh::M,
                                      boundary_conditions::B,
                                      equation::CompressibleEuler{T},
                                      source::S,
                                      limiter::L) where {M,B,T,S,L}
        new{M,B,T,S,L}(mesh, boundary_conditions, equation, source, limiter)
    end
end

function CompressibleEulerProblem(mesh::M,
                                  boundary_conditions::B,
                                  equation::CompressibleEuler{T}) where {M,B,T}
    return CompressibleEulerProblem(mesh, boundary_conditions, equation, nothing, minmod_limiter)
end

"""
    mesh(problem)

Return the mesh associated with the PDE problem `problem`.
"""
mesh(problem::CompressibleEulerProblem) = problem.mesh

"""
    boundary_conditions(problem)

Return the boundary-condition specification attached to `problem`.
"""
boundary_conditions(problem::CompressibleEulerProblem) = problem.boundary_conditions

"""
    pde(problem)

Return the equation-of-motion descriptor stored within `problem`.
"""
pde(problem::CompressibleEulerProblem) = problem.equation

source(problem::CompressibleEulerProblem) = problem.source

limiter(problem::CompressibleEulerProblem) = problem.limiter

"""
    setup_compressible_euler_problem(nx, ny; lengths=(1.0, 1.0), origin=(0.0, 0.0),
                                     gamma=1.4,
                                     boundary_conditions=PeriodicBoundaryConditions(),
                                     source=nothing)

Create a compressible Euler problem on a structured mesh with configurable
boundary conditions (defaults to fully periodic). Provide a `source` callback
with signature `(du, u, problem, t)` to add volumetric forcing in-place, and
optionally supply an `AbstractLimiter` to control MUSCL slope limiting.
"""
function setup_compressible_euler_problem(nx::Integer,
                                          ny::Integer;
                                          lengths::NTuple{2,<:Real} = (1.0, 1.0),
                                          origin::NTuple{2,<:Real} = (0.0, 0.0),
                                          gamma::Real = 1.4,
                                          boundary_conditions::AbstractBoundaryConditions = PeriodicBoundaryConditions(),
                                          source = nothing,
                                          limiter::AbstractLimiter = minmod_limiter)
    mesh = StructuredMesh(nx, ny; lengths = lengths, origin = origin)
    eq = CompressibleEuler(; gamma = gamma)
    return CompressibleEulerProblem(mesh, boundary_conditions, eq, source, limiter)
end

"""
    primitive_variables(eq, ρ, ρu, ρv, E)

Convert conserved quantities into primitive variables `(ρ, u, v, p)` for a
compressible Euler equation set.
"""
@inline function primitive_variables(eq::CompressibleEuler,
                                     ρ::Real,
                                     ρu::Real,
                                     ρv::Real,
                                     E::Real)
    ρT = float(ρ)
    ρT > 0 || throw(ArgumentError("Density must remain positive"))

    ρuT = float(ρu)
    ρvT = float(ρv)
    ET = float(E)
    γT = float(gamma(eq))

    invρ = one(ρT) / ρT
    u = ρuT * invρ
    v = ρvT * invρ
    half = convert(typeof(ρT), 0.5)
    kinetic = half * ρT * (u^2 + v^2)

    internal = ET - kinetic
    epsT = eps(typeof(ρT))
    internal = max(internal, epsT)
    p = (γT - convert(typeof(γT), 1)) * internal
    p = max(p, epsT)

    return (ρ = ρT, u = u, v = v, p = p)
end

"""
    primitive_variables(eq, conserved; rho_out=nothing, u_out=nothing,
                        v_out=nothing, p_out=nothing)

Convert a conserved-field array with layout `(4, nx, ny)` to primitive
variables. Optionally provide preallocated output arrays via the keyword
arguments. Returns a named tuple `(rho, u, v, p)`.
"""
function _primitive_variables_cpu(eq::CompressibleEuler,
                                  conserved::CellField;
                                  rho_out = nothing,
                                  u_out = nothing,
                                  v_out = nothing,
                                  p_out = nothing)
    nx, ny = spatial_size(conserved)

    T = component_eltype(conserved)
    out_T = float(T)

    ρ = rho_out === nothing ? Array{out_T}(undef, nx, ny) : rho_out
    u = u_out === nothing ? similar(ρ) : u_out
    v = v_out === nothing ? similar(ρ) : v_out
    p = p_out === nothing ? similar(ρ) : p_out

    size(ρ) == (nx, ny) ||
        throw(ArgumentError("rho_out size mismatch"))
    (size(u) == (nx, ny) && size(v) == (nx, ny) && size(p) == (nx, ny)) ||
        throw(ArgumentError("Primitive output buffers must match conserved field shape"))

    ρc = component(conserved, 1)
    rhouc = component(conserved, 2)
    rhovc = component(conserved, 3)
    Ec = component(conserved, 4)

    @inbounds for j in 1:ny, i in 1:nx
        prim = primitive_variables(eq,
                                   ρc[i, j],
                                   rhouc[i, j],
                                   rhovc[i, j],
                                   Ec[i, j])
        ρ[i, j] = prim.ρ
        u[i, j] = prim.u
        v[i, j] = prim.v
        p[i, j] = prim.p
    end

    return (; rho = ρ, u = u, v = v, p = p)
end

function _primitive_variables_cpu(eq::CompressibleEuler,
                                  conserved::AbstractArray{T,3};
                                  rho_out = nothing,
                                  u_out = nothing,
                                  v_out = nothing,
                                  p_out = nothing) where {T}
    size(conserved, 1) == 4 ||
        throw(ArgumentError("Conserved field must have first dimension of length 4"))

    nx, ny = size(conserved, 2), size(conserved, 3)
    out_T = float(T)

    ρ = rho_out === nothing ? Array{out_T}(undef, nx, ny) : rho_out
    u = u_out === nothing ? similar(ρ) : u_out
    v = v_out === nothing ? similar(ρ) : v_out
    p = p_out === nothing ? similar(ρ) : p_out

    size(ρ) == (nx, ny) ||
        throw(ArgumentError("rho_out size mismatch"))
    (size(u) == (nx, ny) && size(v) == (nx, ny) && size(p) == (nx, ny)) ||
        throw(ArgumentError("Primitive output buffers must match conserved field shape"))

    @inbounds for j in 1:ny, i in 1:nx
        prim = primitive_variables(eq,
                                   conserved[1, i, j],
                                   conserved[2, i, j],
                                   conserved[3, i, j],
                                   conserved[4, i, j])
        ρ[i, j] = prim.ρ
        u[i, j] = prim.u
        v[i, j] = prim.v
        p[i, j] = prim.p
    end

    return (; rho = ρ, u = u, v = v, p = p)
end

function _primitive_variables_gpu(eq::CompressibleEuler,
                                  conserved::CellField,
                                  backend::KernelAbstractionsBackend;
                                  rho_out = nothing,
                                  u_out = nothing,
                                  v_out = nothing,
                                  p_out = nothing)
    size(conserved, 1) == 4 ||
        throw(ArgumentError("Conserved field must have first dimension of length 4"))

    ρc = component(conserved, 1)
    rhouc = component(conserved, 2)
    rhovc = component(conserved, 3)
    Ec = component(conserved, 4)

    ρ = rho_out === nothing ? similar(ρc) : rho_out
    u = u_out === nothing ? similar(ρc) : u_out
    v = v_out === nothing ? similar(ρc) : v_out
    p = p_out === nothing ? similar(ρc) : p_out

    size(ρ) == size(ρc) ||
        throw(ArgumentError("rho_out size mismatch"))
    (size(u) == size(ρc) && size(v) == size(ρc) && size(p) == size(ρc)) ||
        throw(ArgumentError("Primitive output buffers must match conserved field shape"))

    T = eltype(ρc)
    γT = convert(T, gamma(eq))
    γm1 = γT - one(γT)
    epsT = eps(T)

    primitive_variables_kernel!(backend,
                                ρ, u, v, p,
                                ρc, rhouc, rhovc, Ec,
                                γm1, epsT)

    return (; rho = ρ, u = u, v = v, p = p)
end

function primitive_variables(eq::CompressibleEuler,
                             conserved::CellField;
                             backend::ExecutionBackend = SerialBackend(),
                             rho_out = nothing,
                             u_out = nothing,
                             v_out = nothing,
                             p_out = nothing)
    if backend isa KernelAbstractionsBackend
        return _primitive_variables_gpu(eq, conserved, backend;
                                        rho_out = rho_out,
                                        u_out = u_out,
                                        v_out = v_out,
                                        p_out = p_out)
    else
        return _primitive_variables_cpu(eq, conserved;
                                        rho_out = rho_out,
                                        u_out = u_out,
                                        v_out = v_out,
                                        p_out = p_out)
    end
end

function primitive_variables(eq::CompressibleEuler,
                             conserved::AbstractArray{T,3};
                             backend::ExecutionBackend = SerialBackend(),
                             rho_out = nothing,
                             u_out = nothing,
                             v_out = nothing,
                             p_out = nothing) where {T}
    backend isa KernelAbstractionsBackend &&
        throw(ArgumentError("primitive_variables does not support generic AbstractArray inputs on GPU"))
    return _primitive_variables_cpu(eq, conserved;
                                     rho_out = rho_out,
                                     u_out = u_out,
                                     v_out = v_out,
                                     p_out = p_out)
end

primitive_variables(problem::CompressibleEulerProblem,
                    state;
                    backend::ExecutionBackend = backend(state), kwargs...) =
    primitive_variables(pde(problem),
                        solution(state);
                        backend = backend,
                        kwargs...)

primitive_variables(problem::CompressibleEulerProblem,
                    conserved::AbstractArray{T,3};
                    backend::ExecutionBackend = SerialBackend(), kwargs...) where {T} =
    primitive_variables(pde(problem), conserved;
                        backend = backend,
                        kwargs...)

primitive_variables(problem::CompressibleEulerProblem,
                    conserved::CellField;
                    backend::ExecutionBackend = SerialBackend(), kwargs...) =
    primitive_variables(pde(problem), conserved;
                        backend = backend,
                        kwargs...)

"""
    RK2Workspace(k1, k2, stage)

Workspace buffers used to evaluate explicit second-order Runge-Kutta (Heun)
updates without allocating temporary arrays.
"""
struct RK2Workspace{A}
    k1::A
    k2::A
    stage::A
end

"""
    LinearAdvectionState(solution, workspace)

Hold the cell-centered solution field alongside scratch buffers required by the
RK2 integrator.
"""
struct LinearAdvectionState{A,W}
    solution::A
    workspace::W
end

solution(state::LinearAdvectionState) = state.solution
workspace(state::LinearAdvectionState) = state.workspace

const _DefaultArrayType = Array

function LinearAdvectionState(problem::LinearAdvectionProblem;
                              T::Type = Float64,
                              array_type = _DefaultArrayType,
                              init = nothing)
    mesh_obj = mesh(problem)
    dims = size(mesh_obj)
    field = _allocate_field(array_type, T, dims)
    _initialize_field!(field, init, mesh_obj, T)

    k1 = _allocate_field(array_type, T, dims)
    k2 = _allocate_field(array_type, T, dims)
    stage = _allocate_field(array_type, T, dims)
    fill!(k1, zero(T))
    fill!(k2, zero(T))
    fill!(stage, zero(T))

    return LinearAdvectionState(field, RK2Workspace(k1, k2, stage))
end

function _allocate_field(array_type, ::Type{T}, dims::Tuple{Vararg{Int}}) where {T}
    return array_type{T}(undef, dims...)
end

function _initialize_field!(field, init, mesh::StructuredMesh, ::Type{T}) where {T}
    if init === nothing
        fill!(field, zero(T))
        return field
    elseif init isa Number
        fill!(field, T(init))
        return field
    elseif init isa AbstractArray
        size(init) == size(field) ||
            throw(ArgumentError("Initializer array must match mesh dimensions"))
        field .= T.(init)
        return field
    elseif init isa Function
        centers_x, centers_y = cell_centers(mesh)
        nx, ny = size(mesh)
        @inbounds for j in 1:ny, i in 1:nx
            field[i, j] = T(init(centers_x[i], centers_y[j]))
        end
        return field
    else
        throw(ArgumentError("Unsupported initializer type $(typeof(init))"))
    end
end

"""
    rk2_step!(state, problem, dt)

Advance the solution stored in `state` by a single explicit second-order
Runge-Kutta step of size `dt`.
"""
function rk2_step!(state::LinearAdvectionState,
                   problem::LinearAdvectionProblem,
                   dt::Real)
    u = solution(state)
    ws = workspace(state)
    Tsol = eltype(u)
    dtT = convert(Tsol, dt)
    half = convert(Tsol, 0.5)

    compute_rhs!(ws.k1, u, problem)
    @. ws.stage = u + dtT * ws.k1
    compute_rhs!(ws.k2, ws.stage, problem)
    @. u = u + (dtT * half) * (ws.k1 + ws.k2)

    return state
end

"""
    compute_rhs!(du, u, problem)

Populate `du` with the spatial derivative of `u` for a linear advection problem,
using a second-order upwind stencil that honors periodic boundary conditions.
"""
function compute_rhs!(du::AbstractArray{T,2},
                      u::AbstractArray{T,2},
                      problem::LinearAdvectionProblem) where {T}
    mesh_obj = mesh(problem)
    bc = boundary_conditions(problem)
    eq = pde(problem)

    axes = periodic_axes(bc)
    vel = velocity(eq)
    nx, ny = size(mesh_obj)
    dx, dy = spacing(mesh_obj)

    vel[1] == 0 || axes[1] ||
        throw(ArgumentError("Periodic boundary conditions required along x for advection"))
    vel[2] == 0 || axes[2] ||
        throw(ArgumentError("Periodic boundary conditions required along y for advection"))
    vel[1] == 0 || nx >= 3 ||
        throw(ArgumentError("At least three cells along x are required for 2nd-order advection"))
    vel[2] == 0 || ny >= 3 ||
        throw(ArgumentError("At least three cells along y are required for 2nd-order advection"))

    inv2dx = vel[1] == 0 ? zero(T) : T(1) / (T(2) * T(dx))
    inv2dy = vel[2] == 0 ? zero(T) : T(1) / (T(2) * T(dy))

    ax, ay = vel

    @inbounds for j in 1:ny
        jm1 = mod1(j - 1, ny)
        jm2 = mod1(j - 2, ny)
        jp1 = mod1(j + 1, ny)
        jp2 = mod1(j + 2, ny)
        for i in 1:nx
            im1 = mod1(i - 1, nx)
            im2 = mod1(i - 2, nx)
            ip1 = mod1(i + 1, nx)
            ip2 = mod1(i + 2, nx)

            dudx = zero(T)
            if ax > 0
                dudx = (T(3) * u[i, j] - T(4) * u[im1, j] + u[im2, j]) * inv2dx
            elseif ax < 0
                dudx = (-T(3) * u[i, j] + T(4) * u[ip1, j] - u[ip2, j]) * inv2dx
            end

            dudy = zero(T)
            if ay > 0
                dudy = (T(3) * u[i, j] - T(4) * u[i, jm1] + u[i, jm2]) * inv2dy
            elseif ay < 0
                dudy = (-T(3) * u[i, j] + T(4) * u[i, jp1] - u[i, jp2]) * inv2dy
            end

            du[i, j] = -(ax * dudx + ay * dudy)
        end
    end

    return du
end

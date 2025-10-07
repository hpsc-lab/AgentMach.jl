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
struct LinearAdvectionState{A,W,B}
    solution::A
    workspace::W
    backend::B
end

"""
    solution(state)

Return the primary solution array stored in `state`.
"""
solution(state::LinearAdvectionState) = state.solution

"""
    workspace(state)

Access the Runge-Kutta scratch buffers bundled with `state`.
"""
workspace(state::LinearAdvectionState) = state.workspace

"""
    backend(state)

Return the execution backend associated with `state`.
"""
backend(state::LinearAdvectionState) = state.backend

"""
    CompressibleEulerState(solution, workspace)

Hold the conserved variables for the compressible Euler system along with RK2
scratch storage.
"""
struct CompressibleEulerState{A,W,B}
    solution::A
    workspace::W
    backend::B
end

"""
    solution(state)

Return the conserved-variable tensor stored in `state`.
"""
solution(state::CompressibleEulerState) = state.solution

"""
    workspace(state)

Access the Runge-Kutta scratch buffers bundled with `state`.
"""
workspace(state::CompressibleEulerState) = state.workspace

backend(state::CompressibleEulerState) = state.backend

const _DefaultArrayType = Array

function LinearAdvectionState(problem::LinearAdvectionProblem;
                              T::Type = Float64,
                              array_type = _DefaultArrayType,
                              init = nothing,
                              backend::ExecutionBackend = default_backend())
    mesh_obj = mesh(problem)
    dims = size(mesh_obj)
    field = allocate_cellfield(array_type, T, dims, 1)
    _initialize_scalar_field!(field, init, mesh_obj, T)

    k1 = allocate_like(field)
    k2 = allocate_like(field)
    stage = allocate_like(field)
    fill!(k1, zero(T))
    fill!(k2, zero(T))
    fill!(stage, zero(T))

    return LinearAdvectionState(field, RK2Workspace(k1, k2, stage), backend)
end

function _initialize_scalar_field!(field::CellField, init, mesh::StructuredMesh, ::Type{T}) where {T}
    data = component(field, 1)
    if init === nothing
        fill!(data, zero(T))
    elseif init isa Number
        fill!(data, T(init))
    elseif init isa AbstractArray
        size(init) == size(data) ||
            throw(ArgumentError("Initializer array must match mesh dimensions"))
        data .= T.(init)
    elseif init isa Function
        centers_x, centers_y = cell_centers(mesh)
        nx, ny = size(mesh)
        @inbounds for j in 1:ny, i in 1:nx
            data[i, j] = T(init(centers_x[i], centers_y[j]))
        end
    else
        throw(ArgumentError("Unsupported initializer type $(typeof(init))"))
    end
    return field
end

const _EulerVarCount = 4

function CompressibleEulerState(problem::CompressibleEulerProblem;
                                T::Type = Float64,
                                array_type = _DefaultArrayType,
                                init = nothing,
                                backend::ExecutionBackend = default_backend())
    mesh_obj = mesh(problem)
    dims = size(mesh_obj)
    eq = pde(problem)

    field = allocate_cellfield(array_type, T, dims, _EulerVarCount)
    _initialize_euler_field!(field, init, mesh_obj, eq, T)

    k1 = allocate_like(field)
    k2 = allocate_like(field)
    stage = allocate_like(field)
    fill!(k1, zero(T))
    fill!(k2, zero(T))
    fill!(stage, zero(T))

    return CompressibleEulerState(field, RK2Workspace(k1, k2, stage), backend)
end

function _initialize_euler_field!(field::CellField,
                                  init,
                                  mesh::StructuredMesh,
                                  equation::CompressibleEuler,
                                  ::Type{T}) where {T}
    if init === nothing
        fill!(field, zero(T))
        return field
    elseif init isa Number
        fill!(field, T(init))
        return field
    elseif init isa AbstractArray
        size(init) == size(field) ||
            throw(ArgumentError("Initializer array must match (4, nx, ny) layout"))
        for comp in 1:ncomponents(field)
            component(field, comp) .= T.(view(init, comp, :, :))
        end
        return field
    elseif init isa Function
        centers_x, centers_y = cell_centers(mesh)
        nx, ny = size(mesh)
        γ = gamma(equation)
        @inbounds for j in 1:ny, i in 1:nx
            ρ, u, v, p = _extract_primitive(init, centers_x[i], centers_y[j])
            _store_conserved!(field, i, j, ρ, u, v, p, γ, T)
        end
        return field
    else
        throw(ArgumentError("Unsupported initializer type $(typeof(init)) for Euler state"))
    end
end

@inline function _extract_primitive(func::Function, x, y)
    val = func(x, y)
    if val isa NamedTuple
        ρ = get(val, :rho, nothing)
        u = get(val, :u, get(val, :v1, nothing))
        v = get(val, :v, get(val, :v2, nothing))
        p = get(val, :p, nothing)
        (ρ === nothing || u === nothing || v === nothing || p === nothing) &&
            throw(ArgumentError("Initializer NamedTuple must provide :rho, :u/:v1, :v/:v2, and :p"))
        return float(ρ), float(u), float(v), float(p)
    elseif val isa Tuple || val isa AbstractVector
        length(val) == 4 ||
            throw(ArgumentError("Initializer tuple/vector must have (rho, u, v, p)"))
        return float(val[1]), float(val[2]), float(val[3]), float(val[4])
    else
        throw(ArgumentError("Initializer function must return NamedTuple or tuple of primitives"))
    end
end

@inline function _store_conserved!(field, i, j, ρ, u, v, p, γ, ::Type{T}) where {T}
    ρT = T(ρ)
    uT = T(u)
    vT = T(v)
    pT = T(p)
    γT = T(γ)
    kinetic = T(0.5) * ρT * (uT^2 + vT^2)
    energy = pT / (γT - T(1)) + kinetic
    field[1, i, j] = ρT
    field[2, i, j] = ρT * uT
    field[3, i, j] = ρT * vT
    field[4, i, j] = energy
    return field
end

"""
    rk2_step!(state, problem, dt)

Advance the solution stored in `state` by a single explicit second-order
Runge-Kutta step of size `dt`.
"""
function rk2_step!(state::LinearAdvectionState,
                   problem::LinearAdvectionProblem,
                   dt::Real)
    return _rk2_step!(state, problem, dt)
end

function rk2_step!(state::CompressibleEulerState,
                   problem::CompressibleEulerProblem,
                   dt::Real)
    return _rk2_step!(state, problem, dt)
end

function _rk2_step!(state, problem, dt::Real)
    backend_obj = backend(state)
    if backend_obj isa SerialBackend
        return _rk2_step_serial!(state, problem, dt)
    elseif backend_obj isa KernelAbstractionsBackend
        throw(ArgumentError("KernelAbstractions backends are not yet implemented for time integration"))
    else
        throw(ArgumentError("Unsupported execution backend $(describe(backend_obj))"))
    end
end

function _rk2_step_serial!(state, problem, dt::Real)
    u = solution(state)
    ws = workspace(state)
    Tsol = eltype(u)
    dtT = convert(Tsol, dt)
    half = convert(Tsol, 0.5)
    timer = simulation_timers()

    @timeit timer "RK2 step" begin
        @timeit timer "RHS compute" compute_rhs!(ws.k1, u, problem)
        @timeit timer "Stage predictor" _rk2_stage_predict!(ws.stage, u, ws.k1, dtT)
        @timeit timer "RHS compute" compute_rhs!(ws.k2, ws.stage, problem)
        @timeit timer "Solution update" _rk2_solution_update!(u, ws.k1, ws.k2, dtT * half)
    end

    return state
end

@inline function _rk2_stage_predict!(stage::CellField,
                                     u::CellField,
                                     rhs::CellField,
                                     α)
    n = ncomponents(stage)
    @inbounds for comp in 1:n
        stage_comp = component(stage, comp)
        u_comp = component(u, comp)
        rhs_comp = component(rhs, comp)
        @. stage_comp = u_comp + α * rhs_comp
    end
    return stage
end

@inline function _rk2_solution_update!(u::CellField,
                                       k1::CellField,
                                       k2::CellField,
                                       α)
    n = ncomponents(u)
    @inbounds for comp in 1:n
        u_comp = component(u, comp)
        k1_comp = component(k1, comp)
        k2_comp = component(k2, comp)
        @. u_comp = u_comp + α * (k1_comp + k2_comp)
    end
    return u
end

"""
    compute_rhs!(du, u, problem)

Populate `du` with the spatial derivative of `u` for the PDE described by
`problem`. Linear advection problems use a second-order upwind stencil, while
compressible Euler problems employ slope-limited Rusanov fluxes.
"""
function compute_rhs!(du::CellField,
                      u::CellField,
                      problem::LinearAdvectionProblem)
    return _linear_advection_rhs!(component(du, 1), component(u, 1), problem)
end

function compute_rhs!(du::AbstractArray{T,2},
                      u::AbstractArray{T,2},
                      problem::LinearAdvectionProblem) where {T}
    return _linear_advection_rhs!(du, u, problem)
end

function _linear_advection_rhs!(du::AbstractArray{T,2},
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

function compute_rhs!(du::CellField,
                      u::CellField,
                      problem::CompressibleEulerProblem)
    return _compressible_euler_rhs!(du, u, problem)
end

function compute_rhs!(du::AbstractArray{T,3},
                      u::AbstractArray{T,3},
                      problem::CompressibleEulerProblem) where {T}
    du_field = CellField(ntuple(i -> view(du, i, :, :), Val(_EulerVarCount)))
    u_field = CellField(ntuple(i -> view(u, i, :, :), Val(_EulerVarCount)))
    _compressible_euler_rhs!(du_field, u_field, problem)
    return du
end

function _compressible_euler_rhs!(du::CellField,
                                  u::CellField,
                                  problem::CompressibleEulerProblem)
    mesh_obj = mesh(problem)
    bc = boundary_conditions(problem)
    eq = pde(problem)
    bc isa PeriodicBoundaryConditions ||
        throw(ArgumentError("Compressible Euler RHS currently supports only periodic boundary conditions"))
    axes = periodic_axes(bc)
    axes == (true, true) ||
        throw(ArgumentError("Compressible Euler RHS currently supports only fully periodic boundaries"))

    nx, ny = size(mesh_obj)
    nx >= 3 || throw(ArgumentError("At least three cells required along x"))
    ny >= 3 || throw(ArgumentError("At least three cells required along y"))

    T = component_eltype(u)
    dx, dy = spacing(mesh_obj)
    inv_dx = one(T) / T(dx)
    inv_dy = one(T) / T(dy)
    γ = gamma(eq)
    fill!(du, zero(T))

    ρ = component(u, 1)
    rhou = component(u, 2)
    rhov = component(u, 3)
    E = component(u, 4)

    dρ = component(du, 1)
    drhou = component(du, 2)
    drhov = component(du, 3)
    dE = component(du, 4)

    # x-direction fluxes
    @inbounds for j in 1:ny
        jm1 = j == 1 ? ny : j - 1
        jp1 = j == ny ? 1 : j + 1
        for i in 1:nx
            ip = i == nx ? 1 : i + 1
            ip2 = ip == nx ? 1 : ip + 1
            im = i == 1 ? nx : i - 1

            # Slopes for cell i
            ΔLρ = ρ[i, j] - ρ[im, j]
            ΔRρ = ρ[ip, j] - ρ[i, j]
            ΔLrhox = rhou[i, j] - rhou[im, j]
            ΔRrhox = rhou[ip, j] - rhou[i, j]
            ΔLrhoy = rhov[i, j] - rhov[im, j]
            ΔRrhoy = rhov[ip, j] - rhov[i, j]
            ΔLE = E[i, j] - E[im, j]
            ΔRE = E[ip, j] - E[i, j]

            sρ = _minmod(ΔLρ, ΔRρ)
            srhox = _minmod(ΔLrhox, ΔRrhox)
            srhoy = _minmod(ΔLrhoy, ΔRrhoy)
            sE = _minmod(ΔLE, ΔRE)

            ρL = ρ[i, j] + T(0.5) * sρ
            rhouL = rhou[i, j] + T(0.5) * srhox
            rhovL = rhov[i, j] + T(0.5) * srhoy
            EL = E[i, j] + T(0.5) * sE

            # Slopes for cell ip
            ΔLρ_ip = ρ[ip, j] - ρ[i, j]
            ΔRρ_ip = ρ[ip2, j] - ρ[ip, j]
            ΔLrhox_ip = rhou[ip, j] - rhou[i, j]
            ΔRrhox_ip = rhou[ip2, j] - rhou[ip, j]
            ΔLrhoy_ip = rhov[ip, j] - rhov[i, j]
            ΔRrhoy_ip = rhov[ip2, j] - rhov[ip, j]
            ΔLE_ip = E[ip, j] - E[i, j]
            ΔRE_ip = E[ip2, j] - E[ip, j]

            sρ_ip = _minmod(ΔLρ_ip, ΔRρ_ip)
            srhox_ip = _minmod(ΔLrhox_ip, ΔRrhox_ip)
            srhoy_ip = _minmod(ΔLrhoy_ip, ΔRrhoy_ip)
            sE_ip = _minmod(ΔLE_ip, ΔRE_ip)

            ρR = ρ[ip, j] - T(0.5) * sρ_ip
            rhouR = rhou[ip, j] - T(0.5) * srhox_ip
            rhovR = rhov[ip, j] - T(0.5) * srhoy_ip
            ER = E[ip, j] - T(0.5) * sE_ip

            flux1, flux2, flux3, flux4 = _rusanov_flux_x(eq, ρL, rhouL, rhovL, EL,
                                                         ρR, rhouR, rhovR, ER)

            dρ[i, j] -= flux1 * inv_dx
            drhou[i, j] -= flux2 * inv_dx
            drhov[i, j] -= flux3 * inv_dx
            dE[i, j] -= flux4 * inv_dx

            dρ[ip, j] += flux1 * inv_dx
            drhou[ip, j] += flux2 * inv_dx
            drhov[ip, j] += flux3 * inv_dx
            dE[ip, j] += flux4 * inv_dx
        end
    end

    # y-direction fluxes
    @inbounds for j in 1:ny
        jp = j == ny ? 1 : j + 1
        jp2 = jp == ny ? 1 : jp + 1
        jm = j == 1 ? ny : j - 1
        for i in 1:nx
            ΔLρ = ρ[i, j] - ρ[i, jm]
            ΔRρ = ρ[i, jp] - ρ[i, j]
            ΔLrhox = rhou[i, j] - rhou[i, jm]
            ΔRrhox = rhou[i, jp] - rhou[i, j]
            ΔLrhoy = rhov[i, j] - rhov[i, jm]
            ΔRrhoy = rhov[i, jp] - rhov[i, j]
            ΔLE = E[i, j] - E[i, jm]
            ΔRE = E[i, jp] - E[i, j]

            sρ = _minmod(ΔLρ, ΔRρ)
            srhox = _minmod(ΔLrhox, ΔRrhox)
            srhoy = _minmod(ΔLrhoy, ΔRrhoy)
            sE = _minmod(ΔLE, ΔRE)

            ρL = ρ[i, j] + T(0.5) * sρ
            rhouL = rhou[i, j] + T(0.5) * srhox
            rhovL = rhov[i, j] + T(0.5) * srhoy
            EL = E[i, j] + T(0.5) * sE

            ΔLρ_jp = ρ[i, jp] - ρ[i, j]
            ΔRρ_jp = ρ[i, jp2] - ρ[i, jp]
            ΔLrhox_jp = rhou[i, jp] - rhou[i, j]
            ΔRrhox_jp = rhou[i, jp2] - rhou[i, jp]
            ΔLrhoy_jp = rhov[i, jp] - rhov[i, j]
            ΔRrhoy_jp = rhov[i, jp2] - rhov[i, jp]
            ΔLE_jp = E[i, jp] - E[i, j]
            ΔRE_jp = E[i, jp2] - E[i, jp]

            sρ_jp = _minmod(ΔLρ_jp, ΔRρ_jp)
            srhox_jp = _minmod(ΔLrhox_jp, ΔRrhox_jp)
            srhoy_jp = _minmod(ΔLrhoy_jp, ΔRrhoy_jp)
            sE_jp = _minmod(ΔLE_jp, ΔRE_jp)

            ρR = ρ[i, jp] - T(0.5) * sρ_jp
            rhouR = rhou[i, jp] - T(0.5) * srhox_jp
            rhovR = rhov[i, jp] - T(0.5) * srhoy_jp
            ER = E[i, jp] - T(0.5) * sE_jp

            flux1, flux2, flux3, flux4 = _rusanov_flux_y(eq, ρL, rhouL, rhovL, EL,
                                                         ρR, rhouR, rhovR, ER)

            dρ[i, j] -= flux1 * inv_dy
            drhou[i, j] -= flux2 * inv_dy
            drhov[i, j] -= flux3 * inv_dy
            dE[i, j] -= flux4 * inv_dy

            dρ[i, jp] += flux1 * inv_dy
            drhou[i, jp] += flux2 * inv_dy
            drhov[i, jp] += flux3 * inv_dy
            dE[i, jp] += flux4 * inv_dy
        end
    end

    return du
end

"""
    cfl_number(problem, dt)

Compute the nondimensional Courant-Friedrichs-Lewy number for the provided
linear advection problem and timestep length `dt`.
"""
function cfl_number(problem::LinearAdvectionProblem, dt::Real)
    mesh_obj = mesh(problem)
    vel = velocity(pde(problem))
    dx, dy = spacing(mesh_obj)

    ax = abs(vel[1])
    ay = abs(vel[2])

    dtT = float(dt)
    termx = ax == 0 ? zero(dtT) : dtT * ax / dx
    termy = ay == 0 ? zero(dtT) : dtT * ay / dy

    return termx + termy
end

"""
    stable_timestep(problem; cfl = 0.9)

Return a timestep size that satisfies `cfl_number(problem, dt) <= cfl` for the
explicit RK2 integrator. The default `cfl = 0.9` provides a modest safety margin
under the ideal limit of 1.0.
"""
function stable_timestep(problem::LinearAdvectionProblem; cfl::Real = 0.9)
    cfl > 0 || throw(ArgumentError("CFL target must be positive"))

    mesh_obj = mesh(problem)
    vel = velocity(pde(problem))
    dx, dy = spacing(mesh_obj)

    ax = abs(vel[1])
    ay = abs(vel[2])

    denom = (ax == 0 ? zero(float(cfl)) : ax / dx) +
            (ay == 0 ? zero(float(cfl)) : ay / dy)

    iszero(denom) && return Inf

    return float(cfl) / denom
end

"""
    cfl_number(problem, state, dt)

Compute the CFL number implied by timestep `dt` for the compressible Euler
problem `problem` evaluated on the state `state`.
"""
function cfl_number(problem::CompressibleEulerProblem,
                    state::CompressibleEulerState,
                    dt::Real)
    mesh_obj = mesh(problem)
    dx, dy = spacing(mesh_obj)
    γ = gamma(pde(problem))
    u = solution(state)
    nx, ny = size(mesh_obj)

    max_ax = zero(eltype(u))
    max_ay = zero(eltype(u))

    ρ = component(u, 1)
    rhou = component(u, 2)
    rhov = component(u, 3)
    E = component(u, 4)

    @inbounds for j in 1:ny, i in 1:nx
        ρval = ρ[i, j]
        rhouval = rhou[i, j]
        rhovval = rhov[i, j]
        Eval = E[i, j]
        invρ = one(ρval) / ρval
        ux = rhouval * invρ
        uy = rhovval * invρ
        kinetic = 0.5 * ρval * (ux^2 + uy^2)
        p = (γ - one(γ)) * (Eval - kinetic)
        c = sqrt(abs(γ * p * invρ))
        max_ax = max(max_ax, abs(ux) + c)
        max_ay = max(max_ay, abs(uy) + c)
    end

    dtT = float(dt)
    return dtT * (max_ax / dx + max_ay / dy)
end

"""
    stable_timestep(problem, state; cfl=0.45)

Return a CFL-limited timestep estimate for the compressible Euler problem using
the current `state` and requested `cfl` target.
"""
function stable_timestep(problem::CompressibleEulerProblem,
                         state::CompressibleEulerState;
                         cfl::Real = 0.45)
    cfl > 0 || throw(ArgumentError("CFL target must be positive"))

    mesh_obj = mesh(problem)
    dx, dy = spacing(mesh_obj)
    γ = gamma(pde(problem))
    u = solution(state)
    nx, ny = size(mesh_obj)

    max_ax = zero(eltype(u))
    max_ay = zero(eltype(u))

    ρ = component(u, 1)
    rhou = component(u, 2)
    rhov = component(u, 3)
    E = component(u, 4)

    @inbounds for j in 1:ny, i in 1:nx
        ρval = ρ[i, j]
        rhouval = rhou[i, j]
        rhovval = rhov[i, j]
        Eval = E[i, j]
        invρ = one(ρval) / ρval
        ux = rhouval * invρ
        uy = rhovval * invρ
        kinetic = 0.5 * ρval * (ux^2 + uy^2)
        p = (γ - one(γ)) * (Eval - kinetic)
        c = sqrt(abs(γ * p * invρ))
        max_ax = max(max_ax, abs(ux) + c)
        max_ay = max(max_ay, abs(uy) + c)
    end

    denom = (max_ax == 0 ? zero(Float64) : max_ax / dx) +
            (max_ay == 0 ? zero(Float64) : max_ay / dy)

    iszero(denom) && return Inf

    return float(cfl) / denom
end

# Utility micro-kernels for Euler RHS

@inline function _minmod(a, b)
    if a * b <= 0
        return zero(promote_type(typeof(a), typeof(b)))
    end
    return copysign(min(abs(a), abs(b)), a)
end

@inline function _thermodynamics(eq::CompressibleEuler, ρ, rhou, rhov, E)
    prim = primitive_variables(eq, ρ, rhou, rhov, E)
    return prim.u, prim.v, prim.p
end

@inline function _rusanov_flux_x(eq::CompressibleEuler,
                                 ρL, rhouL, rhovL, EL,
                                 ρR, rhouR, rhovR, ER)
    uxL, uyL, pL = _thermodynamics(eq, ρL, rhouL, rhovL, EL)
    uxR, uyR, pR = _thermodynamics(eq, ρR, rhouR, rhovR, ER)
    γ = gamma(eq)
    cL = sqrt(abs(γ * pL / ρL))
    cR = sqrt(abs(γ * pR / ρR))
    smax = max(abs(uxL) + cL, abs(uxR) + cR)

    FL1 = rhouL
    FL2 = rhouL * uxL + pL
    FL3 = rhouL * uyL
    FL4 = (EL + pL) * uxL

    FR1 = rhouR
    FR2 = rhouR * uxR + pR
    FR3 = rhouR * uyR
    FR4 = (ER + pR) * uxR

    flux1 = 0.5 * (FL1 + FR1) - 0.5 * smax * (ρR - ρL)
    flux2 = 0.5 * (FL2 + FR2) - 0.5 * smax * (rhouR - rhouL)
    flux3 = 0.5 * (FL3 + FR3) - 0.5 * smax * (rhovR - rhovL)
    flux4 = 0.5 * (FL4 + FR4) - 0.5 * smax * (ER - EL)

    return flux1, flux2, flux3, flux4
end

@inline function _rusanov_flux_y(eq::CompressibleEuler,
                                 ρL, rhouL, rhovL, EL,
                                 ρR, rhouR, rhovR, ER)
    uxL, uyL, pL = _thermodynamics(eq, ρL, rhouL, rhovL, EL)
    uxR, uyR, pR = _thermodynamics(eq, ρR, rhouR, rhovR, ER)
    γ = gamma(eq)
    cL = sqrt(abs(γ * pL / ρL))
    cR = sqrt(abs(γ * pR / ρR))
    smax = max(abs(uyL) + cL, abs(uyR) + cR)

    GL1 = rhovL
    GL2 = rhovL * uxL
    GL3 = rhovL * uyL + pL
    GL4 = (EL + pL) * uyL

    GR1 = rhovR
    GR2 = rhovR * uxR
    GR3 = rhovR * uyR + pR
    GR4 = (ER + pR) * uyR

    flux1 = 0.5 * (GL1 + GR1) - 0.5 * smax * (ρR - ρL)
    flux2 = 0.5 * (GL2 + GR2) - 0.5 * smax * (rhouR - rhouL)
    flux3 = 0.5 * (GL3 + GR3) - 0.5 * smax * (rhovR - rhovL)
    flux4 = 0.5 * (GL4 + GR4) - 0.5 * smax * (ER - EL)

    return flux1, flux2, flux3, flux4
end

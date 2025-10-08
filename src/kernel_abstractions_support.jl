using KernelAbstractions

const _DEVICE_REGISTRY = Dict{Symbol,Function}()

function register_backend!(name::Symbol, factory::Function)
    _DEVICE_REGISTRY[name] = factory
    return name
end

available_backends() = collect(keys(_DEVICE_REGISTRY))

function _resolve_ka_device(spec)
    if spec isa Symbol
        factory = get(_DEVICE_REGISTRY, spec) do
            throw(ArgumentError("Unsupported KernelAbstractions device spec $(spec); supported values: $(join(sort!(collect(keys(_DEVICE_REGISTRY))), ", "))"))
        end
        return factory()
    elseif spec isa KernelAbstractions.AbstractDevice
        return spec
    else
        throw(ArgumentError("KernelAbstractionsBackend device must be a Symbol or device, got $(typeof(spec))"))
    end
end

@inline function _ka_velocity_pressure(γ, ρ, rhou, rhov, E)
    invρ = one(ρ) / ρ
    ux = rhou * invρ
    uy = rhov * invρ
    kinetic = (one(ρ) / 2) * ρ * (ux^2 + uy^2)
    p = (γ - one(γ)) * (E - kinetic)
    return ux, uy, p
end

@inline function _ka_rusanov_flux_x(γ, ρL, rhouL, rhovL, EL,
                                    ρR, rhouR, rhovR, ER)
    T = promote_type(typeof(ρL), typeof(ρR))
    half = inv(T(2))
    uxL, uyL, pL = _ka_velocity_pressure(γ, ρL, rhouL, rhovL, EL)
    uxR, uyR, pR = _ka_velocity_pressure(γ, ρR, rhouR, rhovR, ER)
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

    flux1 = half * (FL1 + FR1) - half * smax * (ρR - ρL)
    flux2 = half * (FL2 + FR2) - half * smax * (rhouR - rhouL)
    flux3 = half * (FL3 + FR3) - half * smax * (rhovR - rhovL)
    flux4 = half * (FL4 + FR4) - half * smax * (ER - EL)

    return flux1, flux2, flux3, flux4
end

@inline function _ka_rusanov_flux_y(γ, ρL, rhouL, rhovL, EL,
                                    ρR, rhouR, rhovR, ER)
    T = promote_type(typeof(ρL), typeof(ρR))
    half = inv(T(2))
    uxL, uyL, pL = _ka_velocity_pressure(γ, ρL, rhouL, rhovL, EL)
    uxR, uyR, pR = _ka_velocity_pressure(γ, ρR, rhouR, rhovR, ER)
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

    flux1 = half * (GL1 + GR1) - half * smax * (ρR - ρL)
    flux2 = half * (GL2 + GR2) - half * smax * (rhouR - rhouL)
    flux3 = half * (GL3 + GR3) - half * smax * (rhovR - rhovL)
    flux4 = half * (GL4 + GR4) - half * smax * (ER - EL)

    return flux1, flux2, flux3, flux4
end

@kernel function _linear_advection_rhs_kernel!(du, u, ax, ay, inv2dx, inv2dy)
    i, j = @index(Global, NTuple)
    nx, ny = size(u)
    if i <= nx && j <= ny
        im1 = i == 1 ? nx : i - 1
        im2 = im1 == 1 ? nx : im1 - 1
        ip1 = i == nx ? 1 : i + 1
        ip2 = ip1 == nx ? 1 : ip1 + 1

        jm1 = j == 1 ? ny : j - 1
        jm2 = jm1 == 1 ? ny : jm1 - 1
        jp1 = j == ny ? 1 : j + 1
        jp2 = jp1 == ny ? 1 : jp1 + 1

        zero_ax = zero(ax)
        dudx = zero_ax
        if ax > zero_ax
            dudx = (3 * u[i, j] - 4 * u[im1, j] + u[im2, j]) * inv2dx
        elseif ax < zero_ax
            dudx = (-3 * u[i, j] + 4 * u[ip1, j] - u[ip2, j]) * inv2dx
        end

        zero_ay = zero(ay)
        dudy = zero_ay
        if ay > zero_ay
            dudy = (3 * u[i, j] - 4 * u[i, jm1] + u[i, jm2]) * inv2dy
        elseif ay < zero_ay
            dudy = (-3 * u[i, j] + 4 * u[i, jp1] - u[i, jp2]) * inv2dy
        end

        du[i, j] = -(ax * dudx + ay * dudy)
    end
end

@kernel function _rk2_stage_kernel!(stage, u, rhs, dt)
    I = @index(Global)
    if I <= length(stage)
        stage[I] = u[I] + dt * rhs[I]
    end
end

@kernel function _rk2_update_kernel!(u, k1, k2, factor)
    I = @index(Global)
    if I <= length(u)
        u[I] = u[I] + factor * (k1[I] + k2[I])
    end
end

@kernel function _primitive_variables_kernel!(ρ_out, u_out, v_out, p_out,
                                              ρc, rhouc, rhovc, Ec,
                                              γm1, epsT)
    i, j = @index(Global, NTuple)
    nx, ny = size(ρc)
    if i <= nx && j <= ny
        ρval = ρc[i, j]
        ρval = ρval < epsT ? epsT : ρval
        invρ = one(ρval) / ρval
        ux = rhouc[i, j] * invρ
        uy = rhovc[i, j] * invρ
        half = inv(convert(typeof(ρval), 2))
        kinetic = half * ρval * (ux^2 + uy^2)
        internal = Ec[i, j] - kinetic
        internal = internal < epsT ? epsT : internal
        p = γm1 * internal
        p = p < epsT ? epsT : p

        ρ_out[i, j] = ρval
        u_out[i, j] = ux
        v_out[i, j] = uy
        p_out[i, j] = p
    end
end

function linear_advection_rhs_kernel!(backend::KernelAbstractionsBackend,
                                      du, u,
                                      nx::Int, ny::Int,
                                      ax, ay,
                                      inv2dx, inv2dy)
    device = _resolve_ka_device(backend.device)
    kernel = backend.workgroupsize === nothing ?
        _linear_advection_rhs_kernel!(device) :
        _linear_advection_rhs_kernel!(device, backend.workgroupsize)
    kernel(du, u, ax, ay, inv2dx, inv2dy; ndrange = (nx, ny))
    KernelAbstractions.synchronize(device)
    return du
end

function rk2_stage_kernel!(backend::KernelAbstractionsBackend,
                           stage, u, rhs, dt)
    device = _resolve_ka_device(backend.device)
    kernel = backend.workgroupsize === nothing ?
        _rk2_stage_kernel!(device) :
        _rk2_stage_kernel!(device, backend.workgroupsize)
    kernel(stage, u, rhs, dt; ndrange = length(stage))
    KernelAbstractions.synchronize(device)
    return stage
end

function rk2_update_kernel!(backend::KernelAbstractionsBackend,
                            u, k1, k2, factor)
    device = _resolve_ka_device(backend.device)
    kernel = backend.workgroupsize === nothing ?
        _rk2_update_kernel!(device) :
        _rk2_update_kernel!(device, backend.workgroupsize)
    kernel(u, k1, k2, factor; ndrange = length(u))
    KernelAbstractions.synchronize(device)
    return u
end

function primitive_variables_kernel!(backend::KernelAbstractionsBackend,
                                     ρ_out, u_out, v_out, p_out,
                                     ρc, rhouc, rhovc, Ec,
                                     γm1, epsT)
    device = _resolve_ka_device(backend.device)
    kernel = backend.workgroupsize === nothing ?
        _primitive_variables_kernel!(device) :
        _primitive_variables_kernel!(device, backend.workgroupsize)
    kernel(ρ_out, u_out, v_out, p_out,
           ρc, rhouc, rhovc, Ec,
           γm1, epsT; ndrange = size(ρc))
    KernelAbstractions.synchronize(device)
    return ρ_out, u_out, v_out, p_out
end

@kernel function _compressible_euler_rhs_kernel!(dρ, drhou, drhov, dE,
                                                 ρ, rhou, rhov, E,
                                                 γ, inv_dx, inv_dy, lim)
    i, j = @index(Global, NTuple)
    nx, ny = size(ρ)
    if i <= nx && j <= ny
        ip = i == nx ? 1 : i + 1
        ip2 = ip == nx ? 1 : ip + 1
        im = i == 1 ? nx : i - 1
        im2 = im == 1 ? nx : im - 1

        jp = j == ny ? 1 : j + 1
        jp2 = jp == ny ? 1 : jp + 1
        jm = j == 1 ? ny : j - 1
        jm2 = jm == 1 ? ny : jm - 1

        half = typeof(ρ[i, j])(0.5)

        # Slopes for cell i
        ΔLρ = ρ[i, j] - ρ[im, j]
        ΔRρ = ρ[ip, j] - ρ[i, j]
        ΔLrhox = rhou[i, j] - rhou[im, j]
        ΔRrhox = rhou[ip, j] - rhou[i, j]
        ΔLrhoy = rhov[i, j] - rhov[im, j]
        ΔRrhoy = rhov[ip, j] - rhov[i, j]
        ΔLE = E[i, j] - E[im, j]
        ΔRE = E[ip, j] - E[i, j]

        sρ = apply_limiter(lim, ΔLρ, ΔRρ)
        srhox = apply_limiter(lim, ΔLrhox, ΔRrhox)
        srhoy = apply_limiter(lim, ΔLrhoy, ΔRrhoy)
        sE = apply_limiter(lim, ΔLE, ΔRE)

        ρL_plus = ρ[i, j] + half * sρ
        rhouL_plus = rhou[i, j] + half * srhox
        rhovL_plus = rhov[i, j] + half * srhoy
        EL_plus = E[i, j] + half * sE

        # Slopes for cell ip
        ΔLρ_ip = ρ[ip, j] - ρ[i, j]
        ΔRρ_ip = ρ[ip2, j] - ρ[ip, j]
        ΔLrhox_ip = rhou[ip, j] - rhou[i, j]
        ΔRrhox_ip = rhou[ip2, j] - rhou[ip, j]
        ΔLrhoy_ip = rhov[ip, j] - rhov[i, j]
        ΔRrhoy_ip = rhov[ip2, j] - rhov[ip, j]
        ΔLE_ip = E[ip, j] - E[i, j]
        ΔRE_ip = E[ip2, j] - E[ip, j]

        sρ_ip = apply_limiter(lim, ΔLρ_ip, ΔRρ_ip)
        srhox_ip = apply_limiter(lim, ΔLrhox_ip, ΔRrhox_ip)
        srhoy_ip = apply_limiter(lim, ΔLrhoy_ip, ΔRrhoy_ip)
        sE_ip = apply_limiter(lim, ΔLE_ip, ΔRE_ip)

        ρR_plus = ρ[ip, j] - half * sρ_ip
        rhouR_plus = rhou[ip, j] - half * srhox_ip
        rhovR_plus = rhov[ip, j] - half * srhoy_ip
        ER_plus = E[ip, j] - half * sE_ip

        flux1_plus, flux2_plus, flux3_plus, flux4_plus =
            _ka_rusanov_flux_x(γ, ρL_plus, rhouL_plus, rhovL_plus, EL_plus,
                               ρR_plus, rhouR_plus, rhovR_plus, ER_plus)

        # Interface i-1/2
        ΔLρ_im = ρ[im, j] - ρ[im2, j]
        ΔRρ_im = ρ[i, j] - ρ[im, j]
        ΔLrhox_im = rhou[im, j] - rhou[im2, j]
        ΔRrhox_im = rhou[i, j] - rhou[im, j]
        ΔLrhoy_im = rhov[im, j] - rhov[im2, j]
        ΔRrhoy_im = rhov[i, j] - rhov[im, j]
        ΔLE_im = E[im, j] - E[im2, j]
        ΔRE_im = E[i, j] - E[im, j]

        sρ_im = apply_limiter(lim, ΔLρ_im, ΔRρ_im)
        srhox_im = apply_limiter(lim, ΔLrhox_im, ΔRrhox_im)
        srhoy_im = apply_limiter(lim, ΔLrhoy_im, ΔRrhoy_im)
        sE_im = apply_limiter(lim, ΔLE_im, ΔRE_im)

        ρL_minus = ρ[im, j] + half * sρ_im
        rhouL_minus = rhou[im, j] + half * srhox_im
        rhovL_minus = rhov[im, j] + half * srhoy_im
        EL_minus = E[im, j] + half * sE_im

        ρR_minus = ρ[i, j] - half * sρ
        rhouR_minus = rhou[i, j] - half * srhox
        rhovR_minus = rhov[i, j] - half * srhoy
        ER_minus = E[i, j] - half * sE

        flux1_minus, flux2_minus, flux3_minus, flux4_minus =
            _ka_rusanov_flux_x(γ, ρL_minus, rhouL_minus, rhovL_minus, EL_minus,
                               ρR_minus, rhouR_minus, rhovR_minus, ER_minus)

        # Y-direction interface j+1/2
        ΔLρ_y = ρ[i, j] - ρ[i, jm]
        ΔRρ_y = ρ[i, jp] - ρ[i, j]
        ΔLrhox_y = rhou[i, j] - rhou[i, jm]
        ΔRrhox_y = rhou[i, jp] - rhou[i, j]
        ΔLrhoy_y = rhov[i, j] - rhov[i, jm]
        ΔRrhoy_y = rhov[i, jp] - rhov[i, j]
        ΔLE_y = E[i, j] - E[i, jm]
        ΔRE_y = E[i, jp] - E[i, j]

        sρ_y = apply_limiter(lim, ΔLρ_y, ΔRρ_y)
        srhox_y = apply_limiter(lim, ΔLrhox_y, ΔRrhox_y)
        srhoy_y = apply_limiter(lim, ΔLrhoy_y, ΔRrhoy_y)
        sE_y = apply_limiter(lim, ΔLE_y, ΔRE_y)

        ρL_plus_y = ρ[i, j] + half * sρ_y
        rhouL_plus_y = rhou[i, j] + half * srhox_y
        rhovL_plus_y = rhov[i, j] + half * srhoy_y
        EL_plus_y = E[i, j] + half * sE_y

        ΔLρ_jp = ρ[i, jp] - ρ[i, j]
        ΔRρ_jp = ρ[i, jp2] - ρ[i, jp]
        ΔLrhox_jp = rhou[i, jp] - rhou[i, j]
        ΔRrhox_jp = rhou[i, jp2] - rhou[i, jp]
        ΔLrhoy_jp = rhov[i, jp] - rhov[i, j]
        ΔRrhoy_jp = rhov[i, jp2] - rhov[i, jp]
        ΔLE_jp = E[i, jp] - E[i, j]
        ΔRE_jp = E[i, jp2] - E[i, jp]

        sρ_jp = apply_limiter(lim, ΔLρ_jp, ΔRρ_jp)
        srhox_jp = apply_limiter(lim, ΔLrhox_jp, ΔRrhox_jp)
        srhoy_jp = apply_limiter(lim, ΔLrhoy_jp, ΔRrhoy_jp)
        sE_jp = apply_limiter(lim, ΔLE_jp, ΔRE_jp)

        ρR_plus_y = ρ[i, jp] - half * sρ_jp
        rhouR_plus_y = rhou[i, jp] - half * srhox_jp
        rhovR_plus_y = rhov[i, jp] - half * srhoy_jp
        ER_plus_y = E[i, jp] - half * sE_jp

        flux1_plus_y, flux2_plus_y, flux3_plus_y, flux4_plus_y =
            _ka_rusanov_flux_y(γ, ρL_plus_y, rhouL_plus_y, rhovL_plus_y, EL_plus_y,
                               ρR_plus_y, rhouR_plus_y, rhovR_plus_y, ER_plus_y)

        # Interface j-1/2
        ΔLρ_jm = ρ[i, jm] - ρ[i, jm2]
        ΔRρ_jm = ρ[i, j] - ρ[i, jm]
        ΔLrhox_jm = rhou[i, jm] - rhou[i, jm2]
        ΔRrhox_jm = rhou[i, j] - rhou[i, jm]
        ΔLrhoy_jm = rhov[i, jm] - rhov[i, jm2]
        ΔRrhoy_jm = rhov[i, j] - rhov[i, jm]
        ΔLE_jm = E[i, jm] - E[i, jm2]
        ΔRE_jm = E[i, j] - E[i, jm]

        sρ_jm = apply_limiter(lim, ΔLρ_jm, ΔRρ_jm)
        srhox_jm = apply_limiter(lim, ΔLrhox_jm, ΔRrhox_jm)
        srhoy_jm = apply_limiter(lim, ΔLrhoy_jm, ΔRrhoy_jm)
        sE_jm = apply_limiter(lim, ΔLE_jm, ΔRE_jm)

        ρL_minus_y = ρ[i, jm] + half * sρ_jm
        rhouL_minus_y = rhou[i, jm] + half * srhox_jm
        rhovL_minus_y = rhov[i, jm] + half * srhoy_jm
        EL_minus_y = E[i, jm] + half * sE_jm

        ρR_minus_y = ρ[i, j] - half * sρ_y
        rhouR_minus_y = rhou[i, j] - half * srhox_y
        rhovR_minus_y = rhov[i, j] - half * srhoy_y
        ER_minus_y = E[i, j] - half * sE_y

        flux1_minus_y, flux2_minus_y, flux3_minus_y, flux4_minus_y =
            _ka_rusanov_flux_y(γ, ρL_minus_y, rhouL_minus_y, rhovL_minus_y, EL_minus_y,
                               ρR_minus_y, rhouR_minus_y, rhovR_minus_y, ER_minus_y)

        dρ_val = -(flux1_plus - flux1_minus) * inv_dx -
                 (flux1_plus_y - flux1_minus_y) * inv_dy
        drhou_val = -(flux2_plus - flux2_minus) * inv_dx -
                   (flux2_plus_y - flux2_minus_y) * inv_dy
        drhov_val = -(flux3_plus - flux3_minus) * inv_dx -
                   (flux3_plus_y - flux3_minus_y) * inv_dy
        dE_val = -(flux4_plus - flux4_minus) * inv_dx -
                 (flux4_plus_y - flux4_minus_y) * inv_dy

        dρ[i, j] = dρ_val
        drhou[i, j] = drhou_val
        drhov[i, j] = drhov_val
        dE[i, j] = dE_val
    end
end

register_backend!(:cpu, () -> KernelAbstractions.CPU())

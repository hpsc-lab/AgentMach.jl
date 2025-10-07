using KernelAbstractions

const _SUPPORTED_SYMBOLIC_DEVICES = (:cpu,)

function _resolve_ka_device(spec)
    if spec isa Symbol
        spec === :cpu && return KernelAbstractions.CPU()
        throw(ArgumentError("Unsupported KernelAbstractions device spec $(spec); supported values: $(_SUPPORTED_SYMBOLIC_DEVICES)"))
    elseif spec isa KernelAbstractions.AbstractDevice
        return spec
    else
        throw(ArgumentError("KernelAbstractionsBackend device must be a Symbol or device, got $(typeof(spec))"))
    end
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

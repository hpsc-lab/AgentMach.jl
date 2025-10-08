"""
    AbstractLimiter

Abstract supertype for MUSCL slope limiters. To implement a custom limiter,
create a subtype and define `apply_limiter(limiter, ΔL, ΔR)` returning the
limited slope given left and right one-sided differences.
"""
abstract type AbstractLimiter end

"""
    MinmodLimiter()

Classical minmod limiter that clamps slopes to the smallest magnitude when the
left/right differences agree in sign and zero otherwise.
"""
struct MinmodLimiter <: AbstractLimiter end

"""
    UnlimitedLimiter()

Simple MUSCL slope with no limiting; returns the centred average of the left and
right differences. Useful for smooth manufactured solutions or convergence
studies.
"""
struct UnlimitedLimiter <: AbstractLimiter end

"""
    minmod_limiter

Singleton instance of [`MinmodLimiter`](@ref) that can be reused across
problems without allocating a new limiter.
"""
const minmod_limiter = MinmodLimiter()

"""
    unlimited_limiter

Singleton instance of [`UnlimitedLimiter`](@ref) that disables slope limiting
and yields the centred MUSCL slope average.
"""
const unlimited_limiter = UnlimitedLimiter()

"""
    apply_limiter(limiter, ΔL, ΔR)

Apply `limiter` to the left/right MUSCL differences `ΔL` and `ΔR`, returning
the limited slope.
"""
function apply_limiter end

@inline function apply_limiter(::MinmodLimiter, ΔL, ΔR)
    S = promote_type(typeof(ΔL), typeof(ΔR))
    if ΔL * ΔR <= 0
        return zero(S)
    end
    return S(copysign(min(abs(ΔL), abs(ΔR)), ΔL))
end

@inline function apply_limiter(::UnlimitedLimiter, ΔL, ΔR)
    S = promote_type(typeof(ΔL), typeof(ΔR))
    return S((ΔL + ΔR) / 2)
end

@inline (lim::AbstractLimiter)(ΔL, ΔR) = apply_limiter(lim, ΔL, ΔR)

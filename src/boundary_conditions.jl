abstract type AbstractBoundaryConditions end

"""
    PeriodicBoundaryConditions(; x = true, y = true)

Describe periodic boundary conditions along the coordinate axes. Periodicity can
be toggled independently for the x- and y-directions.
"""
struct PeriodicBoundaryConditions <: AbstractBoundaryConditions
    axes::NTuple{2,Bool}
    function PeriodicBoundaryConditions(axes::NTuple{2,Bool})
        new(axes)
    end
end

PeriodicBoundaryConditions(; x::Bool = true, y::Bool = true) =
    PeriodicBoundaryConditions((x, y))

"""
    is_periodic(bc, axis)

Return `true` when the boundary condition is periodic along the requested axis
(`axis = 1` for x, `axis = 2` for y).
"""
function is_periodic(bc::PeriodicBoundaryConditions, axis::Integer)
    axis < 1 && throw(ArgumentError("Axis index must be positive, got $(axis)"))
    axis > 2 && throw(ArgumentError("Axis index exceeds dimensionality: $(axis)"))
    return bc.axes[axis]
end

periodic_axes(bc::PeriodicBoundaryConditions) = bc.axes

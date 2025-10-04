"""
    StructuredMesh(nx, ny; lengths=(1.0, 1.0), origin=(0.0, 0.0))

Create a 2D structured mesh with `nx` by `ny` cells, spanning the box defined by
`origin` and `origin .+ lengths`. Cell centers are positioned halfway across
each cell.
"""
struct StructuredMesh{T}
    dims::NTuple{2,Int}
    spacing::NTuple{2,T}
    origin::NTuple{2,T}
    cell_centers::NTuple{2,Vector{T}}
    function StructuredMesh{T}(dims::NTuple{2,Int},
                               spacing::NTuple{2,T},
                               origin::NTuple{2,T},
                               cell_centers::NTuple{2,Vector{T}}) where {T}
        new{T}(dims, spacing, origin, cell_centers)
    end
end

function StructuredMesh(nx::Integer, ny::Integer;
                        lengths::NTuple{2,<:Real} = (1.0, 1.0),
                        origin::NTuple{2,<:Real} = (0.0, 0.0))
    nx < 1 && throw(ArgumentError("nx must be positive, got $(nx)"))
    ny < 1 && throw(ArgumentError("ny must be positive, got $(ny)"))

    dims = (Int(nx), Int(ny))
    lengths_float = (float(lengths[1]), float(lengths[2]))
    origin_float = (float(origin[1]), float(origin[2]))
    spacing = (lengths_float[1] / dims[1], lengths_float[2] / dims[2])

    centers_x = [origin_float[1] + spacing[1] * (0.5 + i) for i in 0:dims[1]-1]
    centers_y = [origin_float[2] + spacing[2] * (0.5 + j) for j in 0:dims[2]-1]

    T = promote_type(eltype(centers_x), eltype(centers_y))
    return StructuredMesh{T}(
        dims,
        (T(spacing[1]), T(spacing[2])),
        (T(origin_float[1]), T(origin_float[2])),
        (Vector{T}(centers_x), Vector{T}(centers_y)),
    )
end

Base.size(mesh::StructuredMesh) = mesh.dims

"""
    cell_centers(mesh)

Return the tuple of vectors containing the x- and y-direction cell center
coordinates.
"""
cell_centers(mesh::StructuredMesh) = mesh.cell_centers

"""
    spacing(mesh)

Return the cell spacing `(dx, dy)`.
"""
spacing(mesh::StructuredMesh) = mesh.spacing

"""
    origin(mesh)

Return the mesh origin.
"""
origin(mesh::StructuredMesh) = mesh.origin

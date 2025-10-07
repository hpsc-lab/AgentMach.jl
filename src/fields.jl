"""
    CellField(components...)

Bundle one or more cell-centered arrays that share a common `(nx, ny)` shape.
Components are stored in a structure-of-arrays layout, which keeps per-variable
data contiguous for GPU and cache-friendly execution. `CellField` behaves like a
3D `AbstractArray` whose first dimension indexes the component.
"""
struct CellField{T,N,A<:NTuple{N,AbstractArray{T,2}}} <: AbstractArray{T,3}
    components::A
    function CellField(components::A) where {T,N,A<:NTuple{N,AbstractArray{T,2}}}
        N > 0 || throw(ArgumentError("CellField must contain at least one component"))
        dims = size(components[1])
        for (idx, comp) in pairs(components)
            ndims(comp) == 2 ||
                throw(ArgumentError("Component $(idx) must be two-dimensional"))
            size(comp) == dims ||
                throw(ArgumentError("Component $(idx) does not match CellField shape"))
        end
        return new{T,N,A}(components)
    end
end

CellField(component::AbstractArray{T,2}) where {T} =
    CellField{T,1,Tuple{typeof(component)}}((component,))

"""
    cell_components(field)

Return the tuple of component arrays stored within `field`.
"""
cell_components(field::CellField) = field.components

"""
    ncomponents(field)

Return the number of component arrays bundled inside `field`.
"""
ncomponents(::CellField{T,N}) where {T,N} = N

"""
    spatial_size(field)

Return the `(nx, ny)` logical dimensions shared by all components.
"""
spatial_size(field::CellField) = size(cell_components(field)[1])

"""
    component(field, i)

Access the `i`-th component array stored inside `field`.
"""
component(field::CellField, i::Integer) = cell_components(field)[i]

component(field::CellField, ::Val{i}) where {i} = component(field, i)

component_eltype(::CellField{T}) where {T} = T

scalar_component(field::CellField) = component(field, 1)

Base.eltype(field::CellField) = component_eltype(field)

Base.size(field::CellField) = (ncomponents(field), spatial_size(field)...)

Base.axes(field::CellField) = (Base.OneTo(ncomponents(field)),
                               axes(component(field, 1))...)

Base.IndexStyle(::Type{<:CellField}) = IndexCartesian()

@inline function Base.getindex(field::CellField, i, j, k)
    return component(field, i)[j, k]
end

@inline function Base.getindex(field::CellField{T,1}, i, j) where {T}
    return component(field, 1)[i, j]
end

@inline function Base.getindex(field::CellField{T,1}, ::Colon, ::Colon) where {T}
    return component(field, 1)[:, :]
end

@inline function Base.getindex(field::CellField, I::CartesianIndex{3})
    return field[I[1], I[2], I[3]]
end

@inline function Base.getindex(field::CellField, i, ::Colon, ::Colon)
    return component(field, i)[:, :]
end

@inline function Base.getindex(field::CellField, i, j, ::Colon)
    return component(field, i)[j, :]
end

@inline function Base.getindex(field::CellField, i, ::Colon, k)
    return component(field, i)[:, k]
end

@inline function Base.setindex!(field::CellField, value, i, j, k)
    component(field, i)[j, k] = value
    return field
end

@inline function Base.setindex!(field::CellField{T,1}, value, i, j) where {T}
    component(field, 1)[i, j] = value
    return field
end

@inline function Base.setindex!(field::CellField, value, I::CartesianIndex{3})
    field[I[1], I[2], I[3]] = value
    return field
end

"""
    allocate_cellfield(array_type, T, dims, n)

Allocate a `CellField` with `n` components, each created via
`array_type{T}(undef, dims...)`.
"""
function allocate_cellfield(array_type,
                            ::Type{T},
                            dims::NTuple{2,<:Integer},
                            n::Integer) where {T}
    n > 0 || throw(ArgumentError("Number of components must be positive"))
    dimsT = (Int(dims[1]), Int(dims[2]))
    components = ntuple(_ -> array_type{T}(undef, dimsT...), n)
    return CellField(components)
end

"""
    allocate_like(field)

Allocate a new `CellField` whose components mirror the array types and shapes of
`field`.
"""
function allocate_like(field::CellField)
    comps = ntuple(i -> similar(component(field, i)), ncomponents(field))
    return CellField(comps)
end

"""
    allocate_like(field, T)

Allocate a new `CellField` with components similar to those in `field` but whose
entries have element type `T`.
"""
function allocate_like(field::CellField, ::Type{T}) where {T}
    comps = ntuple(i -> similar(component(field, i), T, size(component(field, i))),
                   ncomponents(field))
    return CellField(comps)
end

Base.fill!(field::CellField, value::Number) = begin
    for comp in cell_components(field)
        fill!(comp, value)
    end
    return field
end

function Base.fill!(field::CellField, values::Union{Tuple,NTuple})
    length(values) == ncomponents(field) ||
        throw(ArgumentError("Tuple length must match number of components"))
    for (comp, val) in zip(cell_components(field), values)
        fill!(comp, val)
    end
    return field
end

function Base.copyto!(dest::CellField, src::CellField)
    ncomponents(dest) == ncomponents(src) ||
        throw(ArgumentError("Component count mismatch in copy"))
    for i in 1:ncomponents(dest)
        copyto!(component(dest, i), component(src, i))
    end
    return dest
end

Base.similar(field::CellField) = allocate_like(field)

Base.similar(field::CellField, ::Type{T}) where {T} = allocate_like(field, T)

"""
    map_components!(f, dest, inputs...)

Apply `f` to each component array in `dest`, passing the corresponding
components drawn from each `inputs...` collection. All inputs must either be
`CellField`s with matching component counts or scalar arrays reused for every
component.
"""
function map_components!(f, dest::CellField, inputs...)
    n = ncomponents(dest)
    nin = length(inputs)
    for i in 1:n
        args = ntuple(j -> begin
                input = inputs[j]
                input isa CellField ? component(input, i) : input
            end, nin)
        f(component(dest, i), args...)
    end
    return dest
end

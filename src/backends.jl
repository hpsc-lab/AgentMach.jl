abstract type ExecutionBackend end

struct SerialBackend <: ExecutionBackend
end

struct KernelAbstractionsBackend{D,WS} <: ExecutionBackend
    device::D
    workgroupsize::WS
end

function KernelAbstractionsBackend(device; workgroupsize = nothing)
    return KernelAbstractionsBackend{typeof(device), typeof(workgroupsize)}(device, workgroupsize)
end

KernelAbstractionsBackend(; device = :cpu, workgroupsize = nothing) =
    KernelAbstractionsBackend(device; workgroupsize = workgroupsize)

default_backend() = SerialBackend()

function describe(backend::ExecutionBackend)
    backend isa SerialBackend && return "SerialBackend"
    backend isa KernelAbstractionsBackend && return "KernelAbstractionsBackend($(backend.device))"
    return string(typeof(backend))
end

function default_array_type(::SerialBackend)
    return Array
end

function default_array_type(backend::KernelAbstractionsBackend)
    return _default_array_type_from_spec(backend.device)
end

_default_array_type_from_spec(::Any) = Array

function _default_array_type_from_spec(spec::Symbol)
    return _default_array_type_from_symbol(Val(spec))
end

_default_array_type_from_symbol(::Val) = Array

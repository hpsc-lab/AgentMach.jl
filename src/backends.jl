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

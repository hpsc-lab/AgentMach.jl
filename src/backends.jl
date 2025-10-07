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
    if backend isa SerialBackend
        return "SerialBackend"
    elseif backend isa KernelAbstractionsBackend
        return "KernelAbstractionsBackend($(backend.device))"
    else
        return string(typeof(backend))
    end
end

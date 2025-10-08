module AgentMachCUDAExt

using AgentMach
using KernelAbstractions
using CUDA

function __init__()
    AgentMach.register_backend!(:cuda, () -> KernelAbstractions.CUDADevice())
end

AgentMach._default_array_type_from_symbol(::Val{:cuda}) = CUDA.CuArray

end # module

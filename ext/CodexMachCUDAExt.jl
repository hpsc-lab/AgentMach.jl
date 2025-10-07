module CodexMachCUDAExt

using CodexMach
using KernelAbstractions
using CUDA

function __init__()
    CodexMach.register_backend!(:cuda, () -> KernelAbstractions.CUDADevice())
end

CodexMach._default_array_type_from_symbol(::Val{:cuda}) = CUDA.CuArray

end # module

module CodexMachCUDAExt

using CodexMach
using KernelAbstractions
using CUDA

function __init__()
    CodexMach.register_backend!(:cuda, () -> KernelAbstractions.CUDADevice())
end

end # module

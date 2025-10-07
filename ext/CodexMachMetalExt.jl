module CodexMachMetalExt

using CodexMach
using KernelAbstractions
using Metal

function __init__()
    CodexMach.register_backend!(:metal, () -> KernelAbstractions.MetalDevice())
end

end # module

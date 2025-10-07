module CodexMachMetalExt

using CodexMach
using KernelAbstractions
using Metal

function __init__()
    CodexMach.register_backend!(:metal, () -> Metal.MetalBackend())
end

CodexMach._default_array_type_from_symbol(::Val{:metal}) = Metal.MtlArray
CodexMach._default_array_type_from_spec(::Metal.MetalBackend) = Metal.MtlArray

end # module

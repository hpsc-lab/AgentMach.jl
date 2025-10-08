module AgentMachMetalExt

using AgentMach
using KernelAbstractions
using Metal

function __init__()
    AgentMach.register_backend!(:metal, () -> Metal.MetalBackend())
end

AgentMach._default_array_type_from_symbol(::Val{:metal}) = Metal.MtlArray
AgentMach._default_array_type_from_spec(::Metal.MetalBackend) = Metal.MtlArray

end # module

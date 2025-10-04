module CodexPar

export greet

"""
    greet(name::AbstractString="world")

Return a friendly greeting so downstream users can smoke-test that the package is correctly installed.
"""
function greet(name::AbstractString="world")
    return "Hello, $(name)! Welcome to CodexPar."
end

end # module

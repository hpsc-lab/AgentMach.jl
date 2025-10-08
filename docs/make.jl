using Documenter
using AgentMach

DocMeta.setdocmeta!(AgentMach, :DocTestSetup, :(using AgentMach); recursive=true)

makedocs(
    modules = [AgentMach],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    sitename = "AgentMach.jl",
    doctest = true,
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "getting-started.md",
            "Examples" => "examples.md",
        ],
        "Reference" => [
            "API Reference" => "api.md",
        ],
        "Project" => [
            "Authors" => "authors.md",
            "License" => "license.md",
        ],
    ],
)

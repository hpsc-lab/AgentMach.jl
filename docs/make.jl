using Documenter
using CodexPar

DocMeta.setdocmeta!(CodexPar, :DocTestSetup, :(using CodexPar); recursive=true)

makedocs(
    modules = [CodexPar],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    sitename = "CodexPar.jl",
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

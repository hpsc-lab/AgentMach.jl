using Documenter
using CodexMach

DocMeta.setdocmeta!(CodexMach, :DocTestSetup, :(using CodexMach); recursive=true)

makedocs(
    modules = [CodexMach],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    sitename = "CodexMach.jl",
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

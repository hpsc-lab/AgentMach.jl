# CodexPar.jl

CodexPar.jl is a fresh Julia package scaffold created as a starting point for parallel-computing experiments. The package currently exposes a tiny `greet` utility so that you can validate your development environment end-to-end.

## Getting started

```julia
pkg> activate .
pkg> instantiate
julia> using CodexPar
julia> greet()
"Hello, world! Welcome to CodexPar."
```

## Testing

Run the package tests with Julia's package manager:

```julia
pkg> activate .
pkg> test
```

## Documentation

The repository ships with a bare `docs/` folder ready to host Documenter.jl-based documentation if you choose to add it later. A GitHub Actions workflow is also included so CI runs tests on every push.

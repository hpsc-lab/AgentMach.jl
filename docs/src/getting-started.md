# Getting Started

```@meta
CurrentModule = CodexMach
```

Follow the steps below to set up CodexMach.jl for local development and to verify
that the package is working as intended.

## Installation

1. Install [Julia 1.11](https://julialang.org/downloads/) or newer.
2. Clone the repository and instantiate the package environment:

   ```julia
   julia> import Pkg; Pkg.activate("/path/to/CodexMach.jl"); Pkg.instantiate()
   ```

   Instantiation resolves both runtime and test dependencies so you can execute
   the examples and regression tests without extra setup.

## Smoke Test

Run the unit tests to confirm your toolchain is healthy and the finite-volume
kernels behave as expected:

```julia
julia> import Pkg; Pkg.activate("/path/to/CodexMach.jl"); Pkg.test()
```

The tests cover mesh construction, boundary-condition plumbing, and the explicit
RK2 integrator for both equation sets.

## Building the Documentation

The documentation in this folder is built with Documenter.jl. To generate the
HTML output locally, run:

```julia
julia> import Pkg
julia> Pkg.activate("/path/to/CodexMach.jl/docs")
julia> Pkg.instantiate()
julia> include("/path/to/CodexMach.jl/docs/make.jl")
```

The generated site is written to `docs/build/`. Open `index.html` in your
browser to browse the manual.

The hosted manual is published at [https://schlotm.github.io/CodexMach.jl/dev](https://schlotm.github.io/CodexMach.jl/dev).

## Next Steps

- Work through the [Examples](examples.md) to reproduce the Kelvin–Helmholtz
  validation run and the linear advection convergence study.
- Explore the [API Reference](api.md) for details on composing problems from the
  provided building blocks.

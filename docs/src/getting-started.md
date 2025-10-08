# Getting Started

```@meta
CurrentModule = AgentMach
```

Follow the steps below to set up AgentMach.jl for local development and to verify
that the package is working as intended.

## Installation

1. Install [Julia 1.11](https://julialang.org/downloads/) or newer.
2. Clone the repository and instantiate the package environment:

   ```julia
   julia> import Pkg; Pkg.activate("/path/to/AgentMach.jl"); Pkg.instantiate()
   ```

   Instantiation resolves both runtime and test dependencies so you can execute
   the examples and regression tests without extra setup.

## Smoke Test

Run the unit tests to confirm your toolchain is healthy and the finite-volume
kernels behave as expected:

```julia
julia> import Pkg; Pkg.activate("/path/to/AgentMach.jl"); Pkg.test()
```

The tests cover mesh construction, boundary-condition plumbing, and the explicit
RK2 integrator for both equation sets.

## Visual Check: Linear Advection Pulse

After the tests pass, generate a quick visual confirmation that the linear
advection driver and plotting stack work on your machine. The snippet below
transports a compact Gaussian pulse across a periodic square domain and renders
the final density field with `Plots.jl`:

```julia
julia> using AgentMach, Plots
julia> problem = setup_linear_advection_problem(128, 128; velocity = (0.5, -0.25))
julia> init(x, y) = exp(-80 * ((x - 0.25)^2 + (y - 0.5)^2))
julia> state = LinearAdvectionState(problem; init)
julia> run_linear_advection!(state, problem; steps = 200, dt = 0.0025);
julia> heatmap(AgentMach.scalar_component(solution(state));
               aspect_ratio = 1,
               color = :turbo,
               xlabel = "x",
               ylabel = "y",
               title = "Linear advection after one period")
```

Switch the state construction to `LinearAdvectionState(problem; init, backend =
KernelAbstractionsBackend(:metal))` (or `:cuda`) to exercise the GPU kernels.

## Building the Documentation

The documentation in this folder is built with Documenter.jl. To generate the
HTML output locally, run:

```julia
julia> import Pkg
julia> Pkg.activate("/path/to/AgentMach.jl/docs")
julia> Pkg.instantiate()
julia> include("/path/to/AgentMach.jl/docs/make.jl")
```

The generated site is written to `docs/build/`. Open `index.html` in your
browser to browse the manual.

The hosted manual is published at [https://hpsc-lab.github.io/AgentMach.jl/](https://hpsc-lab.github.io/AgentMach.jl/).

## Next Steps

- Work through the [Examples](examples.md) to reproduce the Kelvin–Helmholtz
  validation run and the linear advection convergence study.
- Explore the [API Reference](api.md) for details on composing problems from the
  provided building blocks.

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

## Example simulation

The `examples/linear_advection_demo.jl` script wires together mesh generation,
problem setup, RK2 time integration, and the high-level driver. Run it from the
repository root:

```bash
julia --project=. examples/linear_advection_demo.jl
```

It prints periodic RMS diagnostics along with the CFL number used for the run.
Tune parameters by editing the keyword arguments in `run_linear_advection_demo`.

To capture diagnostics, pass output paths (created if missing):

```bash
julia --project=. examples/linear_advection_demo.jl diagnostics.csv final_state.csv
```

The first file lists sampled step/time/RMS/CFL data, and the second stores the
final cell-centered field with coordinates.

### Visualise results

To render the sampled diagnostics (and optionally the final field), use the
helper script:

```bash
julia --project=. examples/plot_linear_advection.jl diagnostics.csv final_state.csv plot.png
```

Install `Plots.jl` in your environment first (`import Pkg; Pkg.add("Plots")`).
The script falls back to a readable error if the dependency is missing.

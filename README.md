# CodexMach.jl

CodexMach.jl is a fresh Julia package scaffold created as a starting point for parallel-computing experiments. The package currently exposes a tiny `greet` utility so that you can validate your development environment end-to-end.

## Getting started

```julia
pkg> activate .
pkg> instantiate
julia> using CodexMach
julia> greet()
"Hello, world! Welcome to CodexMach."
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

## Convergence study

To verify spatial accuracy, run the periodic convergence sweep:

```bash
julia --project=. examples/convergence_linear_advection.jl
```

The driver evolves a sinusoidal field across five nested grids, prints the L₂
error at each resolution, estimates the per-level experimental order of
convergence (EOC), and finishes with the average EOC. Edit
`run_convergence_study` inside the script to adjust the final time, CFL target,
or refinement levels.

## Kelvin-Helmholtz instability

The compressible Euler path is exercised by
`examples/kelvin_helmholtz_euler.jl`, which seeds a periodic Kelvin-Helmholtz
roll-up on a square box:

```bash
julia --project=. examples/kelvin_helmholtz_euler.jl
```

By default the driver uses a 256×256 mesh, RK2 time stepping with adaptive CFL
control, and prints periodic log messages. Pass a file path via
`diagnostics_path` to capture per-step CFL and kinetic-energy measurements. The
routine returns the final state so you can post-process density, vorticity, or
other derived fields. Supply `pdf_path` to snapshot the terminal density field,
and `animation_path` (MP4 or GIF) plus `animation_every`/`animation_fps` to
produce a time-resolved movie.

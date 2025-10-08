# AgentMach.jl

[![docs | dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://hpsc-lab.github.io/AgentMach.jl/)
[![CI](https://github.com/hpsc-lab/AgentMach.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/hpsc-lab/AgentMach.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/hpsc-lab/AgentMach.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/hpsc-lab/AgentMach.jl)
[![Coveralls](https://coveralls.io/repos/github/hpsc-lab/AgentMach.jl/badge.svg?branch=main)](https://coveralls.io/github/hpsc-lab/AgentMach.jl?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AgentMach.jl is a Julia playground for evaluating how Codex-style AI systems—`gpt-5-codex` in this repository—can design and grow scientific-computing software without direct human-authored code. The long-term objective is a CFD solver that scales to high-performance workloads, combining performance-minded kernels, multithreading, GPU execution, and (eventually) MPI for distributed runs. Contributions are welcome, with one key rule: all code must be produced through tools such as Copilot or Codex; manual patches are intentionally excluded so we can measure the tooling’s capabilities.

Current highlights:

- 2D structured meshes with second-order finite-volume discretisation.
- Explicit RK2 time integration with shared stage buffers to avoid per-step allocation.
- Linear scalar advection (upwind flux) and compressible Euler equations (MUSCL + Rusanov) with periodic boundaries.
- Optional volumetric source callbacks that operate on both CPU and KernelAbstractions GPU backends.
- Limiter abstraction (`MinmodLimiter`, `UnlimitedLimiter`) for MUSCL slopes, configurable per problem.
- Manufactured-solution convergence suites for advection and Euler, with per-variable error/EOC reporting.

## Getting started

```julia
pkg> activate .
pkg> instantiate
julia> using AgentMach
julia> greet()
"Hello, world! Welcome to AgentMach."
```

## Documentation

Hosted documentation lives at [https://hpsc-lab.github.io/AgentMach.jl/](https://hpsc-lab.github.io/AgentMach.jl/). It covers usage examples, API details, and convergence studies for each physics module.

## Example simulation

Run the linear advection demo on the CPU:

```bash
julia --project=run examples/linear_advection_demo.jl
```

Run the same demo on a Metal-capable GPU:

```bash
julia --project=run -e 'using Metal; include("examples/linear_advection_demo.jl"); run_linear_advection_demo(backend=:metal)'
```

The script wires together mesh generation, problem setup, RK2 time integration,
and the high-level driver. It prints periodic RMS diagnostics along with the CFL
number used for the run. Override the precision explicitly with
`state_eltype=Float64` if you want to run the GPU case in double precision (CUDA)
or stay on `Float32` for Metal.

To capture diagnostics, pass output paths (created if missing):

```bash
julia --project=run examples/linear_advection_demo.jl diagnostics.csv final_state.csv
```

The first file lists sampled step/time/RMS/CFL data, and the second stores the
final cell-centered field with coordinates.

### Visualise results

To render the sampled diagnostics (and optionally the final field), use the
helper script:

```bash
julia --project=run examples/plot_linear_advection.jl diagnostics.csv final_state.csv plot.png
```

Install `Plots.jl` in your environment first (`import Pkg; Pkg.add("Plots")`).
The script falls back to a readable error if the dependency is missing.

## Convergence study

CPU run:

```bash
julia --project=run examples/convergence_linear_advection.jl
```

Metal GPU run:

```bash
julia --project=run -e 'using Metal; include("examples/convergence_linear_advection.jl"); run_convergence_study(backend=:metal, levels=4)'
```

The driver evolves a sinusoidal field across five nested grids, prints the L₂
error at each resolution, estimates the per-level experimental order of
convergence (EOC), and finishes with the average EOC. Edit
`run_convergence_study` inside the script to adjust the final time, CFL target,
or refinement levels.

## Compressible Euler convergence

CPU run:

```bash
julia --project=run examples/convergence_compressible_euler.jl
```

Metal GPU run (requires `using Metal` and a registered backend):

```bash
julia --project=run -e 'using Metal; include("examples/convergence_compressible_euler.jl");
                         run_euler_convergence_study(backend=:metal, limiter=unlimited_limiter)'
```

The script drives a manufactured solution over a hierarchy of meshes and
reports L₂ errors for each conserved component `(ρ, ρu, ρv, E)` plus their
individual EOCs. Pass `limiter=unlimited_limiter` to disable slope limiting and
recover the nominal second-order MUSCL accuracy on smooth problems.

## Kelvin-Helmholtz instability

CPU run:

```bash
julia --project=run examples/kelvin_helmholtz_euler.jl
```

Metal GPU run:

```bash
julia --project=run -e 'using Metal; include("examples/kelvin_helmholtz_euler.jl"); run_kelvin_helmholtz(backend=:metal, final_time=1.0)'
```

By default the driver uses a 256×256 mesh, RK2 time stepping with adaptive CFL
control, and prints periodic log messages. Set `backend=:metal` (or `:cuda`)
to run the scenario on a GPU; the helper automatically falls back to `Float32`
storage for KernelAbstractions backends while keeping `Float64` for the serial
path. Pass a file path via `diagnostics_path` to capture per-step CFL and
kinetic-energy measurements. The routine returns the final state so you can
post-process density, vorticity, or other derived fields. Supply `pdf_path` to
snapshot the terminal density field, and `animation_path` (MP4 or GIF) plus
`animation_every`/`animation_fps` to produce a time-resolved movie.

## Backend profiling

Compare the serial and GPU paths with the profiling helper:

```bash
julia --project=run examples/profile_backends.jl
```

It benchmarks the RK2 loops for linear advection and compressible Euler across
the serial and any registered KernelAbstractions backends (CPU, Metal, CUDA),
forcing `Float32` to keep Metal hardware supported.

## Development challenges

- Adding packages with incorrect UUIDs cropped up often; fixing the registry mismatches slowed down iteration and required manual cleanup of `Project.toml`/`Manifest.toml`.
- macOS sandboxing kept blocking command execution, so routine actions like `Pkg.instantiate` or simple shell checks needed elevated approvals or alternative workflows.
- Extending or refactoring features routinely broke untested paths (especially mixed CPU/GPU modes), revealing how thin our regression suite still is.
- The implementation leans verbose so that backend-specific branches stay explicit, but that also makes large refactors risky and time-consuming.

## Maintainer

AgentMach.jl is maintained by Michael Schlottke-Lakemper
<michael.schlottke-lakemper@uni-a.de>.

## Provenance

This repository was created end-to-end using gpt-5-codex.

# CodexPar.jl

CodexPar.jl is a playground for building high-performance computational fluid
dynamics (CFD) solvers in Julia. The package assembles structured meshes,
configurable physics kernels, and explicit time integration building blocks into
a composable toolkit that targets both CPU and GPU execution and scales to
distributed-memory systems.

## Highlights

- Second-order finite-volume discretisations on 2D structured meshes with
  periodic or Dirichlet boundary conditions.
- Multiple equation sets: linear scalar advection for algorithm development and
  the compressible Euler equations as a stepping stone toward Navier–Stokes.
- Explicit two-stage Runge–Kutta (Heun) time integration with CFL-aware step
  control helpers.
- Portable kernels designed to run on Julia arrays today and on accelerators via
  KernelAbstractions.jl in forthcoming releases.
- Simulation drivers with callback hooks, diagnostics, and CSV/visualisation
  utilities that streamline benchmarking and validation.

## Architecture Overview

CodexPar keeps mesh generation, physics descriptions, flux evaluations, and time
integration decoupled so the components can be recombined when experimenting
with new physics or backends. The top-level module simply re-exports the most
useful types and functions; you can construct meshes and PDE problems directly
and pass them to the provided integrators or embed them within your own driver
code.

For a quick tour of the API, see the [Getting Started](getting-started.md)
guide. Ready-to-run scripts that reproduce the figures in the repository live in
[`examples/`](examples.md); the scripts report diagnostics so you can verify
behaviour as you experiment with mesh resolution, CFL targets, or higher-order
fluxes.

## Roadmap

The current release focuses on CPU execution and correctness-tested kernels.
Upcoming work centres on:

- lifting the existing kernels onto KernelAbstractions for GPU execution;
- wiring MPI-aware distributed arrays into the same API; and
- extending the physics catalogue toward viscous Navier–Stokes flows backed by
  validation test cases.

If you would like to contribute or report an issue, please open a discussion on
GitHub. The [Authors](authors.md) page lists the current maintainers.

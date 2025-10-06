# CodexMach Agent Brief

## Mission
CodexMach.jl is a Julia package for experimenting with high-performance CFD. Every change should move us toward a production-quality solver that can run on both CPUs and GPUs and scale across distributed memory systems.

## Core Requirements
- 2D structured meshes with 2nd-order finite volume discretizations.
- Configurable boundary conditions: periodic and Dirichlet.
- Support multiple equation sets: linear scalar advection, compressible Euler, and Navier–Stokes.
- Explicit 2nd-order Runge-Kutta time integration (no higher-order schemes yet).
- Parallel execution with MPI for distributed-memory scaling.
- GPU execution paths implemented with KernelAbstractions.jl.

## Development Priorities
1. Keep the API composable: mesh generation, physics kernels, flux evaluations, and time integration should remain decoupled.
2. Preserve correctness with regression tests; add problem-specific benchmarks whenever performance changes.
3. Write portable kernels: ensure each computational hot spot works with CPU, GPU, and MPI backends.
4. Favor data layouts and algorithms that minimize memory traffic and maximize parallel efficiency.

## Collaboration Notes
- Document assumptions and derivations in `docs/` so downstream users can reproduce results.
- When adding or modifying physics, include references and validation cases in `test/`.
- Respect existing coding conventions (Julia style guide, ASCII text) and avoid reverting user changes.
- Use `rg` for searches, keep commits atomic, and describe system-specific setup hurdles in issues or docs.

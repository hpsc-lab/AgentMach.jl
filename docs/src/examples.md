# Examples

```@meta
CurrentModule = CodexPar
```

The `examples/` directory contains reproducible scripts that exercise the core
numerics. You can run each script directly with Julia once the project
environment has been instantiated (`julia --project=.`).

## Linear Advection Demo

```
julia --project=. examples/linear_advection_demo.jl
```

This script transports a smooth sine blob around a rectangular periodic domain
until it returns to its starting position. By default it:

- refines the mesh to maintain square cells (`Δx = Δy`);
- picks a timestep that closes an integer number of domain traversals; and
- records RMS amplitude samples plus the CFL number during the run.

The function returns a named tuple with the step size, final time, and diagnostic
history. When you pass `diagnostics_path="diagnostics.csv"` it writes a CSV
mirroring the on-screen log, and `state_path="state.csv"` captures the final
solution for later visualisation.

To plot the diagnostics or the final field, call

```
julia --project=. examples/plot_linear_advection.jl diagnostics.csv state.csv
```

which generates a PDF overlaying the analytic and numerical solutions.

## Linear Advection Convergence Study

```
julia --project=. examples/convergence_linear_advection.jl
```

The convergence driver repeats the sinusoidal advection problem on a hierarchy
of meshes. It prints the L² error for each resolution along with the observed
order of accuracy. You should expect second-order convergence (EOC ≈ 2.0) once
the mesh is sufficiently fine.

## Kelvin–Helmholtz Instability (Compressible Euler)

```
julia --project=. examples/kelvin_helmholtz_euler.jl
```

This example launches a Kelvin–Helmholtz roll-up on a periodic square domain
using the compressible Euler equations and the slope-limited Rusanov fluxes. The
script logs CFL numbers, tracks the volume-averaged kinetic energy, and can
produce publication-ready figures:

- set `diagnostics_path="diagnostics.csv"` to capture the per-step log;
- set `pdf_path="khi_density.pdf"` for a static density snapshot at `t = final_time`;
- provide `animation_path="khi.mp4"` to save a movie (requires ffmpeg support in
  Plots.jl).

A stable run with the default settings reaches `t = 1.5` in roughly a thousand
steps with CFL ≈ 0.4–0.45. The density field develops rolled vortices that match
the reference figures checked into the repository (`khi_*.pdf/mp4`).

## Tips

- All drivers expose keyword arguments for mesh size, timestep control, and file
  output locations so you can run parameter studies without editing source code.
- Long-running cases benefit from enabling the logging hooks (`log_every`) to
  keep an eye on CFL drift or diagnostic regressions.

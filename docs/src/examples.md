# Examples

```@meta
CurrentModule = CodexMach
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
solution for later visualisation. By default the demo runs on the serial backend
with `Float64` storage, but you can switch to a KernelAbstractions backend while
automatically selecting a GPU-friendly element type:

```
julia --project=. -e 'include("examples/linear_advection_demo.jl"); run_linear_advection_demo(backend=:metal)'
```

Override the precision explicitly with `state_eltype=Float64` (for CUDA) or
`state_eltype=Float32` if you want to minimise device memory traffic.

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
the mesh is sufficiently fine. To assess GPU correctness, pass a backend symbol
when calling the driver:

```
julia --project=. -e 'include("examples/convergence_linear_advection.jl"); run_convergence_study(backend=:metal, levels=4)'
```

The serial path remains the default, so simple invocations continue to run on
the CPU without any extra keywords.

## Kelvin–Helmholtz Instability (Compressible Euler)

```
julia --project=. examples/kelvin_helmholtz_euler.jl
```

This example launches a Kelvin–Helmholtz roll-up on a periodic square domain
using the compressible Euler equations and the slope-limited Rusanov fluxes. The
script logs CFL numbers, tracks the volume-averaged kinetic energy, and can
produce publication-ready figures. To exercise GPU execution, pass
`backend=:metal` (or `:cuda` if available); the helper will automatically fall
back to `Float32` storage on GPU backends while keeping `Float64` on the serial
path:

```
julia --project=. -e 'include("examples/kelvin_helmholtz_euler.jl"); run_kelvin_helmholtz(backend=:metal, final_time=1.0)'
```

Further output customisation includes:

- set `diagnostics_path="diagnostics.csv"` to capture the per-step log;
- set `pdf_path="khi_density.pdf"` for a static density snapshot at `t = final_time`;
- provide `animation_path="khi.mp4"` to save a movie (requires ffmpeg support in
  Plots.jl).

A stable run with the default settings reaches `t = 1.5` in roughly a thousand
steps with CFL ≈ 0.4–0.45. The density field develops rolled vortices that match
the reference figures checked into the repository (`khi_*.pdf/mp4`).

## Backend Profiling

```
julia --project=. examples/profile_backends.jl
```

This utility benchmarks the serial driver against available KernelAbstractions
backends for both linear advection and compressible Euler. It uses `Float32`
storage so Metal-backed GPUs remain supported and prints per-backend wall-clock
timings for quick regressions after kernel changes.

## Tips

- All drivers expose keyword arguments for mesh size, timestep control, and file
  output locations so you can run parameter studies without editing source code.
- Long-running cases benefit from enabling the logging hooks (`log_every`) to
  keep an eye on CFL drift or diagnostic regressions.

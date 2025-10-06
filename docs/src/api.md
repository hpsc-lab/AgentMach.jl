# API Reference

```@meta
CurrentModule = CodexPar
```

## Utilities

```@docs
greet
```

## Mesh Construction

```@docs
StructuredMesh
cell_centers
spacing
origin
```

## Boundary Conditions

```@docs
PeriodicBoundaryConditions
is_periodic
periodic_axes
```

## Linear Advection

```@docs
LinearAdvection
LinearAdvectionProblem
velocity
mesh
boundary_conditions
pde
setup_linear_advection_problem
LinearAdvectionState
```

## Compressible Euler

```@docs
CompressibleEuler
CompressibleEulerProblem
setup_compressible_euler_problem
CompressibleEulerState
primitive_variables
```

## Time Integration

```@docs
RK2Workspace
solution
workspace
compute_rhs!
rk2_step!
stable_timestep
cfl_number
```

## Simulation Drivers

```@docs
run_linear_advection!
run_compressible_euler!
```

# API Reference

```@meta
CurrentModule = AgentMach
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
source(::AgentMach.LinearAdvectionProblem)
setup_linear_advection_problem
LinearAdvectionState
```

## Compressible Euler

```@docs
CompressibleEuler
CompressibleEulerProblem
source(::AgentMach.CompressibleEulerProblem)
limiter(::AgentMach.CompressibleEulerProblem)
setup_compressible_euler_problem
CompressibleEulerState
primitive_variables
_primitive_variables_cpu
```

## Limiters

```@docs
AbstractLimiter
MinmodLimiter
UnlimitedLimiter
minmod_limiter
unlimited_limiter
apply_limiter
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

## Cell Fields

```@docs
CellField
allocate_cellfield
allocate_like
map_components!
cell_components
component
ncomponents
spatial_size
backend
```

## Simulation Drivers

```@docs
run_linear_advection!
run_compressible_euler!
```

#!/usr/bin/env julia
using Plots

"""
    plot_linear_advection_csv(diagnostics_path; state_path=nothing,
                               output_path="linear_advection.png",
                               velocity=(1.0, 0.0))

Load diagnostics (and optionally final-state) CSV files produced by
`linear_advection_demo.jl` and generate a PNG summary. When a state file is
provided the plot includes a 2-D heatmap of the final field and a quantitative
comparison against the exact solution along the domain diagonal, assuming the
specified constant advection velocity.
"""
function plot_linear_advection_csv(diagnostics_path::AbstractString;
                                   state_path::Union{Nothing,AbstractString} = nothing,
                                   output_path::AbstractString = "linear_advection.png",
                                   velocity::Tuple{<:Real,<:Real} = (1.0, 0.0))
    diagnostics = _read_diagnostics_csv(diagnostics_path)
    state_records = state_path === nothing ? nothing : _read_state_csv(state_path)

    fig = _build_plot(diagnostics, state_records; velocity = velocity)
    savefig(fig, output_path)

    return output_path
end

function _read_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("CSV file is empty: $(path)"))
    header = String.(split(strip(first(lines)), ','))
    data = [String.(split(strip(line), ',')) for line in Iterators.drop(lines, 1) if !isempty(strip(line))]
    return header, data
end

function _column_indices(header::Vector{String}, names::Vector{String})
    lookup = Dict(name => idx for (idx, name) in enumerate(header))
    return map(name -> get(lookup, name, nothing), names)
end

function _read_diagnostics_csv(path::AbstractString)
    header, rows = _read_csv(path)
    want = ["step", "time", "rms", "cfl"]
    indices = _column_indices(header, want)
    for (name, idx) in zip(want, indices)
        idx === nothing && throw(ArgumentError("Diagnostics CSV missing `$name` column"))
    end

    parsecol(idx) = [parse(Float64, row[idx]) for row in rows]

    return (
        step = parsecol(indices[1]),
        time = parsecol(indices[2]),
        rms = parsecol(indices[3]),
        cfl = parsecol(indices[4]),
    )
end

function _read_state_csv(path::AbstractString)
    header, rows = _read_csv(path)
    want = ["i", "j", "x", "y", "u"]
    indices = _column_indices(header, want)
    for (name, idx) in zip(want, indices)
        idx === nothing && throw(ArgumentError("State CSV missing `$name` column"))
    end

    i_vals = [parse(Int, row[indices[1]]) for row in rows]
    j_vals = [parse(Int, row[indices[2]]) for row in rows]
    x_vals = [parse(Float64, row[indices[3]]) for row in rows]
    y_vals = [parse(Float64, row[indices[4]]) for row in rows]
    u_vals = [parse(Float64, row[indices[5]]) for row in rows]

    nx = maximum(i_vals)
    ny = maximum(j_vals)

    field = zeros(Float64, nx, ny)
    xs = similar(field)
    ys = similar(field)

    for (ii, jj, xx, yy, uu) in zip(i_vals, j_vals, x_vals, y_vals, u_vals)
        field[ii, jj] = uu
        xs[ii, jj] = xx
        ys[ii, jj] = yy
    end

    return (; x = xs,
            y = ys,
            u = field)
end

function _build_plot(diagnostics, state_records; velocity)
    diagnostics_plot = plot(diagnostics.time, diagnostics.rms;
                            xlabel = "Time",
                            ylabel = "RMS",
                            label = "RMS",
                            title = "Linear Advection Diagnostics")

    if state_records === nothing
        return diagnostics_plot
    end

    data = state_records
    nx, ny = size(data.u)

    if nx == 1 || ny == 1
        line_plot = plot(vec(data.x), vec(data.u);
                         xlabel = "x",
                         ylabel = "u",
                         label = "Final state",
                         title = "Final field (1-D)")
        return plot(diagnostics_plot, line_plot; layout = (1, 2))
    end

    heat = _build_heatmap(data)
    diagonal = _build_diagonal_comparison(data, diagnostics; velocity = velocity)
    return plot(diagnostics_plot, heat, diagonal; layout = (1, 3))
end

function _build_heatmap(data)
    centers_x = vec(data.x[:, 1])
    centers_y = vec(data.y[1, :])
    nx = length(centers_x)
    ny = length(centers_y)
    dx = nx > 1 ? centers_x[2] - centers_x[1] : 1.0
    dy = ny > 1 ? centers_y[2] - centers_y[1] : 1.0
    lx = max(dx * nx, eps())
    ly = max(dy * ny, eps())
    aspect = ly / lx

    return heatmap(centers_x, centers_y, data.u;
                   xlabel = "x",
                   ylabel = "y",
                   title = "Final field",
                   colorbar = true,
                   aspect_ratio = aspect)
end

function _build_diagonal_comparison(data, diagnostics; velocity)
    nx, ny = size(data.u)
    n_diag = min(nx, ny)
    idxs = collect(1:n_diag)
    x_diag = [data.x[i, i] for i in idxs]
    y_diag = [data.y[i, i] for i in idxs]
    numerical = [data.u[i, i] for i in idxs]

    centers_x = vec(data.x[:, 1])
    centers_y = vec(data.y[1, :])
    dx = centers_x[2] - centers_x[1]
    dy = centers_y[2] - centers_y[1]
    lx = dx * nx
    ly = dy * ny
    origin_x = centers_x[1] - dx / 2
    origin_y = centers_y[1] - dy / 2

    final_time = isempty(diagnostics.time) ? 0.0 : diagnostics.time[end]
    init_fun = _two_wave_initializer((lx, ly))
    vx, vy = velocity

    exact = similar(numerical)
    for k in eachindex(idxs)
        x_adv = _wrap_coordinate(x_diag[k] - vx * final_time, origin_x, lx)
        y_adv = _wrap_coordinate(y_diag[k] - vy * final_time, origin_y, ly)
        exact[k] = init_fun(x_adv, y_adv)
    end

    plt = plot(x_diag, numerical;
               xlabel = "x along diagonal",
               ylabel = "u",
               label = "Numerical",
               title = "Diagonal comparison",
               legend = :bottomleft)
    plot!(plt, x_diag, exact;
          label = "Exact")
    return plt
end

function _two_wave_initializer(lengths::NTuple{2,<:Real})
    Lx, Ly = float.(lengths)
    function init(x, y)
        term1 = sin(2pi * x / Lx) * cos(pi * y / Ly)
        term2 = 0.5 * cos(4pi * x / Lx) * sin(2pi * y / Ly)
        return term1 + term2
    end
    return init
end

_wrap_coordinate(val, origin, length) = origin + mod(val - origin, length)

function plot_main()
    nargs = length(ARGS)
    if nargs < 1
        println(stderr, "usage: julia plot_linear_advection.jl diagnostics.csv [state.csv] [output.png] [--velocity vx vy]")
        return 1
    end

    try
        idx = 1
        diagnostics_path = ARGS[idx]; idx += 1

        state_path = nothing
        if idx <= nargs && !startswith(ARGS[idx], "--")
            state_path = ARGS[idx]
            idx += 1
        end

        output_path = "linear_advection.png"
        if idx <= nargs && !startswith(ARGS[idx], "--")
            output_path = ARGS[idx]
            idx += 1
        end

        velocity = (1.0, 0.0)
        while idx <= nargs
            arg = ARGS[idx]
            if arg == "--velocity"
                idx + 2 <= nargs || throw(ArgumentError("--velocity expects two numeric arguments"))
                vx = parse(Float64, ARGS[idx + 1])
                vy = parse(Float64, ARGS[idx + 2])
                velocity = (vx, vy)
                idx += 3
            else
                throw(ArgumentError("Unknown argument: $(arg)"))
            end
        end

        plot_linear_advection_csv(diagnostics_path;
                                  state_path = state_path,
                                  output_path = output_path,
                                  velocity = velocity)
        println("Saved plot to $(output_path)")
        return 0
    catch err
        showerror(stderr, err)
        println(stderr)
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(plot_main())
end

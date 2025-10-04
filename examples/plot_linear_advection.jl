#!/usr/bin/env julia
using Plots

"""
    plot_linear_advection_csv(diagnostics_path; state_path,
                               heatmap_path="linear_advection_fields.pdf",
                               profiles_path="linear_advection_profiles.pdf")

Load diagnostics and state CSV files produced by `linear_advection_demo.jl` and
emit two PDF figures: one with side-by-side heatmaps of the analytical initial
condition and the simulated final state, and another comparing initial vs final
profiles along the domain diagonal and along the mid-plane in `y`.
"""
function plot_linear_advection_csv(diagnostics_path::AbstractString;
                                   state_path::Union{Nothing,AbstractString} = nothing,
                                   heatmap_path::AbstractString = "linear_advection_fields.pdf",
                                   profiles_path::AbstractString = "linear_advection_profiles.pdf")
    diagnostics = _read_diagnostics_csv(diagnostics_path)
    state_path === nothing &&
        throw(ArgumentError("State CSV is required to build heatmap and profile plots"))
    state_records = _read_state_csv(state_path)

    init_field = _build_initial_field(state_records)

    heatmap_fig = _build_heatmap_figure(state_records, init_field)
    savefig(heatmap_fig, heatmap_path)

    profiles_fig = _build_profiles_figure(state_records, init_field)
    savefig(profiles_fig, profiles_path)

    return (; diagnostics = diagnostics,
            heatmap = heatmap_path,
            profiles = profiles_path)
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

function _build_initial_field(data)
    nx, ny = size(data.u)
    init = similar(data.u)
    centers_x = data.x
    centers_y = data.y

    nx >= 1 || ny >= 1 || throw(ArgumentError("State field is empty"))

    centers_x_vec = vec(centers_x[:, 1])
    centers_y_vec = vec(centers_y[1, :])
    dx = nx > 1 ? centers_x_vec[2] - centers_x_vec[1] : centers_y_vec[2] - centers_y_vec[1]
    dy = ny > 1 ? centers_y_vec[2] - centers_y_vec[1] : dx
    Lx = dx * nx
    Ly = dy * ny

    init_fun = _sine_blob_initializer((Lx, Ly))

    @inbounds for j in 1:ny, i in 1:nx
        init[i, j] = init_fun(centers_x[i, j], centers_y[i, j])
    end

    return init
end

function _build_heatmap_figure(data, init_field)
    centers_x = vec(data.x[:, 1])
    centers_y = vec(data.y[1, :])
    nx = length(centers_x)
    ny = length(centers_y)
    nx > 1 || ny > 1 || throw(ArgumentError("Final field is degenerate; need at least a 2-D state for heatmap"))
    dx = nx > 1 ? centers_x[2] - centers_x[1] : centers_y[2] - centers_y[1]
    dy = ny > 1 ? centers_y[2] - centers_y[1] : dx
    lx = max(dx * nx, eps())
    ly = max(dy * ny, eps())
    aspect = ly / lx

    init_plot = heatmap(centers_x, centers_y, init_field;
                        xlabel = "x",
                        ylabel = "y",
                        title = "Initial field",
                        colorbar = true,
                        aspect_ratio = aspect)
    final_plot = heatmap(centers_x, centers_y, data.u;
                         xlabel = "x",
                         ylabel = "y",
                         title = "Final field",
                         colorbar = true,
                         aspect_ratio = aspect)

    return plot(init_plot, final_plot; layout = (1, 2), size = (900, 400))
end

function _build_profiles_figure(data, init_field)
    nx, ny = size(data.u)
    n_diag = min(nx, ny)
    idxs = collect(1:n_diag)
    x_diag = [data.x[i, i] for i in idxs]
    numerical = [data.u[i, i] for i in idxs]
    initial_diag = [init_field[i, i] for i in idxs]

    centers_x = vec(data.x[:, 1])
    centers_y = vec(data.y[1, :])
    nx >= 2 || ny >= 2 || throw(ArgumentError("Need at least a 2x2 mesh for profile extraction"))
    dx = nx >= 2 ? centers_x[2] - centers_x[1] : centers_y[2] - centers_y[1]
    dy = ny >= 2 ? centers_y[2] - centers_y[1] : dx
    lx = dx * nx
    ly = dy * ny
    origin_y = centers_y[1] - dy / 2

    diagonal_plot = plot(x_diag, initial_diag;
                         xlabel = "x along diagonal",
                         ylabel = "u",
                         label = "Initial",
                         title = "Diagonal profile",
                         legend = :bottomleft)
    plot!(diagonal_plot, x_diag, numerical;
          label = "Final")

    y_mid = origin_y + ly / 2
    j_center = argmin(abs.(centers_y .- y_mid))
    x_line = data.x[:, j_center]
    final_line = data.u[:, j_center]
    initial_line = init_field[:, j_center]

    center_plot = plot(x_line, initial_line;
                       xlabel = "x at y = $(round(centers_y[j_center]; digits = 4))",
                       ylabel = "u",
                       label = "Initial",
                       title = "Mid-plane profile",
                       legend = :bottomleft)
    plot!(center_plot, x_line, final_line;
          label = "Final")

    return plot(diagonal_plot, center_plot; layout = (1, 2), size = (900, 400))
end

function _sine_blob_initializer(lengths::NTuple{2,<:Real})
    Lx, Ly = float.(lengths)
    kx = 4
    ky = 2
    function init(x, y)
        return sin(2pi * kx * x / Lx) * sin(2pi * ky * y / Ly)
    end
    return init
end

function plot_main()
    nargs = length(ARGS)
    if nargs < 2
        println(stderr, "usage: julia plot_linear_advection.jl diagnostics.csv state.csv [heatmap.pdf] [profiles.pdf]")
        return 1
    end

    try
        diagnostics_path = ARGS[1]
        state_path = ARGS[2]

        heatmap_path = nargs >= 3 ? ARGS[3] : "linear_advection_fields.pdf"
        profiles_path = nargs >= 4 ? ARGS[4] : "linear_advection_profiles.pdf"

        nargs <= 4 || throw(ArgumentError("Too many positional arguments provided"))

        plot_linear_advection_csv(diagnostics_path;
                                  state_path = state_path,
                                  heatmap_path = heatmap_path,
                                  profiles_path = profiles_path)
        println("Saved heatmap figure to $(heatmap_path)")
        println("Saved profile figure to $(profiles_path)")
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

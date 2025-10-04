#!/usr/bin/env julia
"""
    plot_linear_advection_csv(diagnostics_path; state_path=nothing,
                               output_path="linear_advection.png")

Load diagnostics (and optionally final-state) CSV files produced by
`linear_advection_demo.jl` and generate a PNG summary. Requires Plots.jl to be
installed in the active environment.
"""
function plot_linear_advection_csv(diagnostics_path::AbstractString;
                                   state_path::Union{Nothing,AbstractString} = nothing,
                                   output_path::AbstractString = "linear_advection.png")
    diagnostics = _read_diagnostics_csv(diagnostics_path)
    state_records = state_path === nothing ? nothing : _read_state_csv(state_path)

    plots_mod = _load_plots()
    fig = _build_plot(plots_mod, diagnostics, state_records)
    plots_mod.savefig(fig, output_path)

    return output_path
end

function _load_plots()
    try
        @eval using Plots
    catch err
        throw(ArgumentError("Plots.jl is required to generate figures. Run `import Pkg; Pkg.add(\"Plots\")` and retry."))
    end
    return Plots
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

function _build_plot(plots_mod, diagnostics, state_records)
    fig = plots_mod.plot(diagnostics.time, diagnostics.rms;
                         xlabel = "Time",
                         ylabel = "RMS",
                         label = "RMS",
                         title = "Linear Advection Diagnostics")

    if state_records !== nothing
        data = state_records
        ny = size(data.u, 2)
        if ny == 1
            plots_mod.plot!(fig, vec(data.x), vec(data.u);
                             label = "Final state",
                             xlabel = "x",
                             ylabel = "u")
        else
            heat = plots_mod.heatmap(vec(data.x[:, 1]), vec(data.y[1, :]), data.u;
                                     xlabel = "x",
                                     ylabel = "y",
                                     title = "Final state",
                                     colorbar = true)
            fig = plots_mod.plot(fig, heat; layout = (1, 2))
        end
    end

    return fig
end

function plot_main()
    nargs = length(ARGS)
    if nargs < 1
        println(stderr, "usage: julia plot_linear_advection.jl diagnostics.csv [state.csv] [output.png]")
        return 1
    end

    diagnostics_path = ARGS[1]
    state_path = nargs >= 2 ? ARGS[2] : nothing
    output_path = nargs >= 3 ? ARGS[3] : "linear_advection.png"

    try
        plot_linear_advection_csv(diagnostics_path; state_path = state_path, output_path = output_path)
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

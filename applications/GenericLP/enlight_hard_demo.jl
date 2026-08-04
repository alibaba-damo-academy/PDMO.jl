# include(joinpath(@__DIR__, "../../warmup.jl"))
import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function setup_and_activate_project_env()
    Pkg.activate(REPO_ROOT)
    # Warmup-style dependency setup for fresh environments.
    Pkg.instantiate()
end

setup_and_activate_project_env()

include(joinpath(@__DIR__, "GenericLP.jl"))
include(joinpath(@__DIR__, "inspect_cocluster.jl"))

using PDMO

using LinearAlgebra
using SparseArrays
using Random
using PyPlot
using Printf
using JuMP 
import MathOptInterface 
using Ipopt
using HiGHS

Random.seed!(126)

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "enlight_hard_plots")

function usage()
    println("""
Usage:
    julia applications/GenericLP/enlight_hard_demo.jl <mps_path> [output_dir]

Defaults:
    output_dir = applications/GenericLP/enlight_hard_plots
""")
end

function _plot_residuals(results::AbstractDict, field::Symbol, title_text::String, outfile::String; k::Int=1)
    labels = Dict(
        "Basic" => "Basic",
        "MILP" => "MILP",
        "BFS" => "BFS",
    )
    linestyles = Dict(
        "Basic" => :solid,
        "MILP" => :dash,
        "BFS" => :dot,
    )
    colors = Dict(
        "Basic" => :blue,
        "MILP" => :green,
        "BFS" => :orange,
    )

    fig, ax = subplots(figsize=(6.4, 4.0))
    all_vals = Float64[]
    for key in ["Basic", "MILP", "BFS"]
        res = get(results, key, nothing)
        res === nothing && continue
        series = getfield(res.iterationInfo, field)
        isempty(series) && continue
        stride = max(1, k)
        iters = collect(1:stride:length(series))
        vals = series[iters]
        if field == :dresL2 && !isempty(vals)
            # The first stored dres entry corresponds to iter 0 and may be Inf.
            iters = iters[2:end]
            vals = vals[2:end]
        end
        isempty(vals) && continue
        yvals = [v > 0 ? log10(v) : NaN for v in vals]
        append!(all_vals, yvals)
        style = linestyles[key] == :solid ? "-" : linestyles[key] == :dash ? "--" : ":"
        ax.plot(iters, yvals;
            label=labels[key],
            linestyle=style,
            color=string(colors[key]),
            linewidth=2.5)
    end

    ax.set_xlabel("Iteration")
    if field == :presL2
        ax.set_ylabel("log10(||Pres||_2)")
    else
        ax.set_ylabel("log10(||Dres||_2)")
    end
    finite_vals = filter(isfinite, all_vals)
    if !isempty(finite_vals)
        ymin = minimum(finite_vals)
        ymax = maximum(finite_vals)
        if ymin == ymax
            # Avoid invalid/singular axis ranges when residuals are constant.
            pad = max(abs(ymin) * 0.05, 1e-6)
            ax.set_ylim(ymin - pad, ymax + pad)
        else
            ax.set_ylim(ymin, ymax)
        end
    end
    ax.grid(true)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.03)
    close(fig)
    println("[info] saved plot -> $(outfile)")
end

function _bipartite_positions(left_nodes::Vector{String}, right_nodes::Vector{String})
    pos = Dict{String, Tuple{Float64, Float64}}()
    nL = length(left_nodes)
    nR = length(right_nodes)
    x_left = 0.0
    x_right = 0.56
    y_top = 0.86
    y_bottom = 0.14
    for (idx, node) in enumerate(left_nodes)
        y = nL == 1 ? 0.5 : y_top - (idx - 1) * (y_top - y_bottom) / (nL - 1)
        pos[node] = (x_left, y)
    end
    for (idx, node) in enumerate(right_nodes)
        y = nR == 1 ? 0.5 : y_top - (idx - 1) * (y_top - y_bottom) / (nR - 1)
        pos[node] = (x_right, y)
    end
    return pos
end

function _trim_node_label(node_id::String)
    if startswith(node_id, "VariableNode(")
        return replace(node_id, "VariableNode(" => "", ")" => "")
    elseif startswith(node_id, "ConstraintNode(")
        cid = replace(node_id, "ConstraintNode(" => "", ")" => "")
        return "C$(cid)"
    elseif startswith(node_id, "ADMMNodeConvertedFromEdge(")
        return "aux"
    else
        return node_id
    end
end

function _latex_symbol_label(label::String)
    if startswith(label, "x_")
        idx = replace(label, "x_" => "")
        return "\$x_{$(idx)}\$"
    elseif startswith(label, "x")
        m = match(r"^x(\d+)$", label)
        if m !== nothing
            return "\$x_{$(m.captures[1])}\$"
        end
    end

    if startswith(label, "C_")
        idx = replace(label, "C_" => "")
        return "\$C_{$(idx)}\$"
    elseif startswith(label, "C")
        m = match(r"^C(\d+)$", label)
        if m !== nothing
            return "\$C_{$(m.captures[1])}\$"
        end
    end
    return label
end

function _node_color(node_id::String)
    if startswith(node_id, "VariableNode(")
        return "#8ecae6" # variable block
    elseif startswith(node_id, "ConstraintNode(")
        return "#90ee90" # constraint node
    elseif startswith(node_id, "ADMMNodeConvertedFromEdge(")
        return "#f4a261" # split/auxiliary node
    else
        return "#d3d3d3"
    end
end

function _extract_variable_index(node_name::AbstractString)
    m = match(r"(\d+)$", node_name)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function _resolve_node_name(name::String, available::Set{String})
    name in available && return name
    alt = replace(name, "C_" => "C")
    alt in available && return alt
    alt = replace(name, "C" => "C_")
    alt in available && return alt
    alt = replace(name, "x_" => "x")
    alt in available && return alt
    if startswith(name, "x") && !startswith(name, "x_")
        alt = replace(name, r"^x" => "x_")
        alt in available && return alt
    end
    return nothing
end

function _constraint_label_from_edge_id(edge_id::String)
    if startswith(edge_id, "TwoBlockEdge(")
        inner = replace(edge_id, "TwoBlockEdge(" => "", ")" => "")
        return "C$(strip(inner))"
    elseif startswith(edge_id, "MultiblockEdge(")
        inner = replace(edge_id, "MultiblockEdge(" => "", ")" => "")
        parts = split(inner, ",")
        if length(parts) >= 2
            cidx = strip(parts[1])
            var = strip(parts[2])
            vidx = _extract_variable_index(var)
            return vidx === nothing ? "C$(cidx)" : "C$(cidx)$(vidx)"
        end
    end
    return "aux"
end

function _edge_label_xy(x1::Float64, y1::Float64, x2::Float64, y2::Float64, label::String)
    # Deterministically spread labels along and around edges to reduce overlaps.
    s = sum(Int(c) for c in label)
    t_choices = (0.35, 0.50, 0.65)
    n_choices = (-0.022, 0.0, 0.022)
    t = t_choices[(s % length(t_choices)) + 1]
    n = n_choices[(s % length(n_choices)) + 1]

    xm = x1 + t * (x2 - x1)
    ym = y1 + t * (y2 - y1)
    dx = x2 - x1
    dy = y2 - y1
    norm = sqrt(dx * dx + dy * dy)
    if norm > 1e-12
        nx = -dy / norm
        ny = dx / norm
        xm += n * nx
        ym += n * ny
    end
    return xm, ym
end

function _set_centered_axis_limits!(ax, pos::Dict{String, Tuple{Float64, Float64}}; xpad::Float64=0.20, ypad::Float64=0.12)
    if isempty(pos)
        return
    end
    xs = [xy[1] for xy in values(pos)]
    ys = [xy[2] for xy in values(pos)]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
end

function _plot_original_cocluster_graph(mbp::MultiblockProblem, outfile::String)
    variable_nodes = sort([string(b.id) for b in mbp.blocks if !startswith(string(b.id), "s_")])
    constraint_nodes = Set{String}()
    edge_list = Vector{Tuple{String, String, String}}() # (u, v, label)
    for (idx, constr) in enumerate(mbp.constraints)
        cid = "C$(idx)"
        non_slack_blocks = String[]
        for bid in constr.involvedBlocks
            bname = string(bid)
            startswith(bname, "s_") && continue
            push!(non_slack_blocks, bname)
        end
        if length(non_slack_blocks) == 2
            u, v = non_slack_blocks[1], non_slack_blocks[2]
            # Two-block constraints are direct variable-variable edges.
            push!(edge_list, (u, v, "C$(idx)"))
        elseif length(non_slack_blocks) > 2
            push!(constraint_nodes, cid)
            for bname in non_slack_blocks
                var_idx = _extract_variable_index(bname)
                elabel = var_idx === nothing ? "C$(idx)" : "C$(idx)$(var_idx)"
                push!(edge_list, (bname, cid, elabel))
            end
        end
    end

    all_nodes = Set{String}(vcat(variable_nodes, collect(constraint_nodes)))
    left_pref = ["C_2", "x_1", "C_4"]
    right_pref = ["x_2", "x_3", "x_4"]

    left_nodes = String[]
    for raw in left_pref
        resolved = _resolve_node_name(raw, all_nodes)
        resolved === nothing && continue
        resolved in left_nodes || push!(left_nodes, resolved)
    end

    right_nodes = String[]
    for raw in right_pref
        resolved = _resolve_node_name(raw, all_nodes)
        resolved === nothing && continue
        resolved in right_nodes || push!(right_nodes, resolved)
    end

    # Append any remaining nodes so the function still works for other instances.
    assigned = Set{String}(vcat(left_nodes, right_nodes))
    for n in sort(collect(all_nodes))
        n in assigned && continue
        if startswith(n, "C")
            push!(left_nodes, n)
        else
            push!(right_nodes, n)
        end
    end

    pos = _bipartite_positions(left_nodes, right_nodes)

    fig, ax = subplots(figsize=(6.6, 4.2))
    for (u, v, label) in edge_list
        haskey(pos, u) && haskey(pos, v) || continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#555555", linewidth=1.1, alpha=0.85)
        xm, ym = _edge_label_xy(x1, y1, x2, y2, label)
        ax.text(xm, ym + 0.025, _latex_symbol_label(label),
            ha="center", va="center", fontsize=9, color="#222222",
            bbox=Dict("facecolor" => "white", "edgecolor" => "none", "alpha" => 0.7, "pad" => 0.1),
            zorder=5)
    end
    for n in left_nodes
        x, y = pos[n]
        node_color = startswith(n, "C") ? "#90ee90" : "#8ecae6"
        ax.scatter([x], [y], s=2160, c=node_color, edgecolors="#333333", linewidths=1.0, zorder=3)
        ax.text(x, y, _latex_symbol_label(n), ha="center", va="center", fontsize=10, zorder=4)
    end
    for n in right_nodes
        x, y = pos[n]
        node_color = startswith(n, "C") ? "#90ee90" : "#8ecae6"
        ax.scatter([x], [y], s=2160, c=node_color, edgecolors="#333333", linewidths=1.0, zorder=3)
        ax.text(x, y, _latex_symbol_label(n), ha="center", va="center", fontsize=10, zorder=4)
    end
    _set_centered_axis_limits!(ax, pos; xpad=0.24, ypad=0.14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.03)
    close(fig)
    println("[info] saved graph -> $(outfile)")
end

function _plot_admm_bipartized_graph(admm_graph::ADMMBipartiteGraph, outfile::String, title_text::String; left_anchor::Union{Nothing,String}=nothing)
    left_nodes = sort(copy(admm_graph.left))
    right_nodes = sort(copy(admm_graph.right))

    # A bipartition and its global left/right complement are the same solution.
    # Apply an explicit visualization-only anchor when solver tie-breaking may
    # otherwise mirror an identical published graph.
    if left_anchor !== nothing && left_anchor in right_nodes
        left_nodes, right_nodes = right_nodes, left_nodes
    end
    pos = _bipartite_positions(left_nodes, right_nodes)

    fig, ax = subplots(figsize=(6.6, 4.2))
    for (_, e) in admm_graph.edges
        haskey(pos, e.nodeID1) && haskey(pos, e.nodeID2) || continue
        x1, y1 = pos[e.nodeID1]
        x2, y2 = pos[e.nodeID2]
        ax.plot([x1, x2], [y1, y2], color="#555555", linewidth=1.0, alpha=0.8)
    end

    aux_counter = 0
    label_cache = Dict{String, String}()
    for node in vcat(left_nodes, right_nodes)
        if startswith(node, "ADMMNodeConvertedFromEdge(")
            aux_counter += 1
            converted = get(admm_graph.nodes, node, nothing)
            if converted === nothing || isempty(converted.convertedEdgeID)
                label_cache[node] = "aux$(aux_counter)"
            else
                label_cache[node] = _constraint_label_from_edge_id(converted.convertedEdgeID)
            end
        else
            label_cache[node] = _trim_node_label(node)
        end
    end

    for node in left_nodes
        x, y = pos[node]
        ax.scatter([x], [y], s=1960, c=_node_color(node), edgecolors="#333333", linewidths=1.0, zorder=3)
        ax.text(x, y, _latex_symbol_label(label_cache[node]), ha="center", va="center", fontsize=9, zorder=4)
    end
    for node in right_nodes
        x, y = pos[node]
        ax.scatter([x], [y], s=1960, c=_node_color(node), edgecolors="#333333", linewidths=1.0, zorder=3)
        ax.text(x, y, _latex_symbol_label(label_cache[node]), ha="center", va="center", fontsize=9, zorder=4)
    end
    _set_centered_axis_limits!(ax, pos; xpad=0.24, ypad=0.14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.03)
    close(fig)
    println("[info] saved graph -> $(outfile)")
end


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        usage()
        exit(1)
    end

    mps_path   = abspath(ARGS[1])
    outdir = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT_DIR
    isdir(outdir) || mkpath(outdir)

    lp = GenericLP(mps_path)

    cocluster_k = 4
    cocluster_iters = 10
    cocluster_force_split = true
    cocluster_promote_pairwise_rows = true
    initialRho = 1000.0 
    maxIter = 100000
    logInterval = 1000

    inspect_cocluster(mps_path;
        output_dir=outdir,
        k=cocluster_k,
        iters=cocluster_iters,
        forceSplit=cocluster_force_split,
        promotePairwiseRows=cocluster_promote_pairwise_rows)

    # Export graph views for the coclustered MBP and its bipartizations.
    mbp_graph = generateGenericLPWithCoClustering(lp;
        k=cocluster_k,
        iters=cocluster_iters,
        forceSplitSingleBlock=cocluster_force_split,
        promotePairwiseRows=cocluster_promote_pairwise_rows)
    _plot_original_cocluster_graph(mbp_graph, joinpath(outdir, "graph_original_coclustered.png"))

    graph_for_milp = MultiblockGraph(mbp_graph)
    admm_graph_milp = ADMMBipartiteGraph(graph_for_milp, mbp_graph, MILP_BIPARTIZATION, 1)
    _plot_admm_bipartized_graph(
        admm_graph_milp,
        joinpath(outdir, "graph_bipartization_milp.png"),
        "Bipartization Graph (MILP)";
        left_anchor="VariableNode(x_1)",
    )

    graph_for_bfs = MultiblockGraph(mbp_graph)
    admm_graph_bfs = ADMMBipartiteGraph(graph_for_bfs, mbp_graph, BFS_BIPARTIZATION, 1)
    _plot_admm_bipartized_graph(admm_graph_bfs, joinpath(outdir, "graph_bipartization_bfs.png"), "Bipartization Graph (BFS)")

    results = Dict()
     try 
        mbp = generateGenericLP(lp)
        param = ADMMParam(
            initialRho = initialRho,
            maxIter = maxIter,
            logInterval = logInterval,
            solver = DoublyLinearizedSolver(),
            applyScaling = false
        )
        results["Basic"] = runBipartiteADMM(mbp, param)
    catch e
        @error "Failed to solve the problem with classic bipartization." exception=(e, catch_backtrace())
        return
    end


    try 
        mbp = generateGenericLPWithCoClustering(lp;
            k=cocluster_k,
            iters=cocluster_iters,
            forceSplitSingleBlock=cocluster_force_split,
            promotePairwiseRows=cocluster_promote_pairwise_rows)
        param = ADMMParam(
            initialRho = initialRho,
            maxIter = maxIter,
            logInterval = logInterval,
            solver = DoublyLinearizedSolver(),
            applyScaling = false
        )
        results["BFS"] = runBipartiteADMM(mbp, param; bipartizationAlgorithm = BFS_BIPARTIZATION)
    catch e
        @error "Failed to solve the problem with BFS bipartization." exception=(e, catch_backtrace())
        return
    end

    try 
        mbp = generateGenericLPWithCoClustering(lp;
            k=cocluster_k,
            iters=cocluster_iters,
            forceSplitSingleBlock=cocluster_force_split,
            promotePairwiseRows=cocluster_promote_pairwise_rows)
        param = ADMMParam(
            initialRho = initialRho,
            maxIter = maxIter,
            logInterval = logInterval,
            solver = DoublyLinearizedSolver(),
            applyScaling = false
        )
        results["MILP"] = runBipartiteADMM(mbp, param; bipartizationAlgorithm = MILP_BIPARTIZATION)
    catch e
        @error "Failed to solve the problem with MILP bipartization." exception=(e, catch_backtrace())
        return
    end

    _plot_residuals(
        results,
        :presL2,
        "Primal Residuals",
        joinpath(outdir, "primal_residuals.png"),
        k=500,
    )
    _plot_residuals(
        results,
        :dresL2,
        "Dual Residuals",
        joinpath(outdir, "dual_residuals.png"),
        k=1,
    )

end 
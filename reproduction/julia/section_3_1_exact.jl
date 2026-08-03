#!/usr/bin/env julia

# Add-only reviewer driver for the exact Section 3.1 manuscript experiment.
#
# This driver intentionally does not call heuristic_coclustering from
# applications/GenericLP/GenericLP.jl.  That helper has evolved since the
# manuscript experiment: it shuffles the cyclic initialization, uses additional
# tie breakers, and groups rows by support signatures.  The implementation
# below follows the three manuscript steps literally.
import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(REPO_ROOT)

using JSON
using JuMP
using LinearAlgebra
import MathOptInterface
using PDMO
using Printf
using PyPlot
using Random
using SparseArrays

include(joinpath(REPO_ROOT, "applications", "GenericLP", "GenericLP.jl"))

const PAPER_K = 4
const PAPER_PASSES = 5
const PAPER_RHO = 1000.0
# The original enlight_hard experiment seeds the global RNG at 126.  Its
# historical co-clustering helper separately owns a MersenneTwister(42), but
# the literal cyclic manuscript co-clustering below is deterministic and does
# not consume that helper-local RNG.  This seed controls the Basic random
# initial point and must therefore remain 126.
const PAPER_SEED = 126
const PAPER_MIP_REL_GAP = 0.01
const PAPER_MIP_TIME_LIMIT = 60.0
const PAPER_MIP_HEURISTIC_EFFORT = 0.2
const RectPatch = PyPlot.matplotlib[:patches][:Rectangle]

function usage()
    println("Usage: julia section_3_1_exact.jl MPS_PATH OUTPUT_DIR [MAX_ITER]")
end

"""Return adjacency lists for the nonzero pattern of `A`."""
function incidence_lists(A::SparseMatrixCSC)
    number_rows, number_cols = size(A)
    row_to_cols = [Int[] for _ in 1:number_rows]
    col_to_rows = [Int[] for _ in 1:number_cols]
    @inbounds for col in 1:number_cols
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[ptr]
            push!(row_to_cols[row], col)
            push!(col_to_rows[col], row)
        end
    end
    return row_to_cols, col_to_rows
end

"""Index of the first maximum; all-zero counts therefore select cluster 1."""
function smallest_argmax(counts::Vector{Int})
    isempty(counts) && error("Cannot select a cluster from an empty count vector")
    best = 1
    @inbounds for index in 2:length(counts)
        if counts[index] > counts[best]
            best = index
        end
    end
    return best
end

"""
    manuscript_coclustering(A, k, passes)

Implement the co-clustering procedure stated in Section 3.1:

1. assign columns cyclically, without shuffling, and put every row in cluster 1;
2. for each fixed pass, assign rows from incident column-cluster counts and then
   assign columns from incident row-cluster counts;
3. break every tie toward the smallest cluster index.

No load-balancing tie breaker, previous-label preference, empty-cluster repair,
early stopping, or pairwise-row promotion is applied.
"""
function manuscript_coclustering(A::SparseMatrixCSC, k::Int, passes::Int)
    number_rows, number_cols = size(A)
    k > 0 || error("The number of clusters must be positive")
    passes >= 0 || error("The number of passes must be nonnegative")
    number_cols >= k || error("Cannot form $(k) column clusters from $(number_cols) columns")

    row_to_cols, col_to_rows = incidence_lists(A)
    col_cluster = [mod(col - 1, k) + 1 for col in 1:number_cols]
    row_cluster = ones(Int, number_rows)
    counts = zeros(Int, k)

    for _ in 1:passes
        @inbounds for row in 1:number_rows
            fill!(counts, 0)
            for col in row_to_cols[row]
                counts[col_cluster[col]] += 1
            end
            row_cluster[row] = smallest_argmax(counts)
        end

        @inbounds for col in 1:number_cols
            fill!(counts, 0)
            for row in col_to_rows[col]
                counts[row_cluster[row]] += 1
            end
            col_cluster[col] = smallest_argmax(counts)
        end
    end

    return row_cluster, col_cluster, row_to_cols
end

"""Form one block constraint from each final row cluster."""
function manuscript_constraint_groups(
    row_cluster::Vector{Int},
    col_cluster::Vector{Int},
    row_to_cols::Vector{Vector{Int}},
    k::Int,
)
    group_rows = [Int[] for _ in 1:k]
    @inbounds for row in eachindex(row_cluster)
        push!(group_rows[row_cluster[row]], row)
    end

    group_blocks = Vector{Vector{Int}}(undef, k)
    for cluster in 1:k
        present = falses(k)
        for row in group_rows[cluster]
            for col in row_to_cols[row]
                present[col_cluster[col]] = true
            end
        end
        group_blocks[cluster] = findall(identity, present)
    end

    empty_row_clusters = findall(isempty, group_rows)
    isempty(empty_row_clusters) || error(
        "The literal manuscript algorithm produced empty row clusters $(empty_row_clusters); " *
        "the paper reports four nonempty block constraints for enlight_hard.",
    )
    empty_col_clusters = [cluster for cluster in 1:k if !(cluster in col_cluster)]
    isempty(empty_col_clusters) || error(
        "The literal manuscript algorithm produced empty column clusters $(empty_col_clusters); " *
        "the paper reports four nonempty block variables for enlight_hard.",
    )
    empty_group_support = findall(isempty, group_blocks)
    isempty(empty_group_support) || error(
        "Row clusters $(empty_group_support) contain no nonzero coefficient and cannot form constraints.",
    )

    return group_rows, group_blocks
end

function cluster_spans(groups::Vector{Vector{Int}})
    spans = Tuple{Int, Int}[]
    offset = 0
    for group in groups
        first_index = offset + 1
        offset += length(group)
        push!(spans, (first_index, offset))
    end
    return spans
end

function configure_matrix_axis!(axis, number_rows::Int, number_cols::Int)
    axis.set_xlabel("Columns")
    axis.set_ylabel("Rows")
    axis.set_xlim(-0.5, number_cols - 0.5)
    axis.set_ylim(number_rows - 0.5, -0.5)
end

function draw_cluster_boundaries!(axis, row_spans, col_spans; linewidth=0.8, linestyle="--")
    for index in 1:(length(row_spans) - 1)
        axis.axhline(row_spans[index][2] - 0.5; color="black", linewidth=linewidth, linestyle=linestyle)
    end
    for index in 1:(length(col_spans) - 1)
        axis.axvline(col_spans[index][2] - 0.5; color="black", linewidth=linewidth, linestyle=linestyle)
    end
end

"""Write the exact-clustering matrix sources consumed by section_3_1.py."""
function write_matrix_figures(
    lp::GenericLP,
    group_rows::Vector{Vector{Int}},
    col_cluster::Vector{Int},
    group_blocks::Vector{Vector{Int}},
    output_dir::String,
)
    col_groups = [findall(==(cluster), col_cluster) for cluster in 1:PAPER_K]
    row_permutation = reduce(vcat, group_rows)
    col_permutation = reduce(vcat, col_groups)
    length(row_permutation) == lp.number_rows || error("Row permutation is incomplete")
    length(col_permutation) == lp.number_cols || error("Column permutation is incomplete")
    length(unique(row_permutation)) == lp.number_rows || error("Row permutation is duplicated")
    length(unique(col_permutation)) == lp.number_cols || error("Column permutation is duplicated")

    row_spans = cluster_spans(group_rows)
    col_spans = cluster_spans(col_groups)
    permuted = lp.A[row_permutation, col_permutation]

    figure, axis = subplots(figsize=(6, 6))
    axis.spy(lp.A; markersize=1)
    configure_matrix_axis!(axis, lp.number_rows, lp.number_cols)
    figure.tight_layout()
    figure.savefig(
        joinpath(output_dir, "matrix_original.png");
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    close(figure)

    figure, axis = subplots(figsize=(6, 6))
    axis.spy(permuted; markersize=1)
    draw_cluster_boundaries!(axis, row_spans, col_spans; linewidth=0.8, linestyle=":")
    configure_matrix_axis!(axis, lp.number_rows, lp.number_cols)
    figure.tight_layout()
    figure.savefig(
        joinpath(output_dir, "matrix_coclustered.png");
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    close(figure)

    figure, axis = subplots(figsize=(6, 6))
    axis.spy(permuted; markersize=1)
    draw_cluster_boundaries!(axis, row_spans, col_spans; linewidth=0.6, linestyle=":")
    for row_group in 1:length(group_rows)
        row_start, row_stop = row_spans[row_group]
        for col_group in group_blocks[row_group]
            col_start, col_stop = col_spans[col_group]
            rectangle = RectPatch(
                (col_start - 1.5, row_start - 1.5),
                col_stop - col_start + 1,
                row_stop - row_start + 1;
                fill=false,
                edgecolor="black",
                linewidth=1.0,
                linestyle="--",
            )
            axis.add_patch(rectangle)
        end
    end
    configure_matrix_axis!(axis, lp.number_rows, lp.number_cols)
    figure.tight_layout()
    figure.savefig(
        joinpath(output_dir, "matrix_coclustered_stacked.png");
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    close(figure)

    return col_groups
end

function node_kind(node_id::String)
    startswith(node_id, "VariableNode(") && return "variable"
    startswith(node_id, "ConstraintNode(") && return "constraint"
    startswith(node_id, "ADMMNodeConvertedFromEdge(") && return "auxiliary"
    return "unknown"
end

function original_graph_payload(graph::MultiblockGraph)
    nodes = [
        Dict(
            "id" => node_id,
            "kind" => node_kind(node_id),
            "source" => string(node.source),
        )
        for (node_id, node) in sort(collect(graph.nodes); by=first)
    ]
    edges = [
        Dict(
            "id" => edge_id,
            "u" => edge.nodeID1,
            "v" => edge.nodeID2,
            "label" => "C$(edge.sourceBlockConstraint)",
        )
        for (edge_id, edge) in sort(collect(graph.edges); by=first)
    ]
    return Dict("nodes" => nodes, "edges" => edges, "left" => String[], "right" => String[])
end

function bipartite_graph_payload(graph::ADMMBipartiteGraph)
    nodes = [
        Dict(
            "id" => node_id,
            "kind" => node_kind(node_id),
            "converted_edge" => node.convertedEdgeID,
            "assignment" => node.assignment,
        )
        for (node_id, node) in sort(collect(graph.nodes); by=first)
    ]
    edges = [
        Dict(
            "id" => edge_id,
            "u" => edge.nodeID1,
            "v" => edge.nodeID2,
            "label" => edge.splittedEdgeID,
        )
        for (edge_id, edge) in sort(collect(graph.edges); by=first)
    ]
    return Dict(
        "nodes" => nodes,
        "edges" => edges,
        "left" => sort(copy(graph.left)),
        "right" => sort(copy(graph.right)),
        "partition_time_seconds" => graph.partitionAlgorithmTime,
    )
end

function decisions_payload(nodes_assignment, edges_splitting)
    return Dict(
        "node_assignments" => Dict(key => value for (key, value) in sort(collect(nodes_assignment); by=first)),
        "edge_splitting" => Dict(
            key => [value[1], value[2]]
            for (key, value) in sort(collect(edges_splitting); by=first)
        ),
    )
end

"""Deterministic Algorithm 2 traversal with lexical node and edge ordering."""
function deterministic_bfs_bipartization(
    graph::MultiblockGraph,
    nodes_assignment::Dict{String, Int64},
    edges_splitting::Dict{String, Tuple{Int64, Int64}},
)
    empty!(nodes_assignment)
    empty!(edges_splitting)
    neighbors = Dict(
        node_id => Tuple{String, String}[] for node_id in sort(collect(keys(graph.nodes)))
    )
    for edge_id in sort(collect(keys(graph.edges)))
        edge = graph.edges[edge_id]
        push!(neighbors[edge.nodeID1], (edge_id, edge.nodeID2))
        push!(neighbors[edge.nodeID2], (edge_id, edge.nodeID1))
    end
    for neighbor_values in values(neighbors)
        sort!(neighbor_values; by=item -> (item[2], item[1]))
    end

    start_partition = 0
    for start_node in sort(collect(keys(graph.nodes)))
        haskey(nodes_assignment, start_node) && continue
        nodes_assignment[start_node] = start_partition
        queue = String[start_node]
        while !isempty(queue)
            current = popfirst!(queue)
            current_partition = nodes_assignment[current]
            for (edge_id, neighbor) in neighbors[current]
                if !haskey(nodes_assignment, neighbor)
                    nodes_assignment[neighbor] = 1 - current_partition
                    push!(queue, neighbor)
                elseif nodes_assignment[neighbor] == current_partition
                    edges_splitting[edge_id] = (1, 1 - current_partition)
                end
            end
        end
        start_partition = 1 - start_partition
    end
    for edge_id in sort(collect(keys(graph.edges)))
        get!(edges_splitting, edge_id, (0, 0))
    end
end

"""
Build a graph without the high-level constructor's silent MILP-to-BFS fallback.
The returned graph object is later passed directly to ADMM.
"""
function build_exact_admm_graph(
    graph::MultiblockGraph,
    mbp::MultiblockProblem,
    algorithm::Symbol,
)
    nodes_assignment = Dict{String, Int64}()
    edges_splitting = Dict{String, Tuple{Int64, Int64}}()

    if graph.isBipartite
        for (node_id, assignment) in graph.colors
            nodes_assignment[node_id] = assignment
        end
        for edge_id in keys(graph.edges)
            edges_splitting[edge_id] = (0, 0)
        end
        elapsed = 0.0
    else
        started = time()
        if algorithm == :bfs
            deterministic_bfs_bipartization(graph, nodes_assignment, edges_splitting)
        elseif algorithm == :milp
            # Call the algorithm directly: ADMMBipartiteGraph's convenience
            # constructor catches every MILP error and silently substitutes BFS.
            MilpBipartization(
                graph,
                mbp,
                nodes_assignment,
                edges_splitting;
                mipRelGap=PAPER_MIP_REL_GAP,
                mipTimeLimit=PAPER_MIP_TIME_LIMIT,
                mipHeuristicEffort=PAPER_MIP_HEURISTIC_EFFORT,
            )
        else
            error("Unsupported bipartization algorithm: $(algorithm)")
        end
        elapsed = time() - started
    end

    isempty(nodes_assignment) && error("$(algorithm) produced no node assignments")
    length(nodes_assignment) == length(graph.nodes) || error(
        "$(algorithm) assigned $(length(nodes_assignment)) of $(length(graph.nodes)) graph nodes",
    )
    length(edges_splitting) == length(graph.edges) || error(
        "$(algorithm) decided $(length(edges_splitting)) of $(length(graph.edges)) graph edges",
    )

    admm_graph = ADMMBipartiteGraph(
        graph,
        mbp,
        nodes_assignment,
        edges_splitting,
        elapsed,
    )
    return admm_graph, nodes_assignment, edges_splitting
end

function make_parameter(max_iter::Int)
    return ADMMParam(
        initialRho=PAPER_RHO,
        maxIter=max_iter,
        logInterval=1,
        solver=DoublyLinearizedSolver(),
        applyScaling=false,
    )
end

"""Solve the exact graph object that was serialized for the structural figure."""
function solve_prebuilt(label::String, graph::ADMMBipartiteGraph, max_iter::Int)
    println("Running $(label) on its preconstructed ADMM graph")
    PDMO.summary(graph, 1)
    parameter = make_parameter(max_iter)
    parameter.applyScaling && error("A preconstructed graph requires applyScaling=false")
    info = PDMO.BipartiteADMM(graph, parameter)
    solver_name = PDMO.getADMMSubproblemSolverName(parameter.solver)
    solver_name == "DOUBLY_LINEARIZED_SOLVER" ||
        error("Doubly linearized solver initialization failed")
    info.partitionAlgorithmTime = graph.partitionAlgorithmTime
    PDMO.ADMMLog(info, parameter.logLevel)

    info.stopIter >= 1 || error("$(label) stopped before completing one iteration")
    length(info.presL2) == info.stopIter + 1 || error(
        "$(label) primal history has $(length(info.presL2)) points for stopIter=$(info.stopIter)",
    )
    length(info.dresL2) == info.stopIter + 1 || error(
        "$(label) dual history has $(length(info.dresL2)) points for stopIter=$(info.stopIter)",
    )
    string(info.terminationStatus) != "ADMM_TERMINATION_UNSPECIFIED" || error(
        "$(label) returned an unspecified terminal status",
    )
    return info
end

function write_json_file(path::String, payload)
    open(path, "w") do io
        JSON.print(io, payload, 2)
        println(io)
    end
end

function write_residuals(output_dir::String, results::Dict{String, Any})
    residual_path = joinpath(output_dir, "residuals.csv")
    open(residual_path, "w") do io
        println(io, "method,iteration,pres_l2,dres_l2,status,stop_iter,admm_time_seconds")
        for method in ("Basic", "BFS", "MILP")
            info = results[method]
            status = string(info.terminationStatus)
            # History index 1 is initialization/iteration 0.  The archived logs
            # and paper plot use completed ADMM iterations 1:stopIter, so skip it.
            for iteration in 1:info.stopIter
                history_index = iteration + 1
                println(
                    io,
                    join(
                        (
                            method,
                            iteration,
                            info.presL2[history_index],
                            info.dresL2[history_index],
                            status,
                            info.stopIter,
                            info.totalTime,
                        ),
                        ",",
                    ),
                )
            end
        end
    end
    return residual_path
end

function terminal_payload(results::Dict{String, Any})
    return Dict(
        method => Dict(
            "status" => string(info.terminationStatus),
            "stop_iter" => info.stopIter,
            "admm_time_seconds" => info.totalTime,
            "partition_time_seconds" => info.partitionAlgorithmTime,
            "exported_iterations" => [1, info.stopIter],
        )
        for (method, info) in results
    )
end

function main(args)
    length(args) in (2, 3) || (usage(); error("Expected two or three arguments"))
    mps_path = abspath(args[1])
    output_dir = abspath(args[2])
    max_iter = length(args) == 3 ? parse(Int, args[3]) : 100_000
    isfile(mps_path) || error("MPS file not found: $(mps_path)")
    max_iter > 0 || error("MAX_ITER must be positive")
    mkpath(output_dir)

    Random.seed!(PAPER_SEED)
    println("Section 3.1 literal manuscript configuration:")
    println("  mps_path = $(mps_path)")
    println("  continuous_relaxation = true")
    println("  threads = $(Threads.nthreads())")
    println("  k = $(PAPER_K)")
    println("  alternating_passes = $(PAPER_PASSES)")
    println("  column_initialization = cyclic, unshuffled")
    println("  tie_break = smallest cluster index")
    println("  global_seed = $(PAPER_SEED)")
    println("  rho = $(PAPER_RHO)")
    println("  max_iter = $(max_iter)")

    lp = GenericLP(mps_path)
    row_cluster, col_cluster, row_to_cols = manuscript_coclustering(
        lp.A,
        PAPER_K,
        PAPER_PASSES,
    )
    group_rows, group_blocks = manuscript_constraint_groups(
        row_cluster,
        col_cluster,
        row_to_cols,
        PAPER_K,
    )

    coclustered, exact_layout = generateGenericLPFromCoClustering(
        lp,
        row_cluster,
        col_cluster;
        k=PAPER_K,
        group_rows_override=group_rows,
        group_blocks_override=group_blocks,
        return_layout=true,
    )
    exact_layout.group_rows == group_rows || error("Exact row groups were not preserved by the MBP builder")
    exact_layout.group_blocks == group_blocks || error("Exact group supports were not preserved by the MBP builder")
    length(coclustered.constraints) == PAPER_K || error(
        "Expected $(PAPER_K) block constraints, constructed $(length(coclustered.constraints))",
    )

    col_groups = write_matrix_figures(lp, group_rows, col_cluster, group_blocks, output_dir)
    write_json_file(
        joinpath(output_dir, "exact_configuration.json"),
        Dict(
            "algorithm" => Dict(
                "column_initialization" => "cyclic_unshuffled",
                "row_initialization" => "all_in_cluster_1",
                "passes" => PAPER_PASSES,
                "tie_break" => "smallest_cluster_index",
                "row_grouping" => "final_row_cluster",
                "bfs_traversal_order" => "lexicographic_node_then_edge",
                "empty_cluster_repair" => false,
                "early_stopping" => false,
            ),
            "k" => PAPER_K,
            "rho" => PAPER_RHO,
            "global_seed" => PAPER_SEED,
            "solver" => "DOUBLY_LINEARIZED_SOLVER",
            "max_iter" => max_iter,
            "log_interval" => 1,
            "apply_scaling" => false,
            "threads" => Threads.nthreads(),
            "row_cluster" => row_cluster,
            "column_cluster" => col_cluster,
            "group_rows" => group_rows,
            "column_groups" => col_groups,
            "group_blocks" => group_blocks,
        ),
    )

    original_graph = MultiblockGraph(coclustered)
    bfs_graph, bfs_assignment, bfs_splitting = build_exact_admm_graph(
        original_graph,
        coclustered,
        :bfs,
    )
    milp_graph, milp_assignment, milp_splitting = build_exact_admm_graph(
        original_graph,
        coclustered,
        :milp,
    )

    original_payload = original_graph_payload(original_graph)
    bfs_payload = bipartite_graph_payload(bfs_graph)
    milp_payload = bipartite_graph_payload(milp_graph)
    write_json_file(
        joinpath(output_dir, "graphs.json"),
        Dict(
            "configuration" => Dict(
                "k" => PAPER_K,
                "passes" => PAPER_PASSES,
                "global_seed" => PAPER_SEED,
                "mip_rel_gap" => PAPER_MIP_REL_GAP,
                "mip_time_limit_seconds" => PAPER_MIP_TIME_LIMIT,
                "mip_heuristic_effort" => PAPER_MIP_HEURISTIC_EFFORT,
            ),
            "original" => original_payload,
            "bfs" => bfs_payload,
            "milp" => milp_payload,
            "bfs_decisions" => decisions_payload(bfs_assignment, bfs_splitting),
            "milp_decisions" => decisions_payload(milp_assignment, milp_splitting),
        ),
    )

    # Seed immediately before the only legacy helper that uses global rand: the
    # basic slack formulation initializes x with rand(number_cols).
    Random.seed!(PAPER_SEED)
    basic = generateGenericLP(lp)
    basic_source_graph = MultiblockGraph(basic)
    basic_graph, _, _ = build_exact_admm_graph(basic_source_graph, basic, :bfs)

    results = Dict{String, Any}(
        "Basic" => solve_prebuilt("Basic reformulation", basic_graph, max_iter),
        "BFS" => solve_prebuilt("BFS bipartization", bfs_graph, max_iter),
        "MILP" => solve_prebuilt("MILP bipartization", milp_graph, max_iter),
    )

    # The payload excludes numerical iterate state.  Equality here guarantees
    # the exact structural graph shown to the reviewer was the one solved.
    bipartite_graph_payload(bfs_graph) == bfs_payload || error("BFS graph structure changed during ADMM")
    bipartite_graph_payload(milp_graph) == milp_payload || error("MILP graph structure changed during ADMM")

    residual_path = write_residuals(output_dir, results)
    write_json_file(joinpath(output_dir, "terminal.json"), terminal_payload(results))
    println("Wrote exact Section 3.1 residuals to $(residual_path)")
    println("Wrote exact Section 3.1 structural artifacts to $(output_dir)")
end

main(ARGS)

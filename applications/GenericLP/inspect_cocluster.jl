#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    Pkg.activate(repo_root)
    # Warmup-style dependency setup for fresh environments.
    Pkg.instantiate()
end

using PDMO
using LinearAlgebra
using SparseArrays
using Random
using PyPlot
using Printf
using JuMP 
using MathOptInterface

include(joinpath(@__DIR__, "GenericLP.jl"))

const RectPatch = PyPlot.matplotlib[:patches][:Rectangle]

# """
#     inspect_cocluster(mps_path; output_dir=".", k=6, iters=5, forceSplit=true)

# Generate spy plots for `lp.A` before/after co-clustering, and write assignments.
# """
function inspect_cocluster(mps_path::AbstractString; 
    output_dir::AbstractString=".",
    k::Int=6, 
    iters::Int=5, 
    forceSplit::Bool=true,
    promotePairwiseRows::Bool=false)
    
    out_dir = abspath(output_dir)
    isdir(out_dir) || mkpath(out_dir)

    println("Loading LP: ", mps_path)
    lp = GenericLP(mps_path)
    A = lp.A

    function plot_spy(A::SparseMatrixCSC; show_clusters::Bool=false,
            row_perm=nothing, col_perm=nothing, row_labels=nothing, col_labels=nothing,
            group_row_spans=nothing, group_row_ids=nothing,
            block_col_spans=nothing, group_block_ids=nothing,
            title_str::AbstractString="")
        fig, ax = subplots(figsize=(6,6))
        img = row_perm === nothing ? A : A[row_perm, :]
        img = col_perm === nothing ? img : img[:, col_perm]
        ax.spy(img; markersize=1)
        ax.set_title(title_str)
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        seq_rows = row_perm === nothing ? collect(1:size(A,1)) : collect(1:length(row_perm))
        seq_cols = col_perm === nothing ? collect(1:size(A,2)) : collect(1:length(col_perm))
        row_labels === nothing && (row_labels = ones(Int, length(seq_rows)))
        col_labels === nothing && (col_labels = ones(Int, length(seq_cols)))

        row_edges = Int[]
        if length(row_labels) > 1
            @inbounds for idx in 1:length(row_labels)-1
                if row_labels[idx] != row_labels[idx+1]
                    push!(row_edges, idx)
                end
            end
        end
        col_edges = Int[]
        if length(col_labels) > 1
            @inbounds for idx in 1:length(col_labels)-1
                if col_labels[idx] != col_labels[idx+1]
                    push!(col_edges, idx)
                end
            end
        end
        if show_clusters && row_perm !== nothing && col_perm !== nothing &&
           group_row_spans !== nothing && block_col_spans !== nothing &&
           group_block_ids !== nothing
            for (span_idx, (row_start, row_end)) in enumerate(group_row_spans)
                gid = group_row_ids === nothing ? span_idx : group_row_ids[span_idx]
                blocks = gid <= length(group_block_ids) ? group_block_ids[gid] : Int[]
                isempty(blocks) && continue
                for b in blocks
                    haskey(block_col_spans, b) || continue
                    col_start, col_end = block_col_spans[b]
                    width = col_end - col_start + 1
                    height = row_end - row_start + 1
                    rect = RectPatch((col_start - 0.5, row_start - 0.5), width, height;
                        fill=false, edgecolor="black", linewidth=0.9, linestyle="--", alpha=0.8)
                    ax.add_patch(rect)
                end
            end
        end
        tight_layout()
        fig
    end

    mbp, layout = generateGenericLPWithCoClustering(
        lp;
        k = k,
        iters = iters,
        rng = MersenneTwister(42),
        forceSplitSingleBlock = forceSplit,
        promotePairwiseRows = promotePairwiseRows,
        return_layout = true,
    )
    row_block = layout.row_block
    col_block = layout.col_block
    group_block_ids = layout.group_blocks

    row_group_assign = zeros(Int, lp.number_rows)
    row_perm = Int[]
    group_row_spans = Tuple{Int,Int}[]
    group_row_ids = Int[]
    offset = 0
    for (g_idx, rows) in enumerate(layout.group_rows)
        isempty(rows) && continue
        append!(row_perm, rows)
        for i in rows
            row_group_assign[i] = g_idx
        end
        start_idx = offset + 1
        offset += length(rows)
        push!(group_row_spans, (start_idx, offset))
        push!(group_row_ids, g_idx)
    end
    @assert length(row_perm) == lp.number_rows "Row permutation missing entries"

    col_perm = Int[]
    block_col_spans = Dict{Int,Tuple{Int,Int}}()
    offset = 0
    for b in 1:layout.k_cols
        Jb = layout.cols_in_block[b]
        isempty(Jb) && continue
        append!(col_perm, Jb)
        start_idx = offset + 1
        offset += length(Jb)
        block_col_spans[b] = (start_idx, offset)
    end
    @assert length(col_perm) == lp.number_cols "Column permutation missing entries"

    raw_path = joinpath(out_dir, "matrix_original.png")
    fig_original = plot_spy(A; title_str="")
    fig_original.savefig(raw_path, dpi=200)
    close(fig_original)
    println("Saved original matrix plot to ", raw_path)

    clus_path = joinpath(out_dir, "matrix_coclustered.png")
    fig_clustered = plot_spy(A; row_perm=row_perm, col_perm=col_perm,
        row_labels=row_group_assign[row_perm], col_labels=col_block[col_perm],
        group_row_spans=group_row_spans, group_row_ids=group_row_ids,
        block_col_spans=block_col_spans, group_block_ids=group_block_ids,
        show_clusters=true, title_str="")
    fig_clustered.savefig(clus_path, dpi=200)
    close(fig_clustered)
    println("Saved coclustered matrix plot with block boundaries to ", clus_path)

    nonempty_blocks = [b for b in 1:layout.k_cols if !isempty(layout.cols_in_block[b])]
    block_label_map = Dict{Int,String}(b => string("\$x_{", idx, "}\$") for (idx, b) in enumerate(nonempty_blocks))
    group_views = Vector{NamedTuple{(:rows, :cols, :blocks, :block_spans, :row_range)}}()
    marker_size_for_rows(n::Int) = n == 1 ? 14 : n <= 3 ? 8 : n <= 6 ? 4 : 2
    # Per-constraint zoomed plots
    for (r, rows_span) in enumerate(group_row_spans)
        row_indices = rows_span[1]:rows_span[2]
        row_sel = row_perm[row_indices]
        blocks = group_block_ids[r]
        isempty(blocks) && continue
        isempty(col_perm) && continue
        col_sel = copy(col_perm)
        block_spans = Tuple{Int,Int}[]
        for b in blocks
            haskey(block_col_spans, b) || continue
            push!(block_spans, block_col_spans[b])
        end
        isempty(block_spans) && continue
        subA = A[row_sel, col_sel]
        fig_grp, ax_grp = subplots(figsize=(5,5))
        ax_grp.spy(subA; markersize=marker_size_for_rows(length(row_sel)))
        ax_grp.invert_yaxis()
        latex_blocks = [get(block_label_map, b, string("\$x_{", b, "}\$")) for b in blocks]
        block_label = join(latex_blocks, ", ")
        ax_grp.set_title("")
        ax_grp.set_xlabel("Columns (global order)")
        ax_grp.set_yticks([])
        ax_grp.set_ylabel("")
        ax_grp.set_ylim(length(row_sel) + 0.5, 0.5)
        ax_grp.text(1.02, 0.5,
            "$(length(row_sel)) rows coupling $(block_label)";
            transform=ax_grp.transAxes, va="center", ha="left", fontsize=8,
            rotation=90)
        for span in block_spans
            start_idx, end_idx = span
            ax_grp.axvline(start_idx - 0.5; color="black", linestyle="--", linewidth=0.8)
            ax_grp.axvline(end_idx + 0.5; color="black", linestyle="--", linewidth=0.8)
        end
        grp_path = joinpath(out_dir, "matrix_coclustered_group_$r.png")
        fig_grp.tight_layout()
        fig_grp.savefig(grp_path, dpi=200)
        close(fig_grp)
        println("Saved constraint-specific plot to ", grp_path)

        push!(group_views, (
            rows = copy(row_sel),
            cols = copy(col_perm),
            blocks = copy(blocks),
            block_spans = copy(block_spans),
            row_range = (minimum(row_sel), maximum(row_sel))
        ))
    end

    if !isempty(group_views)
        max_rows = maximum(length(view.rows) for view in group_views)
        height_ratios = [0.9 + 0.4 * (length(view.rows) / max_rows) for view in group_views]
        fig_height = 1.1 * sum(height_ratios)
        fig_stack, axes = subplots(length(group_views), 1;
            figsize=(6, fig_height),
            gridspec_kw=Dict(:height_ratios => height_ratios),
            sharex=true)
        if length(group_views) == 1
            axes = [axes]
        end
        global_cols = length(first(group_views).cols)
        for ax in axes
            ax.set_xlim(0.5, global_cols + 0.5)
        end
        for (idx, view) in enumerate(group_views)
            ax = axes[idx]
            subA = A[view.rows, view.cols]
            ax.spy(subA; markersize=marker_size_for_rows(length(view.rows)))
            ax.set_aspect("auto")
            ax.invert_yaxis()
            row_min, row_max = view.row_range
            ax.set_yticks([])
            ax.set_ylim(length(view.rows) + 0.5, 0.5)
            latex_blocks = [get(block_label_map, b, string("\$x_{", b, "}\$")) for b in view.blocks]
            block_label = join(latex_blocks, ", ")
            ax.text(1.01, 0.5, "$(length(view.rows)) rows coupling $(block_label)";
                transform=ax.transAxes, va="center", ha="left", fontsize=8)
            ax.set_title("")
            for span in view.block_spans
                start_idx, end_idx = span
                ax.axvline(start_idx - 0.5; color="black", linestyle="--", linewidth=0.8)
                ax.axvline(end_idx + 0.5; color="black", linestyle="--", linewidth=0.8)
            end
            # axes share the same global column axis; keep labels off for compactness
            if idx == 1
                ax.set_xticks(0:50:global_cols)
                ax.set_xlabel("")
            else
                ax.set_xticks([])
                ax.set_xlabel("")
            end
        end
        fig_stack.tight_layout(h_pad=0.05)
        fig_stack.subplots_adjust(hspace=0.02)
        stack_path = joinpath(out_dir, "matrix_coclustered_stacked.png")
        fig_stack.savefig(stack_path, dpi=200)
        close(fig_stack)
        println("Saved stacked constraint plot to ", stack_path)
    end

    if !isempty(layout.group_rows) && !isempty(nonempty_blocks)
        block_idx = Dict(b => idx for (idx, b) in enumerate(nonempty_blocks))
        coupling = zeros(Int, length(layout.group_rows), length(nonempty_blocks))
        for (g_idx, blocks) in enumerate(group_block_ids)
            for b in blocks
                haskey(block_idx, b) || continue
                coupling[g_idx, block_idx[b]] = 1
            end
        end
        coup_path = joinpath(out_dir, "matrix_block_couplings.png")
        fig_coup, ax = subplots(figsize=(6, min(6, 3 + 0.3*length(layout.group_rows))))
        ax.imshow(coupling; cmap="Greys", interpolation="nearest", aspect="auto")
        ax.set_xlabel("Block Variables")
        ax.set_ylabel("Block Constraints")
        ax.set_xticks(0:length(nonempty_blocks)-1)
        xtick_labels = [string("\$x_{", idx, "}\$") for (idx, _) in enumerate(nonempty_blocks)]
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_yticks(0:length(layout.group_rows)-1)
        ax.set_yticklabels(["C$(i)" for i in 1:length(layout.group_rows)])
        # draw horizontal separators between constraint rows
        for r in 0:length(layout.group_rows)
            ax.axhline(r - 0.5; color="black", linewidth=0.4, alpha=0.5)
        end
        fig_coup.tight_layout()
        fig_coup.savefig(coup_path, dpi=200)
        close(fig_coup)
        println("Saved block coupling heatmap to ", coup_path)
    end

    cluster_info = joinpath(out_dir, "matrix_cocluster_info.txt")
    open(cluster_info, "w") do io
        println(io, "row_block assignments (raw clustering):")
        for i in 1:length(row_block)
            println(io, @sprintf("row %6d -> block %3d", i, row_block[i]))
        end
        println(io, "\ncol_block assignments (raw clustering):")
        for j in 1:length(col_block)
            println(io, @sprintf("col %6d -> block %3d", j, col_block[j]))
        end
        println(io, "\nconstraint groups used in MBP (ignoring slack blocks):")
        for (idx, rows) in enumerate(layout.group_rows)
            blocks = group_block_ids[idx]
            block_names = ["x_$b" for b in blocks]
            println(io, "group $(idx): rows = $(rows), blocks = $(block_names)")
        end

        println(io, "\nconstraint-to-block mapping read from MBP (ignoring slack blocks):")
        for (idx, constr) in enumerate(mbp.constraints)
            block_names = String[]
            for bid in constr.involvedBlocks
                s = string(bid)
                startswith(s, "s_") && continue
                push!(block_names, s)
            end
            println(io, "constraint $(idx): blocks = $(block_names)")
        end
    end
    println("Saved block assignments to ", cluster_info)
end

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "enlight_hard_plots")

# CLI entry point
if abspath(PROGRAM_FILE) == @__FILE__
    function usage()
        println("""
Usage:
    julia applications/GenericLP/inspect_cocluster.jl <mps_path> [output_dir] [k] [iters] [forceSplit] [promotePairwiseRows]

Defaults:
    output_dir = applications/GenericLP/enlight_hard_plots
    k          = 6
    iters      = 5
    forceSplit = true
    promotePairwiseRows = false
""")
    end

    if length(ARGS) < 1
        usage()
        exit(1)
    end
    mps_path   = abspath(ARGS[1])
    output_dir = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT_DIR
    k          = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6
    iters      = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
    forceSplit = length(ARGS) >= 5 ? lowercase(ARGS[5]) in ("true","1","yes","y") : true
    promotePairwiseRows = length(ARGS) >= 6 ? lowercase(ARGS[6]) in ("true","1","yes","y") : false
    inspect_cocluster(mps_path;
        output_dir=output_dir,
        k=k,
        iters=iters,
        forceSplit=forceSplit,
        promotePairwiseRows=promotePairwiseRows)
end


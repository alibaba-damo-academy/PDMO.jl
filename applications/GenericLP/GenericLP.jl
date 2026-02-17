using Random
using SparseArrays

struct GenericLP
    number_cols::Int64 
    number_rows::Int64 
    obj::Vector{Float64}
    offset::Float64 
    A::SparseMatrixCSC
    row_lower::Vector{Float64}
    row_upper::Vector{Float64}
    col_lower::Vector{Float64}
    col_upper::Vector{Float64}
    col_names::Vector{String}
    row_names::Vector{String}
end 

function GenericLP(mps_dir::String)
    model = JuMP.read_from_file(mps_dir)
    vars = JuMP.all_variables(model)
    number_cols = length(vars)

    # obj 
    obj_expr = JuMP.objective_function(model)
    obj = zeros(number_cols)
    col_names = Vector{String}(undef, number_cols)
    for (i, v) in enumerate(vars)
        obj[i] = JuMP.coefficient(obj_expr, v)
        name_v = JuMP.name(v)
        col_names[i] = name_v == "" ? "PDMO_COL_$i" : name_v
    end 
    offset = JuMP.constant(obj_expr)
    # Normalize objective to minimization if needed
    if JuMP.objective_sense(model) == MathOptInterface.MAX_SENSE
        obj .*= -1.0
        offset *= -1.0
    end

    # constraint 
    constraints = JuMP.all_constraints(model; include_variable_in_set_constraints = true)
    # Keep only linear (row) constraints for rows/A; handle variable constraints separately
    linear_con_refs = [con_ref for con_ref in constraints if isa(JuMP.constraint_object(con_ref).func, JuMP.AffExpr)]
    number_rows = length(linear_con_refs)

    row_inds = Int[]
    col_inds = Int[]
    vals = Float64[]
    row_lower = zeros(number_rows)
    row_upper = zeros(number_rows)
    row_names = Vector{String}(undef, number_rows)

    var2Idx = Dict(vars[i]=>i for i in 1:number_cols)

    col_lower = ones(number_cols) * -Inf
    col_upper = ones(number_cols) * Inf
    isInteger = Bool[false for i in 1:number_cols]

    # Process linear row constraints: build A, bounds, and names
    for (row_idx, con_ref) in enumerate(linear_con_refs)
        con = JuMP.constraint_object(con_ref)

        name_c = JuMP.name(con_ref)
        row_names[row_idx] = name_c == "" ? "PDMO_ROW_$(row_idx)" : name_c

        aff = con.func
        for (v, coeff) in aff.terms
            j = var2Idx[v]
            push!(row_inds, row_idx)
            push!(col_inds, j)
            push!(vals, coeff)
        end

        # Extract the bounds based on the type of the constraint set.
        s = con.set
        if s isa MathOptInterface.EqualTo{Float64}
            row_lower[row_idx] = s.value
            row_upper[row_idx] = s.value
        elseif s isa MathOptInterface.Interval{Float64}
            row_lower[row_idx] = s.lower
            row_upper[row_idx] = s.upper
        elseif s isa MathOptInterface.LessThan{Float64}
            row_lower[row_idx] = -Inf
            row_upper[row_idx] = s.upper
        elseif s isa MathOptInterface.GreaterThan{Float64}
            row_lower[row_idx] = s.lower
            row_upper[row_idx] = Inf
        else
            error("Unsupported constraint set: $s")
        end
    end

    # Process variable constraints to set bounds and integrality
    for con_ref in constraints
        con = JuMP.constraint_object(con_ref)
        if isa(con.func, JuMP.VariableRef)
            j = var2Idx[con.func]
            s = con.set
            if s isa MathOptInterface.EqualTo{Float64}
                col_lower[j] = s.value
                col_upper[j] = s.value
            elseif s isa MathOptInterface.Interval{Float64}
                col_lower[j] = s.lower
                col_upper[j] = s.upper
            elseif s isa MathOptInterface.LessThan{Float64}
                col_upper[j] = s.upper
            elseif s isa MathOptInterface.GreaterThan{Float64}
                col_lower[j] = s.lower
            elseif s isa MathOptInterface.Integer
                isInteger[j] = true
            elseif s isa MathOptInterface.ZeroOne
                isInteger[j] = true
                col_lower[j] = 0.0
                col_upper[j] = 1.0
            else
                error("Unsupported constraint set: $s")
            end
        end
    end

    # Build the sparse matrix A.
    A = sparse(row_inds, col_inds, vals, number_rows, number_cols)

    return GenericLP(number_cols, 
        number_rows, 
        obj, 
        offset, 
        A, 
        row_lower, row_upper, 
        col_lower, col_upper,
        col_names,
        row_names)
end 


function generateGenericLP(lp::GenericLP)
    mbp = MultiblockProblem() 

    block_x = BlockVariable("x")
    block_x.f = AffineFunction(lp.obj, lp.offset)
    block_x.g = IndicatorBox(lp.col_lower, lp.col_upper)
    # block_x.g = IndicatorBinary()
    block_x.val = proximalOracle(block_x.g, rand(lp.number_cols))
    addBlockVariable!(mbp, block_x)

    block_s = BlockVariable("s") 
    block_s.g = IndicatorBox(lp.row_lower, lp.row_upper)
    block_s.val = proximalOracle(block_s.g, zeros(lp.number_rows))
    addBlockVariable!(mbp, block_s)

    constr = BlockConstraint(1) 
    addBlockMappingToConstraint!(constr, "x", LinearMappingMatrix(lp.A))
    addBlockMappingToConstraint!(constr, "s", LinearMappingIdentity(-1.0))
    constr.rhs = zeros(lp.number_rows)
    addBlockConstraint!(mbp, constr)

    return mbp 
end 


function heuristic_coclustering(A::SparseMatrixCSC{<:Real,Int}, k::Int; iters::Int=10, rng::AbstractRNG=MersenneTwister(42), forceSplitSingleBlock::Bool=true)
    m, n = size(A)
    k = max(1, k)
    # Initialize column blocks in a round-robin (deterministic) way
    col_block = [mod(j-1, k) + 1 for j in 1:n]
    # Shuffle for better starting diversity
    shuffle!(rng, col_block)

    row_block = ones(Int, m)

    # Precompute row -> list of (col) indices using A in CSC: build CSR-like access
    row_to_cols = Vector{Vector{Int}}(undef, m)
    for i in 1:m
        row_to_cols[i] = Int[]
    end
    @inbounds for j in 1:n
        for ptr in A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[ptr]
            push!(row_to_cols[i], j)
        end
    end

    # Precompute col -> list of (row) indices from CSC
    col_to_rows = Vector{UnitRange{Int}}(undef, n)
    @inbounds for j in 1:n
        col_to_rows[j] = A.colptr[j]:(A.colptr[j+1]-1)
    end

    # Main alternation
    counts_row = zeros(Int, k)
    counts_col = zeros(Int, k)
    for _ in 1:iters
        # Assign rows to the block that covers most of its nonzeros
        @inbounds for i in 1:m
            fill!(counts_row, 0)
            cols_i = row_to_cols[i]
            for j in cols_i
                b = col_block[j]
                counts_row[b] += 1
            end
            # if a row is empty, keep previous or assign 1
            if isempty(cols_i)
                row_block[i] = 1
            else
                # tie-breaker: smallest block id with max count
                maxc = -1
                argb = 1
                for b in 1:k
                    if counts_row[b] > maxc
                        maxc = counts_row[b]
                        argb = b
                    end
                end
                row_block[i] = argb
            end
        end

        # Reassign columns to the block most represented among its incident rows
        @inbounds for j in 1:n
            fill!(counts_col, 0)
            for ptr in col_to_rows[j]
                i = A.rowval[ptr]
                b = row_block[i]
                counts_col[b] += 1
            end
            # if a column is empty, keep previous or assign 1
            if A.colptr[j] == A.colptr[j+1]
                col_block[j] = 1
            else
                maxc = -1
                argb = 1
                for b in 1:k
                    if counts_col[b] > maxc
                        maxc = counts_col[b]
                        argb = b
                    end
                end
                col_block[j] = argb
            end
        end
    end

    # Fallback: if all columns collapsed into a single block, enforce a multi-block split
    k_cols = maximum(col_block)
    if forceSplitSingleBlock && k_cols == 1 && n >= 2
        new_k = min(max(2, k), n)
        for j in 1:n
            col_block[j] = mod(j-1, new_k) + 1
        end
        tmp_counts = zeros(Int, new_k)
        stamp_block = 0
        row_block .= 1
        @inbounds for i in 1:m
            fill!(tmp_counts, 0)
            cols_i = row_to_cols[i]
            for j in cols_i
                tmp_counts[col_block[j]] += 1
            end
            if isempty(cols_i)
                row_block[i] = 1
            else
                maxc = -1
                argb = 1
                for b in 1:new_k
                    if tmp_counts[b] > maxc
                        maxc = tmp_counts[b]
                        argb = b
                    end
                end
                row_block[i] = argb
            end
        end
    end
    return row_block, col_block
end


struct CoClusterLayout
    row_block::Vector{Int}
    col_block::Vector{Int}
    cols_in_block::Vector{Vector{Int}}
    rows_touched_by_block::Vector{Vector{Int}}
    row_to_cols::Vector{Vector{Int}}
    group_rows::Vector{Vector{Int}}
    group_blocks::Vector{Vector{Int}}
    k_cols::Int
end


function buildCoClusterLayout(lp::GenericLP, row_block::Vector{Int}, col_block::Vector{Int}; k::Int)
    m = lp.number_rows
    n = lp.number_cols
    k_cols = maximum(col_block)
    cols_in_block = [Int[] for _ in 1:k_cols]
    for j in 1:n
        push!(cols_in_block[col_block[j]], j)
    end

    rows_touched_by_block = [Int[] for _ in 1:k_cols]
    visited = zeros(Int, m)
    stamp = 0
    A = lp.A
    @inbounds for b in 1:k_cols
        Jb = cols_in_block[b]
        isempty(Jb) && continue
        stamp += 1
        for j in Jb
            for ptr in A.colptr[j]:(A.colptr[j+1]-1)
                i = A.rowval[ptr]
                if visited[i] != stamp
                    visited[i] = stamp
                    push!(rows_touched_by_block[b], i)
                end
            end
        end
    end

    row_to_cols = [Int[] for _ in 1:m]
    @inbounds for j in 1:n
        for ptr in A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[ptr]
            push!(row_to_cols[i], j)
        end
    end

    row_blocks = Vector{Vector{Int}}(undef, m)
    seen_block = zeros(Int, k_cols)
    stamp_block = 0
    @inbounds for i in 1:m
        cols_i = row_to_cols[i]
        stamp_block += 1
        bi = Int[]
        for j in cols_i
            b = col_block[j]
            if seen_block[b] != stamp_block
                seen_block[b] = stamp_block
                push!(bi, b)
            end
        end
        sort!(bi)
        row_blocks[i] = bi
    end

    sig_to_rows = Dict{Tuple{Vararg{Int}}, Vector{Int}}()
    single_block_rows = Dict{Int, Vector{Int}}()
    for b in 1:k_cols
        single_block_rows[b] = Int[]
    end
    for i in 1:m
        bi = row_blocks[i]
        if length(bi) >= 2
            key = Tuple(bi)
            if haskey(sig_to_rows, key)
                push!(sig_to_rows[key], i)
            else
                sig_to_rows[key] = [i]
            end
        elseif length(bi) == 1
            push!(single_block_rows[bi[1]], i)
        end
    end

    group_rows = Vector{Vector{Int}}()
    group_blocks = Vector{Vector{Int}}()
    for (sig, rows) in sig_to_rows
        push!(group_rows, rows)
        push!(group_blocks, collect(sig))
    end

    find_group_with_block = function(b::Int)
        for idx in eachindex(group_blocks)
            blocks = group_blocks[idx]
            for bb in blocks
                if bb == b
                    return idx
                end
            end
        end
        return 0
    end

    for b in 1:k_cols
        rows_b = single_block_rows[b]
        isempty(rows_b) && continue
        gi = find_group_with_block(b)
        if gi != 0
            append!(group_rows[gi], rows_b)
            empty!(rows_b)
        end
    end

    all_groups_empty = isempty(group_rows)
    if all_groups_empty
        buckets = [(b, single_block_rows[b]) for b in 1:k_cols if !isempty(single_block_rows[b])]
        while length(buckets) >= 2
            (b1, rows1) = popfirst!(buckets)
            (b2, rows2) = popfirst!(buckets)
            push!(group_rows, vcat(rows1, rows2))
            push!(group_blocks, sort([b1, b2]))
        end
        if !isempty(buckets)
            (brem, rowsrem) = buckets[1]
            if isempty(group_rows)
                push!(group_rows, copy(rowsrem))
                push!(group_blocks, [brem])
            else
                gi_max = argmax(length.(group_rows))
                append!(group_rows[gi_max], rowsrem)
            end
        end
        for b in 1:k_cols
            empty!(single_block_rows[b])
        end
    else
        remaining = Int[]
        for b in 1:k_cols
            append!(remaining, single_block_rows[b])
            empty!(single_block_rows[b])
        end
        if !isempty(remaining)
            gi_max = argmax(length.(group_rows))
            append!(group_rows[gi_max], remaining)
        end
    end

    if length(group_rows) > k
        while length(group_rows) > k
            sizes = [length(r) for r in group_rows]
            idxs = sortperm(sizes)[1:2]
            i1, i2 = idxs[1], idxs[2]
            rows_merged = vcat(group_rows[i1], group_rows[i2])
            blocks_merged = sort!(unique(vcat(group_blocks[i1], group_blocks[i2])))
            hi, lo = max(i1, i2), min(i1, i2)
            deleteat!(group_rows, hi)
            deleteat!(group_rows, lo)
            deleteat!(group_blocks, hi)
            deleteat!(group_blocks, lo)
            push!(group_rows, rows_merged)
            push!(group_blocks, blocks_merged)
        end
    end

    # Recompute block signatures from the final grouped rows to reflect any merges
    seen_block = zeros(Int, k_cols)
    stamp_final = 0
    new_group_blocks = Vector{Vector{Int}}(undef, length(group_rows))
    for idx in 1:length(group_rows)
        stamp_final += 1
        blocks = Int[]
        for i in group_rows[idx]
            for j in row_to_cols[i]
                b = col_block[j]
                if seen_block[b] != stamp_final
                    seen_block[b] = stamp_final
                    push!(blocks, b)
                end
            end
        end
        sort!(blocks)
        new_group_blocks[idx] = blocks
    end

    return CoClusterLayout(row_block, col_block, cols_in_block, rows_touched_by_block, row_to_cols, group_rows, new_group_blocks, k_cols)
end


function generateGenericLPWithCoClustering(lp::GenericLP; k::Int=6, iters::Int=5, rng::AbstractRNG=MersenneTwister(42), forceSplitSingleBlock::Bool=true, return_layout::Bool=false)
    # 1) Co-cluster rows/cols on the sparsity pattern of A
    row_block, col_block = heuristic_coclustering(lp.A, k; iters=iters, rng=rng, forceSplitSingleBlock=forceSplitSingleBlock)

    m = lp.number_rows
    layout = buildCoClusterLayout(lp, row_block, col_block; k=k)
    cols_in_block = layout.cols_in_block
    rows_touched_by_block = layout.rows_touched_by_block
    row_to_cols = layout.row_to_cols
    group_rows = layout.group_rows
    k_cols = layout.k_cols
    A = lp.A

    # 6) Build multiblock LP with x-blocks per column cluster
    mbp = MultiblockProblem()

    # x-blocks (IndicatorBox for LP), add objective offset exactly once
    offset_added = false
    for b in 1:k_cols
        Jb = cols_in_block[b]
        if isempty(Jb)
            continue
        end
        name = "x_$b"
        block_x = BlockVariable(name)
        if !offset_added
            block_x.f = AffineFunction(lp.obj[Jb], lp.offset)
            offset_added = true
        else
            block_x.f = AffineFunction(lp.obj[Jb], 0.0)
        end
        block_x.g = IndicatorBox(lp.col_lower[Jb], lp.col_upper[Jb])
        block_x.val = proximalOracle(block_x.g, zeros(length(Jb)))
        addBlockVariable!(mbp, block_x)
    end
    if !offset_added && lp.offset != 0.0
        name = "x_1"
        block_x = BlockVariable(name)
        block_x.f = AffineFunction(zeros(0), lp.offset)
        block_x.g = IndicatorBox(zeros(0), zeros(0))
        block_x.val = zeros(0)
        addBlockVariable!(mbp, block_x)
    end

    # Helper to extract submatrix A[R, J]
    get_submatrix = function(R::Vector{Int}, J::Vector{Int})
        if isempty(R) || isempty(J)
            return spzeros(Float64, length(R), length(J))
        end
        return A[R, J]
    end

    # Build constraint blocks; add slacks only for groups containing inequalities
    row_mark = zeros(Int, m)
    mark_id = 0
    for r in 1:length(group_rows)
        Rr = group_rows[r]
        isempty(Rr) && continue
        mark_id += 1
        @inbounds for i in Rr
            row_mark[i] = mark_id
        end

        all_eq = all(lp.row_lower[i] == lp.row_upper[i] for i in Rr)
        constr = BlockConstraint(r)

        candidate_blocks = Vector{Int}()
        seen_b = zeros(Int, k_cols)
        mark_b = r
        @inbounds for i in Rr
            for j in row_to_cols[i]
                b = col_block[j]
                if seen_b[b] != mark_b
                    seen_b[b] = mark_b
                    push!(candidate_blocks, b)
                end
            end
        end

        for b in candidate_blocks
            Jb = cols_in_block[b]
            isempty(Jb) && continue
            has_intersection = false
            @inbounds for i in rows_touched_by_block[b]
                if row_mark[i] == mark_id
                    has_intersection = true
                    break
                end
            end
            has_intersection || continue
            subA = get_submatrix(Rr, Jb)
            addBlockMappingToConstraint!(constr, "x_$b", LinearMappingMatrix(subA))
        end

        if all_eq
            constr.rhs = lp.row_lower[Rr]
        else
            sname = "s_grp_$r"
            block_s = BlockVariable(sname)
            block_s.g = IndicatorBox(lp.row_lower[Rr], lp.row_upper[Rr])
            block_s.val = proximalOracle(block_s.g, zeros(length(Rr)))
            addBlockVariable!(mbp, block_s)
            addBlockMappingToConstraint!(constr, sname, LinearMappingIdentity(-1.0))
            constr.rhs = zeros(length(Rr))
        end
        addBlockConstraint!(mbp, constr)
    end

    return return_layout ? (mbp, layout) : mbp
end 


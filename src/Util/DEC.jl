# DEC.jl - Parser for DEC files (block decomposition for MILP)

struct DEC
    presolved::Bool
    numberBlocks::Int
    mapBlock2Rows::Dict{Int, Vector{Int}}
    mapBlock2Columns::Dict{Int, Vector{Int}}
end

"""
    parseDEC(A::SparseMatrixCSC, rowNames::Vector{String}, filename::AbstractString; logLevel::Int=1) -> Union{DEC, Nothing}

Parse a DEC file and compute block-to-row/column mappings based on the MILP constraint matrix `A`.

Arguments
- `A::SparseMatrixCSC`: Constraint matrix of the presolved MILP with size (num_rows, num_cols)
- `rowNames::Vector{String}`: Row names of `A` in order; must have length `size(A, 1)`
- `filename::AbstractString`: Path to the DEC file

Returns
- `DEC` on success; `nothing` on failure (mismatched sizes, IO errors, malformed file)
"""
function parseDEC(A::SparseMatrixCSC, 
    rowNames::Vector{String}, 
    filename::AbstractString; 
    logLevel::Int=1)
    try
        num_rows, num_cols = size(A)
        if length(rowNames) != num_rows
            @PDMOInfo logLevel "DEC: row names != num_row in presolved model"
            return nothing
        end

        # Build a fast row -> column incidence using the transpose in CSC
        AT = sparse(transpose(A))

        mapRowNameToIndex = Dict{String, Int}()
        for r in 1:length(rowNames)
            mapRowNameToIndex[rowNames[r]] = r
        end

        presolved = false
        numberBlocks = 0
        mapBlock2Rows = Dict{Int, Vector{Int}}()
        mapBlock2ColumnsSet = Dict{Int, Set{Int}}()

        blockId = -1

        open(filename, "r") do io
            if eof(io)
                error("DEC: File is empty")
            end

            for line in eachline(io)
                s = strip(line)
                isempty(s) && continue
                words = split(s)
                isempty(words) && continue
                word = words[1]

                if startswith(word, "\\")
                    continue
                elseif word == "PRESOLVED"
                    if eof(io) == false
                        status_line = strip(readline(io))
                        if !isempty(status_line)
                            v = lowercase(strip(status_line))
                            presolved = v in ("1", "true", "t", "yes", "y")
                        end
                    end
                elseif word == "NBLOCKS"
                    if eof(io) == false
                        nblocks_line = strip(readline(io))
                        if !isempty(nblocks_line)
                            numberBlocks = parse(Int, split(nblocks_line)[1])
                        end
                    end
                elseif word == "BLOCK"
                    if length(words) >= 2
                        blockId = parse(Int, words[2])
                    else
                        # If block id is on next token/line, try to read next non-empty line
                        got = false
                        while eof(io) == false
                            nxt = strip(readline(io))
                            isempty(nxt) && continue
                            blockId = parse(Int, split(nxt)[1])
                            got = true
                            break
                        end
                        got || (blockId = -1)
                    end
                    continue
                elseif word == "MASTERCONSS"
                    blockId = 0
                    continue
                end

                if blockId >= 0
                    rowname = word
                    row = get(mapRowNameToIndex, rowname, 0)
                    if row == 0
                        @PDMOInfo logLevel "DEC: Unknown row name in DEC file" rowname=rowname
                        continue
                    end

                    v = get!(mapBlock2Rows, blockId, Int[])
                    push!(v, row)
                    @PDMODebug logLevel "DEC: row" row=row block=blockId

                    # Columns of A with nonzero in this row correspond to indices in AT column 'row'
                    beg = AT.colptr[row]
                    endp = AT.colptr[row+1] - 1
                    totalCol = max(endp - beg + 1, 0)
                    colsset = get!(mapBlock2ColumnsSet, blockId, Set{Int}())
                    for i in beg:endp
                        col = AT.rowval[i]
                        @PDMODebug logLevel "DEC: col" col=col idx=(i - beg + 1) total=totalCol block=blockId
                        push!(colsset, col)
                    end
                end
            end
        end

        # Convert column sets to deterministically ordered vectors
        mapBlock2Columns = Dict{Int, Vector{Int}}(bid => sort!(collect(cols)) for (bid, cols) in mapBlock2ColumnsSet)

        # Ensure block IDs are consecutive and increasing (keep 0 if present)
        all_ids = sort(collect(union(keys(mapBlock2Rows), keys(mapBlock2Columns))))
        nonzero_ids = [bid for bid in all_ids if bid != 0]
        expected_nonzero = collect(1:length(nonzero_ids))
        needs_remap = (nonzero_ids != expected_nonzero)
        if needs_remap
            # Build remap: keep 0 -> 0, map sorted positive IDs to 1..n in order
            remap = Dict{Int, Int}()
            for (new_id, old_id) in enumerate(nonzero_ids)
                remap[old_id] = new_id
            end

            newRows = Dict{Int, Vector{Int}}()
            for (bid, rows) in mapBlock2Rows
                new_id = bid == 0 ? 0 : remap[bid]
                newRows[new_id] = get(newRows, new_id, Int[])
                append!(newRows[new_id], rows)
            end
            mapBlock2Rows = newRows

            newCols = Dict{Int, Vector{Int}}()
            for (bid, cols) in mapBlock2Columns
                new_id = bid == 0 ? 0 : remap[bid]
                v = get(newCols, new_id, Int[])
                append!(v, cols)
                newCols[new_id] = v
            end
            mapBlock2Columns = newCols

            numberBlocks = length(nonzero_ids)
        end

        # If NBLOCKS is missing/malformed, infer from observed block IDs.
        # If NBLOCKS disagrees with observed data, trust observed data and report.
        observedBlocks = length(nonzero_ids)
        if numberBlocks <= 0
            numberBlocks = observedBlocks
        elseif numberBlocks != observedBlocks
            @PDMOInfo logLevel "DEC: NBLOCKS mismatch; using observed block count" declared=numberBlocks observed=observedBlocks
            numberBlocks = observedBlocks
        end

        # Ensure deterministic, duplicate-free row/column lists for each block
        for (bid, rows) in mapBlock2Rows
            mapBlock2Rows[bid] = sort!(unique(rows))
        end
        for (bid, cols) in mapBlock2Columns
            mapBlock2Columns[bid] = sort!(unique(cols))
        end

        # Summary logging (non-distributed build equivalent)
        masterRows = length(get(mapBlock2Rows, 0, Int[]))
        masterCols = length(get(mapBlock2Columns, 0, Int[]))

        maxBlockRows = 0
        minBlockRows = num_rows
        maxBlockCols = 0
        minBlockCols = num_cols

        for bid in 1:numberBlocks
            rowSize = length(get(mapBlock2Rows, bid, Int[]))
            colSize = length(get(mapBlock2Columns, bid, Int[]))
            maxBlockCols = max(maxBlockCols, colSize)
            minBlockCols = min(minBlockCols, colSize)
            maxBlockRows = max(maxBlockRows, rowSize)
            minBlockRows = min(minBlockRows, rowSize)
        end

        @PDMOInfo logLevel "DEC contains coupling rows and columns" masterRows=masterRows masterCols=masterCols
        @PDMOInfo logLevel "DEC blocks stats" nBlocks=numberBlocks rowRange=(minBlockRows, maxBlockRows) colRange=(minBlockCols, maxBlockCols)

        return DEC(presolved, numberBlocks, mapBlock2Rows, mapBlock2Columns)
    catch err
        @PDMOInfo 1 "DEC: parsing error" error=err
        return nothing
    end
end



"""
    MultiblockProblem

A container for a multiblock optimization problem. This structure maintains a collection of block variables
and a collection of block constraints.
 
# Fields
- `blocks::Vector{BlockVariable}`: A vector of block variables.
- `constraints::Vector{BlockConstraint}`: A vector of block constraints.
 
A default constructor is provided that initializes both collections as empty.
"""
mutable struct MultiblockProblem
    couplingFunction::Union{AbstractMultiblockFunction, Nothing}
    blocks::Vector{BlockVariable}
    constraints::Vector{BlockConstraint}
    MultiblockProblem() = new(nothing, Vector{BlockVariable}(), Vector{BlockConstraint}())
end 

"""
    addBlockVariable!(mbp::MultiblockProblem, block::BlockVariable)

Add a block variable to the multiblock problem. If the block has a default ID, a new unique ID is assigned.
Otherwise, the ID is checked for uniqueness.
"""
function addBlockVariable!(mbp::MultiblockProblem, block::BlockVariable)
    if isa(block.id, String) && isempty(block.id) # the block has a default ID = ""
        baseId = "Block"
        newId = baseId * string(length(mbp.blocks))
        while any(b -> b.id == newId, mbp.blocks)
            newId = baseId * string(parse(Int, match(r"\d+", newId).match) + 1)
        end
        block.id = newId
        @warn("MultiblockProblem: a block with default ID is added; assigned ID = $newId")
    else # otherwise, check if the ID already exists
         for b in mbp.blocks
            if b.id == block.id
                error("MultiblockProblem: Block with ID '$(block.id)' already exists.")
            end
        end
    end
    push!(mbp.blocks, block)
    return block.id
end 

"""
    addBlockConstraint!(mbp::MultiblockProblem, constraint::BlockConstraint)

Add a block constraint to the multiblock problem. If the constraint has a default ID, a new unique ID is assigned.
Otherwise, the ID is checked for uniqueness.
"""
function addBlockConstraint!(mbp::MultiblockProblem, constraint::BlockConstraint)
    if isa(constraint.id, String) && isempty(constraint.id) # the constraint has a default ID = ""
        baseId = "Constraint"
        newId = baseId * string(length(mbp.constraints))
        while any(c -> c.id == newId, mbp.constraints)
            newId = baseId * string(parse(Int, match(r"\d+", newId).match) + 1)
        end
        constraint.id = newId
        @warn("MultiblockProblem: a constraint with default ID is added; assigned ID = $newId")
    else # otherwise, check if the ID already exists
         for c in mbp.constraints
            if c.id == constraint.id
                error("MultiblockProblem: Constraint with ID '$(constraint.id)' already exists.")
            end
        end
    end
    push!(mbp.constraints, constraint)
    return constraint.id
end 

function checkCouplingFunctionValidity(f::AbstractMultiblockFunction, x::Vector{NumericVariable})
    
    if isSmooth(f) == false 
        println("checkCouplingFunctionValidity: coupling function is not smooth")
        return false 
    end 

    if isConvex(f) == false 
        # println("checkCouplingFunctionValidity: coupling function is not convex")
    end 

    if isSupportedByJuMP(f) == false 
        # println("checkCouplingFunctionValidity: coupling function is not supported by JuMP") 
    end 

    numberBlocks = length(x)
    if numberBlocks != getNumberOfBlocks(f)
        println("checkCouplingFunctionValidity: number of blocks in coupling function does not match number of blocks in problem")
        return false 
    end 

    try 
        validateBlockDimensions(f, x)
    catch err 
        println("checkCouplingFunctionValidity: error encountered while validating block dimensions: error = $err")
        return false 
    end 

    try 
        val = f(x)
    catch err 
        println("checkCouplingFunctionValidity: error encountered while evaluating coupling function: error = $err")
        return false 
    end 

    try 
        grad1 = gradientOracle(f, x)
        grad2 = NumericVariable[similar(xi, Float64) for xi in grad1]
        gradientOracle!(grad2, f, x)
        for i in 1:numberBlocks 
            if norm(grad1[i] - grad2[i], Inf) > ZeroTolerance 
                println("checkCouplingFunctionValidity: gradient oracle is not consistent")
                return false 
            end 
        end 
    catch err 
        println("checkCouplingFunctionValidity: error encountered while evaluating gradient oracle: error = $err")
        return false 
    end 

    try 
        for i in 1:numberBlocks 
            partialGrad1 = partialGradientOracle(f, x, i)
            partialGrad2 = similar(partialGrad1)
            partialGradientOracle!(partialGrad2, f, x, i)
            if norm(partialGrad1 - partialGrad2, Inf) > ZeroTolerance 
                println("checkCouplingFunctionValidity: partial gradient oracle is not consistent")
                return false 
            end 
        end 
    catch err 
        println("checkCouplingFunctionValidity: error encountered while evaluating partial gradient oracle: error = $err")  
        return false 
    end 

    return true 
end 
"""
    checkMultiblockProblemValidity(mbp::MultiblockProblem; addWrapper::Bool=true)

Check whether the given multiblock problem is valid.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to check.
- `addWrapper::Bool=true`: Flag to add wrapper for scalar input functions.

# Returns
- `Bool`: `true` if the problem is valid, `false` otherwise.

# Implementation Details
The function checks the validity of each block variable and constraint. It also attempts
to evaluate the primal residuals of each constraint with the initial solution to ensure
the problem is well-formed.
"""
function checkMultiblockProblemValidity(mbp::MultiblockProblem; addWrapper::Bool=true)
    for block in mbp.blocks 
        if checkBlockVariableValidity(block; addScalarFunctionWrapper=addWrapper) == false 
            return false
        end 
    end 

    if mbp.couplingFunction != nothing 
        x = NumericVariable[block.val for block in mbp.blocks]
        if checkCouplingFunctionValidity(mbp.couplingFunction, x) == false 
            return false 
        end 
    end 

    # Create an initial solution: a vector of the x fields from each block.
    initialSol = Dict{BlockID, NumericVariable}(block.id => block.val for block in mbp.blocks)

    # Validate each block constraint.
    for constr in mbp.constraints 
        if isa(constr.rhs, Number) && addWrapper == true 
            constr.rhs = Float64[constr.rhs]
        end 

        if checkBlockConstraintValidity(constr) == false 
            return false 
        end 

        try 
            pres = blockConstraintViolation(constr, initialSol)
        catch err  
            println("MultiblockProblem: error encountered while evaluating primal residuals of constraint $(constr.id): error = $err")
            return false 
        end
    end 

    return true 
end

"""
    checkMultiblockProblemFeasibility(mbp::MultiblockProblem)

Check the feasibility of the current solution in the multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to check.

# Returns
- `Float64`: The L2 norm of the constraint violations.
- `Float64`: The infinity norm of the constraint violations.

# Implementation Details
The function computes the constraint violations for each constraint using the current
values in the block variables, and returns both the L2 and infinity norms of these violations.
"""
function checkMultiblockProblemFeasibility(mbp::MultiblockProblem, primalSol::Dict{BlockID, NumericVariable})
    presL2 = 0.0
    presLInf = 0.0
    for constr in mbp.constraints 
        res = blockConstraintViolation(constr, primalSol)
        presL2 += dot(res, res)
        presLInf = max(presLInf, norm(res, Inf))
    end 
    presL2 = sqrt(presL2)
    return presL2, presLInf
end 

function checkMultiblockProblemFeasibility(mbp::MultiblockProblem)
    sol = Dict{BlockID, NumericVariable}(block.id => block.val for block in mbp.blocks)
    return checkMultiblockProblemFeasibility(mbp, sol)
end 


"""
    summary(mbp::MultiblockProblem)

Print a summary of the multiblock problem.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to summarize.

# Output
Prints information about the problem including:
- Number of block variables
- Number of block constraints
- Range of number of blocks involved in constraints
"""
function summary(mbp::MultiblockProblem, logLevel::Int64=1)
    if logLevel < 1
        return 
    end 
    @PDMOInfo logLevel "Summary of Multiblock Problem: "
    println("    Number of block variables   = $(length(mbp.blocks))")
    println("    Number of block constraints = $(length(mbp.constraints))")
    println("    Has coupling function       = $(mbp.couplingFunction != nothing)")
   
    
    min_row_blocks = length(mbp.blocks)
    max_row_blocks = 0
    for constr in mbp.constraints 
        number_blocks = length(constr.involvedBlocks)
        if number_blocks < min_row_blocks
            min_row_blocks = number_blocks
        end 
        if number_blocks > max_row_blocks 
            max_row_blocks = number_blocks
        end 
    end     
    println("    Range of num. blocks in row = [$min_row_blocks, $max_row_blocks]")
end 


""" 
    checkCompositeProblemValidity!(mbp::MultiblockProblem)

Check whether `mbp` matches the composite structure with `(p+1)` blocks:

    min sum_{i=1}^p (f_i(x_i) + g_i(x_i)) + g_{p+1}(x_{p+1})
    s.t. A1x1 + A2x2 + ... + Apxp - x_{p+1} = 0

If a valid proximal-only block `x_{p+1}` is identified, it is moved to the end of
`mbp.blocks`.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to check.

# Returns
- `Bool`: `true` if the problem is a valid composite form, `false` otherwise.

"""
function checkCompositeProblemValidity!(mbp::MultiblockProblem, logLevel::Int64=1)

    if detectConsensusStructure(mbp)
        @PDMOInfo logLevel "checkCompositeProblemValidity: consensus structure detected and reformulated"
        return true 
    end 

    if length(mbp.constraints) != 1
        @PDMOError logLevel "checkCompositeProblemValidity: number of constraints must be 1"
        return false 
    end 

    constraint = mbp.constraints[1]

    if norm(constraint.rhs, Inf) > ZeroTolerance 
        @PDMOError logLevel "checkCompositeProblemValidity: right-hand side of the constraint must be 0"
        return false 
    end 

    potentialProximalOnlyBlocks = BlockID[]
    for block in mbp.blocks
        mapping = constraint.mappings[block.id] 
        if isa(mapping, LinearMappingIdentity) && mapping.coe == -1.0 && isa(block.f, Zero) 
            push!(potentialProximalOnlyBlocks, block.id)
        end 

        if isSmooth(block.f) == false || 
            # isConvex(block.f) == false ||
            isProximal(block.g) == false 
            # isConvex(block.g) == false
            @PDMOError logLevel "checkCompositeProblemValidity: block $(block.id) is not valid for AdaPDM."
            return false 
        end 
    end 

    if isempty(potentialProximalOnlyBlocks)
        println("checkCompositeProblemValidity: no proximal-only block found")
        return false 
    end 

    proximalOnlyBlockID = potentialProximalOnlyBlocks[1] 
   
    idxToMove = findfirst(block -> block.id == proximalOnlyBlockID, mbp.blocks)
    if idxToMove != length(mbp.blocks)
        blockToMove = mbp.blocks[idxToMove]
        deleteat!(mbp.blocks, idxToMove)
        push!(mbp.blocks, blockToMove)
    end 
    
    return true 
end 


function detectConsensusStructure(mbp::MultiblockProblem)
    # Consensus structure (classic):
    #   constraints:  A_iz - x_i = 0  (or z - x_i = 0), for i = 1..K
    #   blocks:       {z} plus {x_i} (each x_i appears in exactly one constraint)
    #
    # We detect this pattern and rewrite it into a 2-block composite form with a single stacked constraint:
    #   [A_1z; A_2z; ...; A_Kz] - [x_1; ...; x_K] = 0

    K = length(mbp.constraints)
    if length(mbp.blocks) != K + 1
        return false
    end

    # Fast lookup by block id.
    blockById = Dict{BlockID, BlockVariable}(b.id => b for b in mbp.blocks)

    # Count how many constraints each block participates in, and validate each constraint is 2-block.
    appearCount = Dict{BlockID, Int}(b.id => 0 for b in mbp.blocks)
    for c in mbp.constraints
        if length(c.involvedBlocks) != 2
            return false
        end
        for bid in c.involvedBlocks
            appearCount[bid] = get(appearCount, bid, 0) + 1
        end
    end

    # Identify the unique consensus block: it appears in all constraints.
    consensusIds = BlockID[bid for (bid, cnt) in appearCount if cnt == K]
    if length(consensusIds) != 1
        return false
    end
    consensusId = consensusIds[1]

    # All other blocks must appear in exactly one constraint, and must be proximal-only (f=0, g proximal).
    for b in mbp.blocks
        if b.id == consensusId
            continue
        end
        if get(appearCount, b.id, 0) != 1
            return false
        end
        if isa(b.f, Zero) == false || isProximal(b.g) == false
            return false
        end
    end

    consensusBlock = blockById[consensusId]
    if isa(consensusBlock.val, Number)
        # This reformulation stacks along the first dimension; skip scalar consensus variables.
        return false
    end

    # Helper: given a 2-block constraint, return the "local" block id (the one that isn't the consensus block).
    otherBlockId(c::BlockConstraint) = (c.involvedBlocks[1] == consensusId) ? c.involvedBlocks[2] : c.involvedBlocks[1]

    # Extract local blocks in constraint order.
    #
    # We assume each consensus constraint can be normalized to the form:
    #     Aᵢ(z) - xᵢ = 0
    # where:
    # - `z` is the consensus block variable,
    # - `xᵢ` is a local block variable,
    # - `Aᵢ` is the mapping on the consensus block (can be general),
    # - the local-side mapping is either `-Identity` (already in the desired form), OR `+Identity`
    #   in which case we only accept the constraint when the consensus-side mapping is `-Identity`
    #   (i.e., the constraint is `-z + xᵢ = 0`, equivalent to `z - xᵢ = 0` after multiplying by -1).
    #
    # If a constraint is NOT already of the form `Aᵢ(z) - xᵢ = 0`, we only allow flipping the sign
    # when the consensus-side mapping is an identity mapping (i.e., `xᵢ - z = 0` can be rewritten as
    # `z - xᵢ = 0`). Otherwise return false.
    localIds = BlockID[]
    consensusMappings = Vector{AbstractMapping}(undef, K) # stores Aᵢ after normalization
    sizeAlongFirstDimension = Vector{Int64}(undef, K)
    tailShape = nothing

    for (k, c) in enumerate(mbp.constraints)
        localId = otherBlockId(c)
        maps = c.mappings

        # RHS must be zero.
        rhs = c.rhs
        if isa(rhs, Number)
            if abs(rhs) > ZeroTolerance
                return false
            end
        else
            if norm(rhs, Inf) > ZeroTolerance
                return false
            end
        end

        # Local variable (xᵢ) shape determines the stacking shape.
        localBlock = blockById[localId]
        if isa(localBlock.val, Number)
            return false
        end
        sX = size(localBlock.val)
        if tailShape === nothing
            tailShape = Base.tail(sX)
        elseif Base.tail(sX) != tailShape
            return false
        end
        sizeAlongFirstDimension[k] = sX[1]

        # Normalize mapping orientation to Aᵢ(z) - xᵢ = 0.
        mLocal = maps[localId]
        mCons = maps[consensusId]

        A = nothing
        if isa(mLocal, LinearMappingIdentity) && mLocal.coe == -1.0
            # Already in the desired form: Aᵢ(z) - xᵢ = 0
            A = mCons
        else
            # If not in desired form, only allow flipping when consensus mapping is identity:
            # xᵢ - z = 0  <=>  z - xᵢ = 0
            if isa(mCons, LinearMappingIdentity) &&
                isa(mLocal, LinearMappingIdentity) &&
                mCons.coe == -1.0 && mLocal.coe == 1.0
                A = LinearMappingIdentity(1.0)
            else
                return false
            end
        end

        # Dimension check: Aᵢ(z) must match xᵢ in shape.
        yAz = try
            A(consensusBlock.val)
        catch
            return false
        end
        if isa(yAz, Number) || size(yAz) != sX
            return false
        end

        consensusMappings[k] = A

        push!(localIds, localId)
    end

    # Build stacked initial value, and build the stacked proximal function.
    totalFirstDim = sum(sizeAlongFirstDimension)
    outShape = (totalFirstDim, tailShape...)
    stackedVal = length(outShape) <= 2 ? spzeros(outShape) : zeros(outShape)
    proximalFunctions = Vector{AbstractFunction}(undef, K)

    startIdx = 1
    for (k, bid) in enumerate(localIds)
        localBlock = blockById[bid]
        proximalFunctions[k] = localBlock.g

        blockLen = sizeAlongFirstDimension[k]
        endIdx = startIdx + blockLen - 1
        if ndims(stackedVal) == 1
            @views stackedVal[startIdx:endIdx] .= localBlock.val
        else
            tail = ntuple(_ -> Colon(), ndims(stackedVal) - 1)
            @views stackedVal[startIdx:endIdx, tail...] .= localBlock.val
        end
        startIdx = endIdx + 1
    end

    stackedBlock = BlockVariable("stacked")
    stackedBlock.f = Zero()
    stackedBlock.g = StackingProximalFunctions(proximalFunctions, sizeAlongFirstDimension)
    stackedBlock.val = stackedVal

    # Build the consensus-side stacked mapping:
    # - If all consensus-side mappings are the same scaled identity, use `LinearMappingIdentityStacking`.
    # - Otherwise, use `LinearMappingStacking` to stack their outputs.
    consensusMap = begin
        if all(m -> isa(m, LinearMappingIdentity), consensusMappings)
            coe0 = (consensusMappings[1]::LinearMappingIdentity).coe
            if all(m -> (m::LinearMappingIdentity).coe == coe0, consensusMappings)
                LinearMappingIdentityStacking(coe0, K)
            else
                LinearMappingStacking(consensusMappings, sizeAlongFirstDimension)
            end
        else
            LinearMappingStacking(consensusMappings, sizeAlongFirstDimension)
        end
    end

    # Single stacked constraint: A_stack(z) - x_stacked = 0
    newConstr = BlockConstraint(1)
    addBlockMappingToConstraint!(newConstr, consensusId, consensusMap)
    addBlockMappingToConstraint!(newConstr, stackedBlock.id, LinearMappingIdentity(-1.0))
    newConstr.rhs = zero(stackedVal)

    # Rewrite the problem in-place.
    mbp.blocks = BlockVariable[consensusBlock, stackedBlock]
    mbp.constraints = BlockConstraint[newConstr]

    return true
end 

function createFeasibilityProblem(mbp::MultiblockProblem; penalizeConstraints::Bool=false)
    mbpFeas = deepcopy(mbp)
    for block in mbpFeas.blocks
        block.f = Zero()
        if isSet(block.g) == false 
            block.g = Zero()
        end 
    end 
    if penalizeConstraints
        while true 
            for block in mbpFeas.blocks
                if isa(block.val, Vector{Float64}) == false 
                    @warn "createFeasibilityProblem: block $(block.id) is not a vector of Float64"
                    break 
                end 
            end     
            for constraint in mbpFeas.constraints
                for (id, mapping) in constraint.mappings
                    if isa(mapping, LinearMappingIdentity) == false && isa(mapping, LinearMappingMatrix) == false 
                        @warn "createFeasibilityProblem: constraint $(constraint.id) contains non-linear mapping: $(typeof(mapping))"
                        break 
                    end 
                end 
            end 
            mbpFeas.couplingFunction = transformConstraintsToQuadraticPenalty(mbpFeas)
            empty!(mbpFeas.constraints)
            break 
        end 
    end 
    return mbpFeas
end

"""
    transformConstraintsToQuadraticPenalty(mbp::MultiblockProblem) -> QuadraticMultiblockFunction

Transform all constraints of the form A₁x₁ + A₂x₂ + ... + Aₙxₙ = b into a quadratic penalty 
function ||A₁x₁ + A₂x₂ + ... + Aₙxₙ - b||².

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem containing constraints to transform

# Returns
- `QuadraticMultiblockFunction`: A multiblock quadratic function representing the sum of squared constraint violations

# Mathematical Details
For each constraint of the form:
    ∑ᵢ Aᵢ(xᵢ) = b
    
where Aᵢ are linear mappings, this function constructs a quadratic penalty:
    f(x₁, ..., xₙ) = ∑_constraints ||∑ᵢ Aᵢ(xᵢ) - b||²

The resulting function is a quadratic multiblock function that can be used as a coupling function.

# Implementation Notes
- Each constraint contributes a term ||Ax - b||² to the total penalty
- The function handles various mapping types (identity, matrix, extraction)
- Block dimensions are automatically determined from the problem structure
- All constraints are combined into a single quadratic function

# Example
```julia
# Create a multiblock problem with constraints
mbp = MultiblockProblem()
# ... add blocks and constraints ...

# Transform constraints to quadratic penalty
penalty = transformConstraintsToQuadraticPenalty(mbp)

# The penalty function evaluates ||constraint_violations||²
violation_penalty = penalty([x1, x2, x3])
```
"""
function transformConstraintsToQuadraticPenalty(mbp::MultiblockProblem)
    if isempty(mbp.constraints)
        @warn("No constraints found in the problem")
        # Return a zero quadratic function
        blockDims = [length(block.val) for block in mbp.blocks]
        totalDim = sum(blockDims)
        Q = spzeros(totalDim, totalDim)
        q = zeros(totalDim)
        r = 0.0
        return QuadraticMultiblockFunction(Q, q, r, blockDims)
    end
    
    # Get block dimensions from the problem
    blockDims = [length(block.val) for block in mbp.blocks]
    totalDim = sum(blockDims)
    
    # Create mapping from block IDs to indices
    blockIdToIndex = Dict{BlockID, Int}()
    for (i, block) in enumerate(mbp.blocks)
        blockIdToIndex[block.id] = i
    end
    
    # Build block index ranges
    blockIndices = Vector{UnitRange{Int}}(undef, length(blockDims))
    startIdx = 1
    for i in 1:length(blockDims)
        endIdx = startIdx + blockDims[i] - 1
        blockIndices[i] = startIdx:endIdx
        startIdx = endIdx + 1
    end

    # Initialize the quadratic form components
    Q = spzeros(totalDim, totalDim)
    q = zeros(totalDim)
    r = 0.0
    
    # Process each constraint more efficiently
    for constraint in mbp.constraints
        # Get constraint right-hand side
        b = vec(constraint.rhs)
        constraintDim = length(b)
        
        # Add constant term: r += b'b
        r += dot(b, b)
        
        # Process constraint contribution block by block to avoid materializing full A matrix
        addConstraintContributionToQuadratic!(Q, q, constraint, mbp.blocks, blockIdToIndex, 
                                            blockDims, blockIndices, b)
    end
    
    return QuadraticMultiblockFunction(Q, q, r, blockDims)
end

"""
    addConstraintContributionToQuadratic!(Q::SparseMatrixCSC, q::Vector{Float64}, 
                                        constraint::BlockConstraint, blocks::Vector{BlockVariable},
                                        blockIdToIndex::Dict{BlockID, Int}, blockDims::Vector{Int},
                                        blockIndices::Vector{UnitRange{Int}}, b::Vector{Float64})

Add the contribution of a single constraint to the quadratic form Q and linear term q.

This function computes the contribution ||Ax - b||² = x'A'Ax - 2b'Ax + b'b
by directly updating Q += A'A and q -= 2A'b without materializing the full matrix A.

# Arguments
- `Q::SparseMatrixCSC`: Quadratic form matrix to update (modified in-place)
- `q::Vector{Float64}`: Linear term vector to update (modified in-place)
- `constraint::BlockConstraint`: The constraint to process
- `blocks::Vector{BlockVariable}`: All block variables in the problem
- `blockIdToIndex::Dict{BlockID, Int}`: Mapping from block IDs to indices
- `blockDims::Vector{Int}`: Dimensions of each block
- `blockIndices::Vector{UnitRange{Int}}`: Indices of each block in concatenated vector
- `b::Vector{Float64}`: Right-hand side vector of the constraint

# Implementation Notes
This function is more space-efficient than building the full constraint matrix A because:
1. It processes block mappings one at a time
2. It directly computes A'A contributions block by block
3. It avoids storing the full constraint matrix A
"""
function addConstraintContributionToQuadratic!(Q::SparseMatrixCSC, q::Vector{Float64}, 
                                             constraint::BlockConstraint, blocks::Vector{BlockVariable},
                                             blockIdToIndex::Dict{BlockID, Int}, blockDims::Vector{Int},
                                             blockIndices::Vector{UnitRange{Int}}, b::Vector{Float64})
    
    constraintDim = length(b)
    
    # Check if all mappings are LinearMappingIdentity for ultra-efficient implementation
    allIdentity = all(isa(mapping, LinearMappingIdentity) for mapping in values(constraint.mappings))
    
    if allIdentity
        # Ultra-efficient path for identity mappings: direct coefficient-based computation
        addIdentityConstraintContribution!(Q, q, constraint, blockIdToIndex, blockIndices, b)
    else
        # General path: compute A'A block by block without storing all matrices simultaneously
        for blockId_i in constraint.involvedBlocks
            blockIndex_i = blockIdToIndex[blockId_i]
            blockRange_i = blockIndices[blockIndex_i]
            mapping_i = constraint.mappings[blockId_i]
            
            # Build mapping matrix for block i
            A_i = buildMappingMatrix(mapping_i, blockDims[blockIndex_i], constraintDim)
            
            # Linear term contribution: q[blockRange_i] -= 2 * A_i' * b
            q[blockRange_i] -= 2 * (A_i' * b)
            
            # Quadratic term contributions: Q[blockRange_i, blockRange_j] += A_i' * A_j
            for blockId_j in constraint.involvedBlocks
                blockIndex_j = blockIdToIndex[blockId_j]
                blockRange_j = blockIndices[blockIndex_j]
                mapping_j = constraint.mappings[blockId_j]
                
                # Build mapping matrix for block j
                A_j = buildMappingMatrix(mapping_j, blockDims[blockIndex_j], constraintDim)
                
                # Add A_i' * A_j to the corresponding block of Q
                Q[blockRange_i, blockRange_j] += A_i' * A_j
            end
        end
    end
end

"""
    addIdentityConstraintContribution!(Q::SparseMatrixCSC, q::Vector{Float64}, 
                                     constraint::BlockConstraint, blockIdToIndex::Dict{BlockID, Int},
                                     blockIndices::Vector{UnitRange{Int}}, b::Vector{Float64})

Ultra-efficient implementation for constraints where all mappings are LinearMappingIdentity.

For identity mappings A_i = c_i * I, we have:
- A_i' * A_j = c_i * c_j * I (if i == j) or 0 (if i != j and blocks have different dimensions)  
- A_i' * b = c_i * b

This avoids any matrix construction and uses only scalar operations.
"""
function addIdentityConstraintContribution!(Q::SparseMatrixCSC, q::Vector{Float64}, 
                                          constraint::BlockConstraint, blockIdToIndex::Dict{BlockID, Int},
                                          blockIndices::Vector{UnitRange{Int}}, b::Vector{Float64})
    
    # Extract coefficients for all involved blocks
    coefficients = Dict{BlockID, Float64}()
    for blockId in constraint.involvedBlocks
        mapping = constraint.mappings[blockId]::LinearMappingIdentity
        coefficients[blockId] = mapping.coe
    end
    
    # Add contributions using only scalar operations
    for blockId_i in constraint.involvedBlocks
        blockIndex_i = blockIdToIndex[blockId_i]
        blockRange_i = blockIndices[blockIndex_i]
        c_i = coefficients[blockId_i]
        
        # Linear term: q[blockRange_i] -= 2 * c_i * b
        q[blockRange_i] -= 2 * c_i * b
        
        # Quadratic diagonal terms: Q[blockRange_i, blockRange_i] += c_i^2 * I
        for idx in blockRange_i
            Q[idx, idx] += c_i * c_i
        end
        
        # Quadratic off-diagonal terms: Q[blockRange_i, blockRange_j] += c_i * c_j * I
        for blockId_j in constraint.involvedBlocks
            if blockId_j != blockId_i
                blockIndex_j = blockIdToIndex[blockId_j]
                blockRange_j = blockIndices[blockIndex_j]
                c_j = coefficients[blockId_j]
                
                # Only add cross terms if blocks have same dimension
                if length(blockRange_i) == length(blockRange_j)
                    for (idx_i, idx_j) in zip(blockRange_i, blockRange_j)
                        Q[idx_i, idx_j] += c_i * c_j
                    end
                end
            end
        end
    end
end



"""
    buildMappingMatrix(mapping::AbstractMapping, inputDim::Int, outputDim::Int) -> SparseMatrixCSC

Build a matrix representation of an AbstractMapping.

# Arguments
- `mapping::AbstractMapping`: The mapping to convert to matrix form
- `inputDim::Int`: Dimension of the input space
- `outputDim::Int`: Dimension of the output space

# Returns
- `SparseMatrixCSC{Float64, Int}`: Matrix representation of the mapping

# Implementation Notes
- For `LinearMappingMatrix`: directly returns the stored matrix A
- For `LinearMappingIdentity`: creates a scaled identity matrix
- For other mappings: falls back to applying mapping to standard basis vectors
"""
function buildMappingMatrix(mapping::AbstractMapping, inputDim::Int, outputDim::Int)
    if isa(mapping, LinearMappingMatrix)
        # For matrix mappings, directly return the stored matrix
        return mapping.A
    elseif isa(mapping, LinearMappingIdentity)
        # For identity mappings, create a scaled identity matrix
        @assert inputDim == outputDim "LinearMappingIdentity requires input and output dimensions to be equal"
        if mapping.coe == 1.0
            return sparse(I, inputDim, inputDim)
        else
            return mapping.coe * sparse(I, inputDim, inputDim)
        end
    else
        # Generic fallback: apply mapping to standard basis vectors
        A = spzeros(outputDim, inputDim)
        
        for j in 1:inputDim
            # Create j-th standard basis vector
            e_j = zeros(inputDim)
            e_j[j] = 1.0
            
            # Apply mapping
            result = mapping(e_j)
            
            # Store result in j-th column
            A[:, j] = vec(result)
        end
        
        return A
    end
end 

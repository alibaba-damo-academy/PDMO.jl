"""
    BlockConstraint

Maintains a collection of block variable indices and associated mappings.

# Fields
- `id::BlockID`: A unique identifier for this constraint.
- `involvedBlocks::Vector{BlockID}`: A vector of block IDs; these indices are kept in increasing order.
- `mappings::Dict{BlockID, AbstractMapping}`: A dictionary mapping block IDs to their corresponding linear mappings.
- `rhs::NumericVariable`: The right-hand side of the constraint.

The constraint enforces a relationship of the form

    Σ (mapping(x[id])) = rhs

over the blocks indexed in `involvedBlocks`.
A default constructor is provided with `id = -1`, indicating an uninitialized state.
"""
mutable struct BlockConstraint 
    id::BlockID
    involvedBlocks::Vector{BlockID}         # indices of blocks involved in the constraint; orders matter. 
    mappings::Dict{BlockID, AbstractMapping}
    rhs::NumericVariable
end 

"""
    BlockConstraint(id::BlockID="")

Construct a new block constraint with the specified ID.

# Arguments
- `id::BlockID=""`: The identifier for the constraint. Can be an integer or string. Default is an empty string.

# Returns
- `BlockConstraint`: A new BlockConstraint instance with the given ID and empty collections.

# Throws
- `ErrorException`: If `id` is a negative integer.

# Notes
- If `id` is an empty string, a warning is issued that it might be overwritten later.
- The constraint is initialized with empty collections of involved blocks and mappings, and `0.0` for `rhs`.
"""
function BlockConstraint(id::BlockID="")
    if isa(id, Int64) && id < 0
        error("BlockConstraint: Block ID cannot be a negative integer.")
    end 
    if isa(id, String) && isempty(id)
        @warn("BlockConstraint: initialized with default ID; this ID might be overwritten later.")
    end 
    return BlockConstraint(id, Vector{BlockID}(), Dict{BlockID, AbstractMapping}(), 0.0)
end 

"""
    addBlockMappingToConstraint!(constr::BlockConstraint, blockID::BlockID, mapping::AbstractMapping)

Add a block mapping to a constraint.

# Arguments
- `constr::BlockConstraint`: The constraint to which the mapping will be added.
- `blockID::BlockID`: The identifier of the block to be mapped.
- `mapping::AbstractMapping`: The mapping to apply to the block.

# Throws
- `ErrorException`: If a mapping for the specified block ID already exists in the constraint.

# Notes
- The function adds the block ID to the list of involved blocks in the constraint.
- The mapping is stored in the constraint's mappings dictionary.
"""
function addBlockMappingToConstraint!(constr::BlockConstraint, blockID::BlockID, mapping::AbstractMapping)
    if haskey(constr.mappings, blockID)
        error("BlockConstraint: mapping for block $blockID already exists and cannot be overwritten.")
    end 
    constr.mappings[blockID] = mapping
    push!(constr.involvedBlocks, blockID)
end    

"""
    addBlockMappingsToConstraint!(constr::BlockConstraint, mappings::Dict{BlockID, AbstractMapping})

Add multiple block mappings to a constraint.

# Arguments
- `constr::BlockConstraint`: The constraint to which the mappings will be added.
- `mappings::Dict{BlockID, AbstractMapping}`: A dictionary of block IDs to mappings.

# Notes
- This function iterates through the provided mappings and adds each one to the constraint.
- It calls `addBlockMappingToConstraint!` for each mapping.
"""
function addBlockMappingsToConstraint!(constr::BlockConstraint, mappings::Dict{BlockID, AbstractMapping})
    for (blockID, mapping) in mappings
        addBlockMappingToConstraint!(constr, blockID, mapping)
    end 
end 

"""
    blockConstraintViolation!(constr::BlockConstraint, x::Dict{BlockID, NumericVariable}, ret::NumericVariable)

Compute the violation of the block constraint and store the result in `ret`.

# Arguments
- `constr::BlockConstraint`: The constraint to check.
- `x::Dict{BlockID, NumericVariable}`: A dictionary mapping block IDs to their current values.
- `ret::NumericVariable`: The variable where the result will be stored.

# Throws
- `ErrorException`: If the constraint has a scalar right-hand side, as in-place operation is not supported for scalar RHS.

# Notes
- This is an in-place operation that modifies `ret`.
- The result is the difference between the sum of mapped values and the right-hand side.
"""
function blockConstraintViolation!(constr::BlockConstraint, x::Dict{BlockID, NumericVariable}, ret::NumericVariable)
    if isa(constr.rhs, Number)
        error("BlockConstraint: does not support in-place operation for scalar rhs.")
    end 
    ret .= -constr.rhs 
    for (id, L) in constr.mappings
        L(x[id], ret, true)
    end 
end 

"""
    blockConstraintViolation(constr::BlockConstraint, x::Dict{BlockID, NumericVariable})

Compute the violation of the block constraint.

# Arguments
- `constr::BlockConstraint`: The constraint to check.
- `x::Dict{BlockID, NumericVariable}`: A dictionary mapping block IDs to their current values.

# Returns
- `NumericVariable`: The violation of the constraint.

# Notes
- For scalar right-hand sides, computes and returns the violation directly.
- For non-scalar right-hand sides, uses `blockConstraintViolation!` to compute the violation.
"""
function blockConstraintViolation(constr::BlockConstraint, x::Dict{BlockID, NumericVariable})
    if isa(constr.rhs, Number)
        ret = -constr.rhs 
        for (id, L) in constr.mappings
            ret += L(x[id])
        end 
        return ret
    else 
        ret = similar(constr.rhs)
        blockConstraintViolation!(constr, x, ret)
        return ret 
    end 
end 

"""
    blockConstraintViolationL2Norm!(constr::BlockConstraint, x::Dict{BlockID, NumericVariable}, ret::NumericVariable) -> Float64

Compute the L2 norm of the constraint violation.

# Arguments
- `constr::BlockConstraint`: The constraint to check.
- `x::Dict{BlockID, NumericVariable}`: A dictionary mapping block IDs to their current values.
- `ret::NumericVariable`: The variable where the intermediate violation will be stored.

# Returns
- `Float64`: The L2 norm of the constraint violation.

# Notes
- This is an in-place operation that modifies `ret`.
- First computes the constraint violation using `blockConstraintViolation!`, then its L2 norm.
"""
function blockConstraintViolationL2Norm!(constr, x::Dict{BlockID, NumericVariable}, ret::NumericVariable) 
    blockConstraintViolation!(constr, x, ret)
    return norm(ret, 2)
end 

"""
    blockConstraintViolationL2Norm(constr::BlockConstraint, x::Dict{BlockID, NumericVariable}) -> Float64

Compute the L2 norm of the constraint violation.

# Arguments
- `constr::BlockConstraint`: The constraint to check.
- `x::Dict{BlockID, NumericVariable}`: A dictionary mapping block IDs to their current values.

# Returns
- `Float64`: The L2 norm of the constraint violation.

# Notes
- Allocates a temporary variable for the computation.
- Uses `blockConstraintViolationL2Norm!` for the computation.
"""
function blockConstraintViolationL2Norm(constr, x::Dict{BlockID, NumericVariable}) 
    return blockConstraintViolationL2Norm!(constr, x, similar(constr.rhs))
end 

"""
    blockConstraintViolationLInfNorm!(constr::BlockConstraint, x::Dict{BlockID, NumericVariable}, ret::NumericVariable) -> Float64

Compute the L-infinity norm of the constraint violation.

# Arguments
- `constr::BlockConstraint`: The constraint to check.
- `x::Dict{BlockID, NumericVariable}`: A dictionary mapping block IDs to their current values.
- `ret::NumericVariable`: The variable where the intermediate violation will be stored.

# Returns
- `Float64`: The L-infinity norm of the constraint violation.

# Notes
- This is an in-place operation that modifies `ret`.
- First computes the constraint violation using `blockConstraintViolation!`, then its L-infinity norm.
"""
function blockConstraintViolationLInfNorm!(constr, x::Dict{BlockID, NumericVariable}, ret::NumericVariable) 
    blockConstraintViolation!(constr, x, ret)
    return norm(ret, Inf)
end 

"""
    blockConstraintViolationLInfNorm(constr::BlockConstraint, x::Dict{BlockID, NumericVariable}) -> Float64

Compute the L-infinity norm of the constraint violation.

# Arguments
- `constr::BlockConstraint`: The constraint to check.
- `x::Dict{BlockID, NumericVariable}`: A dictionary mapping block IDs to their current values.

# Returns
- `Float64`: The L-infinity norm of the constraint violation.

# Notes
- Allocates a temporary variable for the computation.
- Uses `blockConstraintViolationLInfNorm!` for the computation.
"""
function blockConstraintViolationLInfNorm(constr, x::Dict{BlockID, NumericVariable}) 
    return blockConstraintViolationLInfNorm!(constr, x, similar(constr.rhs))
end 

"""
    checkConstraintsViolation(constraints::Vector{BlockConstraint}, x::Dict{BlockID, NumericVariable}) -> Tuple{Float64, Float64}

Compute the total violation of a set of constraints.

# Arguments
- `constraints::Vector{BlockConstraint}`: A vector of constraints to check.
- `x::Dict{BlockID, NumericVariable}`: A dictionary mapping block IDs to their current values.

# Returns
- `Tuple{Float64, Float64}`: A tuple of (L2 norm of violations, L-infinity norm of violations).

# Notes
- The L2 norm is the square root of the sum of squared norms of individual constraint violations.
- The L-infinity norm is the maximum of the L-infinity norms of individual constraint violations.
"""
function checkConstraintsViolation(constraints::Vector{BlockConstraint}, x::Dict{BlockID, NumericVariable})
    presL2 = 0.0 
    presLinf = 0.0
    for constr in constraints 
        res = blockConstraintViolation(constr, x)
        presL2 += dot(res, res) 
        presLinf = max(presLinf, norm(res, Inf))
    end 
    presL2 = sqrt(presL2)
    return presL2, presLinf
end 

"""
    checkBlockConstraintValidity(constr::BlockConstraint) -> Bool

Check if a block constraint is valid.

# Arguments
- `constr::BlockConstraint`: The constraint to check.

# Returns
- `Bool`: `true` if the constraint is valid, `false` otherwise.

# Notes
- A valid constraint must:
  1. Have a properly initialized ID (not negative for integer IDs)
  2. Have matching lengths for `involvedBlocks` and `mappings`
  3. Have at least 2 blocks in `involvedBlocks`
  4. Have a mapping for each block in `involvedBlocks`
- Detailed error messages are printed for invalid constraints.
"""
function checkBlockConstraintValidity(constr::BlockConstraint)
    if isa(constr.id, Number) && constr.id < 0
        @warn("BlockConstraint: not initialized.")
        return false 
    end 

    if length(constr.involvedBlocks) != length(constr.mappings)
        @warn("BlockConstraint: involvedBlocks and mappings have different length.")
        return false 
    end 

    if length(constr.involvedBlocks) < 2 
        @warn("BlockConstraint: involvedBlocks has less than 2 blocks.")
        
        if isempty(constr.involvedBlocks)
            return false 
        end 
    end 


    for id in constr.involvedBlocks
        if haskey(constr.mappings, id) == false 
            @warn("BlockConstraint: involvedBlocks have different block indices")
            return false 
        end 
    end 

    return true 
end 


export BlockConstraint 
export addBlockMappingToConstraint!
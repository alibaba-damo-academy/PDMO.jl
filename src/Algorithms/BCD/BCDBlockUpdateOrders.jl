"""
    Block Selection Rules for BCD Algorithm

Block Coordinate Descent algorithms can use different strategies for selecting 
which block to update at each iteration.

The `updateBlockOrder!` function is called at the beginning of each iteration to 
determine which blocks will be updated and in what order. In the first iteration, 
this API also handles initialization of the block selection strategy.
"""

"""
    AbstractBlockUpdateOrder

Abstract base type for block selection strategies in BCD algorithms.
"""
abstract type AbstractBlockUpdateOrder end

"""
    updateBlockOrder!(rule::AbstractBlockUpdateOrder, mbp::MultiblockProblem, info::BCDIterationInfo)

Update the block selection order for the block update order strategy.

This function is called at the beginning of each BCD iteration to determine which 
blocks will be updated and in what order.
"""
function updateBlockOrder!(rule::AbstractBlockUpdateOrder, mbp::MultiblockProblem, info::BCDIterationInfo)
    error(" AbstractBlockUpdateOrder: updateBlockOrder! is not implemented for $(typeof(rule))")
end

"""
    CyclicRule <: AbstractBlockUpdateOrder

Cyclic block selection rule that updates blocks in sequential order 1, 2, ..., n.

This rule implements a deterministic block selection strategy where blocks are 
updated in a fixed cyclic pattern. The order is established during the first 
iteration and remains constant throughout the algorithm execution.

# Fields
- `blocksToUpdate::Vector{Int}`: Vector storing the order of blocks to update

# Constructors
- `CyclicRule()`: Creates a new cyclic rule with empty block order (initialized on first use)

# Example
```julia
rule = CyclicRule()
# Block order will be initialized as [1, 2, 3, ..., n] on first updateBlockOrder! call
```
"""
mutable struct CyclicRule <: AbstractBlockUpdateOrder 
    blocksToUpdate::Vector{Int}

    """
        CyclicRule()

    Create a new cyclic block selection rule.
    
    The block order is initialized as empty and will be set to [1, 2, ..., n] 
    during the first call to `updateBlockOrder!`, where n is the number of blocks 
    in the multiblock problem.
    
    # Returns
    - `CyclicRule`: A new cyclic rule instance with empty block order
    """
    function CyclicRule()
        new(Vector{Int}())
    end
end



"""
    updateBlockOrder!(rule::CyclicRule, mbp::MultiblockProblem, info::BCDIterationInfo)

Update the block selection order for the cyclic rule strategy.

This function is called at the beginning of each BCD iteration to determine which 
blocks will be updated and in what order. For the cyclic rule, blocks are updated 
in sequential order from 1 to n.

# Arguments
- `rule::CyclicRule`: The cyclic block selection rule to update
- `mbp::MultiblockProblem`: The multiblock problem containing the blocks
- `info::BCDIterationInfo`: Information about the current BCD iteration

# Behavior
- On first call (when `rule.blocksToUpdate` is empty), initializes the block order 
  as [1, 2, ..., n] where n is the number of blocks in the problem
- On subsequent calls, the block order remains unchanged (cyclic pattern)

# Example
```julia
rule = CyclicRule()
updateBlockOrder!(rule, problem, iteration_info)
# rule.blocksToUpdate now contains [1, 2, 3, ..., n]
```
"""
function updateBlockOrder!(rule::CyclicRule, mbp::MultiblockProblem, info::BCDIterationInfo)
    numberBlocks = length(mbp.blocks)
    
    # Validate input
    if numberBlocks <= 0
        throw(ArgumentError("MultiblockProblem must have at least one block"))
    end
    
    # Initialize (or resize) block order if needed.
    if length(rule.blocksToUpdate) != numberBlocks
        empty!(rule.blocksToUpdate)
        append!(rule.blocksToUpdate, 1:numberBlocks)
    end 
end



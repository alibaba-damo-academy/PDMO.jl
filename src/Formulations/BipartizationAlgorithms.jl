""" 
    BipartizationAlgorithms.jl

Algorithms for converting a multiblock graph into a bipartite graph for ADMM-style
decomposition.
"""

"""
    BipartizationAlgorithm

Enum of supported bipartization strategies.
"""
@enum BipartizationAlgorithm begin 
    MILP_BIPARTIZATION
    BFS_BIPARTIZATION
    DFS_BIPARTIZATION
    SPANNING_TREE_BIPARTIZATION
    GNN_BIPARTIZATION
end 

# Note: The enum values BFS_BIPARTIZATION, MILP_BIPARTIZATION, etc. are already directly accessible
# No need for additional constants since Julia enum values are available in the global scope 

"""
    getBipartizationAlgorithmName(alg::BipartizationAlgorithm) -> String

Return a stable display name for a `BipartizationAlgorithm`.

# Arguments
- `alg::BipartizationAlgorithm`: The bipartization algorithm enum value

# Returns
- `String`: algorithm name.
"""
function getBipartizationAlgorithmName(alg::BipartizationAlgorithm)
    if alg == MILP_BIPARTIZATION
        return "MILP_BIPARTIZATION"
    elseif alg == BFS_BIPARTIZATION
        return "BFS_BIPARTIZATION"
    elseif alg == DFS_BIPARTIZATION
        return "DFS_BIPARTIZATION"
    elseif alg == SPANNING_TREE_BIPARTIZATION
        return "SPANNING_TREE_BIPARTIZATION"
    elseif alg == GNN_BIPARTIZATION
        return "GNN_BIPARTIZATION"
    else 
        return "Unknown bipartization algorithm"
    end 
end 

# GNN bipartization is application-specific. Register a handler from application code.
const GNN_BIPARTIZATION_HANDLER = Ref{Union{Nothing, Function}}(nothing)

"""
    registerGnnBipartization!(handler::Function)

Register an application-specific GNN bipartization handler.

The handler must have the signature:
`handler(graph::MultiblockGraph, mbp::MultiblockProblem,
         nodesAssignment::Dict{String, Int64},
         edgesSplitting::Dict{String, Tuple{Int64, Int64}}; kwargs...)`
"""
function registerGnnBipartization!(handler::Function)
    GNN_BIPARTIZATION_HANDLER[] = handler
    return nothing
end

"""
    clearGnnBipartization!()

Clear any previously registered GNN bipartization handler.
"""
function clearGnnBipartization!()
    GNN_BIPARTIZATION_HANDLER[] = nothing
    return nothing
end

"""
    hasGnnBipartization() -> Bool

Return true if a GNN bipartization handler is registered.
"""
function hasGnnBipartization()
    return GNN_BIPARTIZATION_HANDLER[] !== nothing
end

"""
    GnnBipartization(graph, mbp, nodesAssignment, edgesSplitting; kwargs...)

Call the registered GNN bipartization handler.
"""
function GnnBipartization(graph::MultiblockGraph,
    mbp::MultiblockProblem,
    nodesAssignment::Dict{String, Int64},
    edgesSplitting::Dict{String, Tuple{Int64, Int64}};
    kwargs...)
    handler = GNN_BIPARTIZATION_HANDLER[]
    if handler === nothing
        error("GNN bipartization is not registered. Call registerGnnBipartization! from application code.")
    end
    return handler(graph, mbp, nodesAssignment, edgesSplitting; kwargs...)
end

# function MilpBipartization0(graph::MultiblockGraph, 
#     mbp::MultiblockProblem, 
#     nodesAssignment::Dict{String, Int64}, 
#     edgesSplitting::Dict{String, Tuple{Int64, Int64}})
    
#     empty!(nodesAssignment)
#     empty!(edgesSplitting)

#     nodes = collect(keys(graph.nodes))
#     edges = collect(keys(graph.edges))

#     model = JuMP.Model(HiGHS.Optimizer)
#     JuMP.set_silent(model)
    
#     # add variables 
#     JuMP.@variable(model, x_node_L[i in nodes], Bin)
#     JuMP.@variable(model, x_node_R[i in nodes], Bin)
#     JuMP.@variable(model, z_edge[e in edges], Bin)
#     JuMP.@variable(model, x_edge_L[e in edges], Bin)
#     JuMP.@variable(model, x_edge_R[e in edges], Bin)

#     # add constraints 
#     JuMP.@constraint(model, [i in nodes], x_node_L[i] + x_node_R[i] == 1)
#     JuMP.@constraint(model, [e in edges], x_edge_L[e] + x_edge_R[e] == z_edge[e])
    
#     JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_node_L[graph.edges[e].nodeID2] <= 1 + z_edge[e])
#     JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_node_L[graph.edges[e].nodeID2] >= 1 - z_edge[e])

#     JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_edge_L[e] <= 2 - z_edge[e])
#     JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_edge_L[e] >= z_edge[e])

#     JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID2] + x_edge_L[e] <= 2 - z_edge[e])
#     JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID2] + x_edge_L[e] >= z_edge[e])

#     # add objective terms for \|A\| and \|B\|
#     JuMP.@variable(model, t_L, lower_bound = 0.0)
#     JuMP.@variable(model, t_R, lower_bound = 0.0)

#     # create a mapping from constraint ID to constraint index in mbp.constraints
#     constraintID2Index = Dict{BlockID, Int64}() 
#     numberConstraints = length(mbp.constraints)
#     for idx in 1:numberConstraints 
#         constraintID2Index[mbp.constraints[idx].id] = idx
#     end 
    
#     for e in edges 
#         nodeID1 = graph.edges[e].nodeID1
#         nodeID2 = graph.edges[e].nodeID2
        
#         opnorm1 = 0.0 
#         opnorm2 = 0.0 

#         constrID = graph.edges[e].sourceBlockConstraint
#         constrIdx = constraintID2Index[constrID]

#         numberInvolvedBlocks = length(mbp.constraints[constrIdx].involvedBlocks)
#         if graph.edges[e].type == TWO_BLOCK_EDGE
#             @assert(numberInvolvedBlocks == 2, "MilpBipartization: $(numberInvolvedBlocks) block indices encountered; 2 expected")

#             blockID1 = mbp.constraints[constrIdx].involvedBlocks[1]
#             @assert(blockID1 == graph.nodes[nodeID1].source, "MilpBipartization: blockID1 mismatch")
            
#             blockID2 = mbp.constraints[constrIdx].involvedBlocks[2]
#             @assert(blockID2 == graph.nodes[nodeID2].source, "MilpBipartization: blockID2 mismatch")

#             opnorm1 = operatorNorm2(mbp.constraints[constrIdx].mappings[blockID1])
#             opnorm2 = operatorNorm2(mbp.constraints[constrIdx].mappings[blockID2])
#         else 
#             @assert(numberInvolvedBlocks > 2, "MilpBipartization: $(numberInvolvedBlocks) block indices encountered; > 2 expected")

#             blockID = graph.edges[e].sourceBlockVariable
#             @assert(blockID == graph.nodes[nodeID1].source, "MilpBipartization: blockID mismatch")
#             @assert(constrID == graph.nodes[nodeID2].source, "MilpBipartization: constraintID mismatch")

#             opnorm1 = operatorNorm2(mbp.constraints[constrIdx].mappings[blockID])
#             opnorm2 = 1.0 
#         end 
        
#         JuMP.@constraint(model, t_L >= opnorm1 * x_node_L[nodeID1] + x_edge_L[e])
#         JuMP.@constraint(model, t_L >= opnorm2 * x_node_L[nodeID2] + x_edge_L[e])

#         JuMP.@constraint(model, t_R >= opnorm1 * x_node_R[nodeID1] + x_edge_R[e])
#         JuMP.@constraint(model, t_R >= opnorm2 * x_node_R[nodeID2] + x_edge_R[e])
#     end

#     # JuMP.@objective(model, Min, t_L + t_R + number of nodes)
#     JuMP.@objective(model, Min, t_L + t_R + sum(x_node_L) + sum(x_node_R) + sum(x_edge_L) + sum(x_edge_R))

#     # optimize
#     JuMP.optimize!(model)

#     # collect features
#     for i in nodes 
#         if JuMP.value(x_node_L[i]) > 0.5 # the node belongs to L 
#             nodesAssignment[i] = 0 
#         else 
#             nodesAssignment[i] = 1      # the node belongs to R
#         end 
#     end 

#     for e in edges 
#         if JuMP.value(z_edge[e]) < 0.5
#             edgesSplitting[e] = (0,0)
#         else 
#             if JuMP.value(x_edge_L[e]) > 0.5 
#                 edgesSplitting[e] = (1,0)
#             else 
#                 edgesSplitting[e] = (1,1)
#             end 
#         end 
#     end 
# end 
""" 
    MilpBipartization(
        graph::MultiblockGraph,
        mbp::MultiblockProblem,
        nodesAssignment::Dict{String, Int64},
        edgesSplitting::Dict{String, Tuple{Int64, Int64}};
        mipRelGap::Float64 = 0.5,
        mipTimeLimit::Float64 = 60.0,
        mipHeuristicEffort::Float64 = 0.2,
    )

Bipartization algorithm using Mixed Integer Linear Programming (MILP).

This algorithm formulates the graph bipartization problem as an optimization problem that 
finds the optimal bipartite structure while minimizing operator norms and graph complexity.

# Arguments
- `graph::MultiblockGraph`: The graph to bipartize.
- `mbp::MultiblockProblem`: The original multiblock problem (used for operator norm calculations).
- `nodesAssignment::Dict{String, Int64}`: Output dictionary for node partition assignments (`0` for left, `1` for right).
  This dictionary is cleared at the beginning of the function and then populated.
- `edgesSplitting::Dict{String, Tuple{Int64, Int64}}`: Output dictionary for edge splitting decisions `(a, b)` where:
  - a = 0: edge not split, a = 1: edge split
  - b = 0: assigned to left partition, b = 1: assigned to right partition
  This dictionary is cleared at the beginning of the function and then populated.

# Keyword Arguments
- `mipRelGap::Float64=0.5`: Relative MIP optimality gap passed to HiGHS (`"mip_rel_gap"`).
- `mipTimeLimit::Float64=60.0`: Time limit (seconds) passed to HiGHS (`"time_limit"`).
- `mipHeuristicEffort::Float64=0.2`: Heuristic effort passed to HiGHS (`"mip_heuristic_effort"`).

# Algorithm Steps
1. **Variable Creation**: Binary variables for node assignments and edge splitting decisions
2. **Constraint Formulation**: Ensures each node belongs to exactly one partition
3. **Bipartite Constraints**: Prevents adjacent nodes from being in the same partition (or splits the edge)
4. **Objective Optimization**: Minimizes operator norms and graph complexity
5. **Solution Extraction**: Converts optimal solution to node assignments and edge splitting decisions

# Mathematical Formulation
- **Variables**: `x_node_L[i]`, `x_node_R[i]` (node assignments), `z_edge[e]` (edge splitting), `x_edge_L[e]`, `x_edge_R[e]` (edge assignments)
- **Constraints**: Partition constraints, bipartite constraints, edge splitting logic
- **Objective**: `min t_L + t_R + complexity_terms` where `t_L`, `t_R` are operator norm bounds

# Advantages
- **Optimal Solution**: Finds the best bipartization according to the objective function
- **Operator Norm Awareness**: Considers the numerical properties of the original problem
- **Principled Approach**: Mathematical optimization rather than heuristic

# Disadvantages
- **Computational Cost**: Slower than heuristic methods, especially for large graphs
- **Solver Dependency**: Requires HiGHS or another MILP solver
- **Memory Usage**: May require significant memory for large problems

# Usage
This algorithm is recommended when solution quality is more important than computational speed,
particularly for problems where operator norm considerations are critical for numerical stability.

# Implementation Notes
- Uses HiGHS solver with silent mode enabled
- Automatically handles both TWO_BLOCK_EDGE and MULTIBLOCK_EDGE types
- Operator norms are computed using the `operatorNorm2` function from the mappings
- The dictionaries `nodesAssignment` and `edgesSplitting` are cleared and populated by the algorithm
"""
function MilpBipartization(graph::MultiblockGraph, 
    mbp::MultiblockProblem, 
    nodesAssignment::Dict{String, Int64}, 
    edgesSplitting::Dict{String, Tuple{Int64, Int64}};
    mipRelGap::Float64=0.5, 
    mipTimeLimit::Float64=60.0,
    mipHeuristicEffort::Float64=0.2)
    
    empty!(nodesAssignment)
    empty!(edgesSplitting)

    nodes = collect(keys(graph.nodes))
    edges = collect(keys(graph.edges))

    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_optimizer_attribute(model, "mip_rel_gap", mipRelGap)
    JuMP.set_optimizer_attribute(model, "mip_heuristic_effort", mipHeuristicEffort)
    JuMP.set_optimizer_attribute(model, "time_limit", mipTimeLimit)
    # JuMP.set_silent(model)
    
    # add variables 
    JuMP.@variable(model, x_node_L[i in nodes], Bin)
    JuMP.@variable(model, x_node_R[i in nodes], Bin)
    JuMP.@variable(model, z_edge[e in edges], Bin)
    JuMP.@variable(model, x_edge_L[e in edges], Bin)
    JuMP.@variable(model, x_edge_R[e in edges], Bin)

    # add constraints 
    JuMP.@constraint(model, [i in nodes], x_node_L[i] + x_node_R[i] == 1)
    JuMP.@constraint(model, [e in edges], x_edge_L[e] + x_edge_R[e] == z_edge[e])
    
    JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_node_L[graph.edges[e].nodeID2] <= 1 + z_edge[e])
    JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_node_L[graph.edges[e].nodeID2] >= 1 - z_edge[e])

    JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_edge_L[e] <= 2 - z_edge[e])
    JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID1] + x_edge_L[e] >= z_edge[e])

    JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID2] + x_edge_L[e] <= 2 - z_edge[e])
    JuMP.@constraint(model, [e in edges], x_node_L[graph.edges[e].nodeID2] + x_edge_L[e] >= z_edge[e])

    # add objective terms for \|A\| and \|B\|
    JuMP.@variable(model, t_L, lower_bound = 0.0)
    JuMP.@variable(model, t_R, lower_bound = 0.0)

    # create a mapping from constraint ID to constraint index in mbp.constraints
    constraintID2Index = Dict{BlockID, Int64}() 
    numberConstraints = length(mbp.constraints)
    for idx in 1:numberConstraints 
        constraintID2Index[mbp.constraints[idx].id] = idx
    end 
    
    nodeNorm = Dict{String, Float64}(i=>0.0 for i in nodes)
    for e in edges 
        nodeID1 = graph.edges[e].nodeID1
        nodeID2 = graph.edges[e].nodeID2
        
        constrID = graph.edges[e].sourceBlockConstraint
        constrIdx = constraintID2Index[constrID]

        numberInvolvedBlocks = length(mbp.constraints[constrIdx].involvedBlocks)
        if graph.edges[e].type == TWO_BLOCK_EDGE
            @assert(numberInvolvedBlocks == 2, "MilpBipartization: $(numberInvolvedBlocks) block indices encountered; 2 expected")

            blockID1 = mbp.constraints[constrIdx].involvedBlocks[1]
            @assert(blockID1 == graph.nodes[nodeID1].source, "MilpBipartization: blockID1 mismatch")
            
            blockID2 = mbp.constraints[constrIdx].involvedBlocks[2]
            @assert(blockID2 == graph.nodes[nodeID2].source, "MilpBipartization: blockID2 mismatch")

            nodeNorm[nodeID1] += operatorNorm2(mbp.constraints[constrIdx].mappings[blockID1])^2
            nodeNorm[nodeID2] += operatorNorm2(mbp.constraints[constrIdx].mappings[blockID2])^2
        else 
            @assert(numberInvolvedBlocks > 2, "MilpBipartization: $(numberInvolvedBlocks) block indices encountered; > 2 expected")

            blockID = graph.edges[e].sourceBlockVariable
            @assert(blockID == graph.nodes[nodeID1].source, "MilpBipartization: blockID mismatch")
            @assert(constrID == graph.nodes[nodeID2].source, "MilpBipartization: constraintID mismatch")

            nodeNorm[nodeID1] += operatorNorm2(mbp.constraints[constrIdx].mappings[blockID])^2
            nodeNorm[nodeID2] += 1.0 
        end 
    end

    for i in nodes 
        nodeNorm[i] = sqrt(nodeNorm[i])
        JuMP.@constraint(model, t_L >= nodeNorm[i] * x_node_L[i])
        JuMP.@constraint(model, t_R >= nodeNorm[i] * x_node_R[i])
    end 
    for e in edges 
        JuMP.@constraint(model, t_L >= 1.414 * x_edge_L[e])
        JuMP.@constraint(model, t_R >= 1.414 * x_edge_R[e])
    end 


    # JuMP.@objective(model, Min, t_L + t_R + number of nodes)
    JuMP.@objective(model, Min, t_L + t_R + sum(x_node_L) + sum(x_node_R) + sum(x_edge_L) + sum(x_edge_R))

    # optimize
    JuMP.optimize!(model)

    # collect features
    for i in nodes 
        if JuMP.value(x_node_L[i]) > 0.5 # the node belongs to L 
            nodesAssignment[i] = 0 
        else 
            nodesAssignment[i] = 1      # the node belongs to R
        end 
    end 

    for e in edges 
        if JuMP.value(z_edge[e]) < 0.5
            edgesSplitting[e] = (0,0)
        else 
            if JuMP.value(x_edge_L[e]) > 0.5 
                edgesSplitting[e] = (1,0)
            else 
                edgesSplitting[e] = (1,1)
            end 
        end 
    end 
end 

""" 
    BfsBipartization(graph::MultiblockGraph, mbp::MultiblockProblem, 
                    nodesAssignment::Dict{String, Int64}, 
                    edgesSplitting::Dict{String, Tuple{Int64, Int64}})

Bipartization algorithm using Breadth-First Search (BFS).

This algorithm uses BFS traversal to quickly assign nodes to partitions and create a bipartite graph
by splitting edges when conflicts are detected.

# Arguments
- `graph::MultiblockGraph`: The graph to bipartize
- `mbp::MultiblockProblem`: The original multiblock problem (not used in algorithm, included for interface consistency)
- `nodesAssignment::Dict{String, Int64}`: Dictionary to store node partition assignments (0 for left, 1 for right)
- `edgesSplitting::Dict{String, Tuple{Int64, Int64}}`: Dictionary to store edge splitting decisions (a,b) where:
  - a = 0: edge not split, a = 1: edge split
  - b = 0: assigned to left partition, b = 1: assigned to right partition

# Algorithm Steps
1. **Neighbor Construction**: Build adjacency list from graph edges
2. **Connected Components**: Process each connected component separately
3. **BFS Traversal**: Starting from unvisited nodes, perform BFS with alternating partition assignment
4. **Conflict Detection**: When adjacent nodes would have the same partition, split the connecting edge
5. **Edge Processing**: Set splitting decisions for all edges (split or keep)

# BFS Process
- **Queue-Based**: Uses FIFO queue for breadth-first traversal
- **Alternating Assignment**: Assigns nodes to alternating partitions (0, 1, 0, 1, ...)
- **Conflict Resolution**: Splits edges when both endpoints would be in the same partition
- **Component Handling**: Alternates starting partition for different connected components

# Advantages
- **Fast Execution**: O(V + E) time complexity where V = vertices, E = edges
- **Simple Implementation**: Straightforward algorithm with predictable behavior
- **Memory Efficient**: Minimal memory overhead beyond input graph
- **Handles Disconnected Graphs**: Automatically processes all connected components

# Disadvantages
- **Suboptimal Results**: May split more edges than necessary
- **No Operator Norm Consideration**: Ignores numerical properties of the original problem
- **Traversal Order Dependence**: Results may depend on node iteration order

# Usage
This algorithm is recommended when:
- Fast execution is more important than optimal results
- The graph is large and MILP would be too slow
- A reasonable bipartite approximation is sufficient
- Operator norm considerations are not critical

# Implementation Notes
- Processes connected components independently
- Alternates starting partition (0, 1, 0, 1, ...) for different components
- All unprocessed edges are marked as not split (0, 0)
- The dictionaries `nodesAssignment` and `edgesSplitting` are cleared and populated by the algorithm
"""
function BfsBipartization(graph::MultiblockGraph, 
    mbp::MultiblockProblem, 
    nodesAssignment::Dict{String, Int64}, 
    edgesSplitting::Dict{String, Tuple{Int64, Int64}})

    empty!(nodesAssignment)
    empty!(edgesSplitting)

    # Get neighbors for each node
    neighbors = Dict{String, Set{Tuple{String, String}}}(nodeID=>Set{Tuple{String, String}}() for nodeID in keys(graph.nodes))
    for (edgeID, edge) in graph.edges 
        push!(neighbors[edge.nodeID1], (edgeID, edge.nodeID2))
        push!(neighbors[edge.nodeID2], (edgeID, edge.nodeID1))
    end

    # BFS to assign nodes to partitions - handle all connected components
    startPartition = 0  # Starting partition for each component
    
    for startNode in keys(graph.nodes)
        if !haskey(nodesAssignment, startNode)  # Process unvisited nodes
            # Start BFS for this connected component
            nodesAssignment[startNode] = startPartition
            queue = String[startNode]

            while !isempty(queue)
                current = popfirst!(queue)
                currentPartition = nodesAssignment[current]

                # Assign neighbors to opposite partition
                for (edgeID, neighbor) in neighbors[current]
                    if !haskey(nodesAssignment, neighbor)
                        nodesAssignment[neighbor] = 1 - currentPartition
                        push!(queue, neighbor)
                    elseif nodesAssignment[neighbor] == currentPartition
                        # Conflict: neighbor has same partition, split the connecting edge
                        edgesSplitting[edgeID] = (1, 1 - currentPartition)
                    end
                end
            end
            
            # Alternate starting partition for next disconnected component
            startPartition = 1 - startPartition
        end
    end

    # Handle any remaining edges
    for (edgeID, edge) in graph.edges
        if haskey(edgesSplitting, edgeID) == false
            edgesSplitting[edgeID] = (0, 0)
        end
    end
end

""" 
    DfsBipartization(graph::MultiblockGraph, mbp::MultiblockProblem, 
                    nodesAssignment::Dict{String, Int64}, 
                    edgesSplitting::Dict{String, Tuple{Int64, Int64}})

Bipartization algorithm using Depth-First Search (DFS).

This algorithm uses DFS traversal to assign nodes to partitions and create a bipartite graph
by splitting edges when conflicts are detected. Similar to BFS but with different traversal order.

# Arguments
- `graph::MultiblockGraph`: The graph to bipartize
- `mbp::MultiblockProblem`: The original multiblock problem (not used in algorithm, included for interface consistency)
- `nodesAssignment::Dict{String, Int64}`: Dictionary to store node partition assignments (0 for left, 1 for right)
- `edgesSplitting::Dict{String, Tuple{Int64, Int64}}`: Dictionary to store edge splitting decisions (a,b) where:
  - a = 0: edge not split, a = 1: edge split
  - b = 0: assigned to left partition, b = 1: assigned to right partition

# Algorithm Steps
1. **Neighbor Construction**: Build adjacency list from graph edges
2. **Connected Components**: Process each connected component separately
3. **DFS Traversal**: Starting from unvisited nodes, perform DFS with alternating partition assignment
4. **Conflict Detection**: When adjacent nodes would have the same partition, split the connecting edge
5. **Edge Processing**: Set splitting decisions for all edges (split or keep)

# DFS Process
- **Stack-Based**: Uses LIFO stack for depth-first traversal
- **Alternating Assignment**: Assigns nodes to alternating partitions (0, 1, 0, 1, ...)
- **Conflict Resolution**: Splits edges when both endpoints would be in the same partition
- **Component Handling**: Alternates starting partition for different connected components

# Advantages
- **Fast Execution**: O(V + E) time complexity where V = vertices, E = edges
- **Simple Implementation**: Straightforward algorithm with predictable behavior
- **Memory Efficient**: Minimal memory overhead beyond input graph
- **Handles Disconnected Graphs**: Automatically processes all connected components
- **Different Traversal Pattern**: May produce different (sometimes better) results than BFS

# Disadvantages
- **Suboptimal Results**: May split more edges than necessary
- **No Operator Norm Consideration**: Ignores numerical properties of the original problem
- **Traversal Order Dependence**: Results may depend on node iteration order
- **Stack Depth**: May require significant stack space for deep graphs

# Comparison with BFS
- **Traversal Order**: DFS explores deeply before backtracking, BFS explores level by level
- **Memory Usage**: DFS uses implicit recursion stack, BFS uses explicit queue
- **Results**: May produce different partitioning results for the same graph
- **Performance**: Similar time complexity, but different memory access patterns

# Usage
This algorithm is recommended when:
- Fast execution is more important than optimal results
- You want to try a different heuristic than BFS
- The graph structure might benefit from depth-first exploration
- Operator norm considerations are not critical

# Implementation Notes
- Uses explicit stack with `pop!()` for LIFO behavior (vs `popfirst!()` for BFS)
- Processes connected components independently
- Alternates starting partition (0, 1, 0, 1, ...) for different components
- All unprocessed edges are marked as not split (0, 0)
- The dictionaries `nodesAssignment` and `edgesSplitting` are cleared and populated by the algorithm
"""
function DfsBipartization(graph::MultiblockGraph, 
    mbp::MultiblockProblem, 
    nodesAssignment::Dict{String, Int64}, 
    edgesSplitting::Dict{String, Tuple{Int64, Int64}})

    empty!(nodesAssignment)
    empty!(edgesSplitting)

    # Get neighbors for each node
    neighbors = Dict{String, Set{Tuple{String, String}}}(nodeID=>Set{Tuple{String, String}}() for nodeID in keys(graph.nodes))
    for (edgeID, edge) in graph.edges 
        push!(neighbors[edge.nodeID1], (edgeID, edge.nodeID2))
        push!(neighbors[edge.nodeID2], (edgeID, edge.nodeID1))
    end

    # DFS to assign nodes to partitions - handle all connected components
    startPartition = 0  # Starting partition for each component
    
    for startNode in keys(graph.nodes)
        if !haskey(nodesAssignment, startNode)  # Process unvisited nodes
            # Start DFS for this connected component
            nodesAssignment[startNode] = startPartition
            stack = String[startNode]

            while !isempty(stack)
                current = pop!(stack)  # Use pop! instead of popfirst! for DFS (LIFO)
                currentPartition = nodesAssignment[current]

                # Assign neighbors to opposite partition
                for (edgeID, neighbor) in neighbors[current]
                    if !haskey(nodesAssignment, neighbor)
                        nodesAssignment[neighbor] = 1 - currentPartition
                        push!(stack, neighbor)
                    elseif nodesAssignment[neighbor] == currentPartition
                        # Conflict: neighbor has same partition, split the connecting edge
                        edgesSplitting[edgeID] = (1, 1 - currentPartition)
                    end
                end
            end
            
            # Alternate starting partition for next disconnected component
            startPartition = 1 - startPartition
        end
    end

    # Handle any remaining edges
    for (edgeID, edge) in graph.edges
        if haskey(edgesSplitting, edgeID) == false
            edgesSplitting[edgeID] = (0, 0)
        end
    end
end

""" 
    SpanningTreeBipartization(graph::MultiblockGraph, mbp::MultiblockProblem, 
                             nodesAssignment::Dict{String, Int64}, 
                             edgesSplitting::Dict{String, Tuple{Int64, Int64}})

Bipartization algorithm using Spanning Tree with Smart Back Edge Processing.

This algorithm provides a balanced approach between solution quality and computational efficiency 
by intelligently processing edges based on their role in the graph structure.

# Arguments
- `graph::MultiblockGraph`: The graph to bipartize
- `mbp::MultiblockProblem`: The original multiblock problem (not used in algorithm, included for interface consistency)
- `nodesAssignment::Dict{String, Int64}`: Dictionary to store node partition assignments (0 for left, 1 for right)
- `edgesSplitting::Dict{String, Tuple{Int64, Int64}}`: Dictionary to store edge splitting decisions (a,b) where:
  - a = 0: edge not split, a = 1: edge split
  - b = 0: assigned to left partition, b = 1: assigned to right partition

# Algorithm Steps
1. **Spanning Tree Construction**: Build spanning tree using DFS traversal
2. **Tree Coloring**: 2-color the spanning tree (trees are always bipartite)
3. **Edge Classification**: Classify edges as tree edges or back edges
4. **Smart Back Edge Processing**: Only split back edges if endpoints have the same color
5. **Edge Decision Assignment**: Set splitting decisions for all edges

# Mathematical Foundation
- **Spanning Tree Property**: Any tree is bipartite and can be 2-colored
- **Back Edge Analysis**: A back edge preserves bipartiteness if its endpoints have different colors
- **Optimal Splitting**: Only splits edges that would violate bipartiteness

# Algorithm Process
- **Tree Edges**: Always kept intact (never split) since they maintain bipartite structure
- **Back Edges with Different Colors**: Kept intact (already valid for bipartite graph)
- **Back Edges with Same Color**: Split to avoid violating bipartiteness
- **Connected Components**: Each component processed independently

# Advantages
- **Balanced Approach**: Good compromise between speed and solution quality
- **Minimal Splitting**: Splits only edges that truly violate bipartiteness
- **Deterministic**: Always produces the same result for the same graph
- **Handles Disconnected Graphs**: Automatically processes all connected components
- **Mathematically Sound**: Based on solid graph theory principles

# Disadvantages
- **No Operator Norm Consideration**: Ignores numerical properties of the original problem
- **Spanning Tree Dependence**: Results may depend on which spanning tree is chosen
- **Not Globally Optimal**: May not find the absolute minimum number of splits

# Comparison with Other Algorithms
- **vs MILP**: Faster but may not be globally optimal
- **vs BFS/DFS**: Usually produces fewer edge splits due to intelligent edge processing
- **vs Naive Spanning Tree**: Much more efficient (doesn't split all back edges)

# Usage
This algorithm is recommended when:
- You want a balance between solution quality and computational speed
- The graph has many cycles (where back edge processing provides benefits)
- You prefer a deterministic algorithm over heuristics
- Operator norm considerations are not critical

# Implementation Notes
- Uses DFS to build spanning tree for each connected component
- Maintains separate sets for visited nodes and tree edges
- 2-colors the spanning tree using BFS
- All back edges are analyzed for potential splitting
- The dictionaries `nodesAssignment` and `edgesSplitting` are cleared and populated by the algorithm

# Time Complexity
- **Spanning Tree Construction**: O(V + E)
- **Tree Coloring**: O(V + E)
- **Back Edge Processing**: O(E)
- **Overall**: O(V + E) where V = vertices, E = edges
"""
function SpanningTreeBipartization(graph::MultiblockGraph, 
    mbp::MultiblockProblem, 
    nodesAssignment::Dict{String, Int64}, 
    edgesSplitting::Dict{String, Tuple{Int64, Int64}})

    empty!(nodesAssignment)
    empty!(edgesSplitting)

    # Build adjacency list
    neighbors = Dict{String, Set{Tuple{String, String}}}(nodeID=>Set{Tuple{String, String}}() for nodeID in keys(graph.nodes))
    for (edgeID, edge) in graph.edges 
        push!(neighbors[edge.nodeID1], (edgeID, edge.nodeID2))
        push!(neighbors[edge.nodeID2], (edgeID, edge.nodeID1))
    end

    # DFS to build spanning tree and handle disconnected components
    visited = Set{String}()
    treeEdges = Set{String}()
    
    function dfs_spanning_tree(node::String)
        push!(visited, node)
        for (edgeID, neighbor) in neighbors[node]
            if neighbor ∉ visited
                push!(treeEdges, edgeID)
                dfs_spanning_tree(neighbor)
            end
        end
    end
    
    # Process all connected components
    for startNode in keys(graph.nodes)
        if startNode ∉ visited
            dfs_spanning_tree(startNode)
        end
    end

    # 2-color the spanning tree (process all components)
    for startNode in keys(graph.nodes)
        if !haskey(nodesAssignment, startNode)
            nodesAssignment[startNode] = 0
            queue = String[startNode]
            
            while !isempty(queue)
                current = popfirst!(queue)
                currentPartition = nodesAssignment[current]
                
                for (edgeID, neighbor) in neighbors[current]
                    if edgeID in treeEdges && !haskey(nodesAssignment, neighbor)
                        nodesAssignment[neighbor] = 1 - currentPartition
                        push!(queue, neighbor)
                    end
                end
            end
        end
    end

    # Process edges based on their classification
    for (edgeID, edge) in graph.edges
        if edgeID in treeEdges
            # Tree edges: Never split (they maintain bipartite structure)
            edgesSplitting[edgeID] = (0, 0)
        else
            # Back edges: Check if subdivision is needed
            node1 = edge.nodeID1
            node2 = edge.nodeID2
            
            if nodesAssignment[node1] == nodesAssignment[node2]
                # Same partition: This back edge would violate bipartiteness, so split it
                edgesSplitting[edgeID] = (1, 1 - nodesAssignment[node1])
            else 
                # Different partitions: This back edge is already valid for bipartite graph
                edgesSplitting[edgeID] = (0, 0)
            end 
        end
    end
end


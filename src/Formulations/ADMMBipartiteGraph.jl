"""
    ADMMNode

A node in the ADMM bipartite graph.

# Fields
- `f::AbstractFunction`: Smooth/primary term for this node.
- `g::AbstractFunction`: Proximal/non-smooth term for this node.
- `val::NumericVariable`: Current node value.
- `neighbors::Set{String}`: Incident ADMM edge IDs.
- `convertedEdgeID::String`: Original edge ID if created by edge splitting, else `""`.
- `assignment::Int`: Bipartite partition assignment (`0` left, `1` right).
"""
mutable struct ADMMNode 
    f::AbstractFunction 
    g::AbstractFunction 
    val::NumericVariable
    neighbors::Set{String}      # neighbors of ADMMEdge
    convertedEdgeID::String     # edge ID from MultiblockGraph; "" if it is a variable node
    assignment::Int
end 

"""
    ADMMEdge

A linear constraint edge between two `ADMMNode`s.

# Fields
- `nodeID1::String`: First endpoint node ID.
- `nodeID2::String`: Second endpoint node ID.
- `mappings::Dict{String, AbstractMapping}`: Per-endpoint linear maps.
- `rhs::NumericVariable`: Constraint right-hand side.
- `splittedEdgeID::String`: Original graph edge ID if this edge was split, else `""`.
"""
mutable struct ADMMEdge 
    nodeID1::String             # ADMM node ID of the first node
    nodeID2::String             # ADMM node ID of the second node; if splitted edge, this is the ID of the new node
    mappings::Dict{String, AbstractMapping}  
    rhs::NumericVariable
    splittedEdgeID::String      # edge ID from MultiblockGraph (= admmGraph.nodes[nodeID2].convertedEdgeID); "" if it is not a splitted edge
end

"""
    createADMMNodeID(edgeID::String) -> String

Create the auxiliary ADMM node ID for a split edge.

# Arguments
- `edgeID::String`: The original edge ID from the MultiblockGraph that is being split

# Returns
- `String`: `ADMMNodeConvertedFromEdge(<edgeID>)`.
"""
function createADMMNodeID(edgeID::String)
    return "ADMMNodeConvertedFromEdge($edgeID)"
end

"""
    createADMMEdgeID(edgeID::String, nodeID::String) -> String

Create the ADMM edge ID for an edge produced by splitting `edgeID`.

# Arguments
- `edgeID::String`: The original edge ID from the MultiblockGraph that was split
- `nodeID::String`: The ID of the node that this new edge connects to (either original node or auxiliary node)

# Returns
- `String`: `ADMMEdgeSplittedFrom(<edgeID>, <nodeID>)`.
"""
function createADMMEdgeID(edgeID::String, nodeID::String)
    return "ADMMEdgeSplittedFrom($edgeID, $nodeID)"
end

"""
    ADMMBipartiteGraph

A bipartite graph representation used by ADMM.

# Fields
- `nodes::Dict{String, ADMMNode}`: ADMM graph nodes.
- `edges::Dict{String, ADMMEdge}`: ADMM graph edges.
- `mbpBlockID2admmNodeID::Dict{BlockID, String}`: Mapping from original block IDs to ADMM node IDs.
  - Enables traceability between original problem formulation and ADMM representation
  - Used for solution extraction and result interpretation
- `left::Vector{String}`: Node IDs assigned to the left partition (typically assignment = 0)
- `right::Vector{String}`: Node IDs assigned to the right partition (typically assignment = 1)

# Bipartite Structure Properties
- **Partition Guarantee**: No edges exist between nodes in the same partition
- **ADMM Compatibility**: Structure enables alternating updates between partitions
- **Constraint Preservation**: All original constraints are represented through edges
- **Auxiliary Nodes**: May contain additional nodes created during bipartization

# Graph Construction Process
1. **Node Creation**: Transform MultiblockProblem blocks into ADMM nodes
2. **Edge Creation**: Transform constraints into ADMM edges
3. **Bipartization**: Apply bipartization algorithm if necessary
4. **Edge Splitting**: Create auxiliary nodes and edges to maintain bipartite structure
5. **Partition Assignment**: Assign nodes to left/right partitions
6. **Validation**: Verify bipartite property and constraint preservation

# Constructors
    ADMMBipartiteGraph()  # Empty graph
    ADMMBipartiteGraph(graph::MultiblockGraph, mbp::MultiblockProblem, nodesAssignment, edgesSplitting)
    ADMMBipartiteGraph(graph::MultiblockGraph, mbp::MultiblockProblem, algorithm::BipartizationAlgorithm)

# Usage in ADMM Algorithm
- **x-update**: Update variables in one partition (typically left)
- **z-update**: Update variables in other partition (typically right)
- **Dual update**: Update Lagrange multipliers associated with edges
- **Residual computation**: Compute primal and dual residuals using edge constraints
- **Convergence check**: Monitor constraint violations and variable changes

# Mathematical Representation
For a bipartite graph with partitions L (left) and R (right):
- Variables: x_L (left partition), x_R (right partition)
- Constraints: Each edge (i,j) with i∈L, j∈R represents A_i x_i + A_j x_j = b_{ij}
- ADMM updates alternate between optimizing over x_L and x_R

# Examples
```julia
# Create from MultiblockProblem with specific algorithm
mbp = MultiblockProblem()
# ... add blocks and constraints ...
graph = MultiblockGraph(mbp)
admm_graph = ADMMBipartiteGraph(graph, mbp, BFS_BIPARTIZATION)

# Access graph properties
println("Left partition size: ", length(admm_graph.left))
println("Right partition size: ", length(admm_graph.right))
println("Number of constraints: ", length(admm_graph.edges))
```

# Related Types
- `ADMMNode`: Individual nodes in the bipartite graph
- `ADMMEdge`: Edges representing constraints between nodes
- `MultiblockGraph`: Original graph representation before bipartization
- `BipartizationAlgorithm`: Algorithms for ensuring bipartite structure
"""
mutable struct ADMMBipartiteGraph 
    nodes::Dict{String, ADMMNode}
    edges::Dict{String, ADMMEdge}
    mbpBlockID2admmNodeID::Dict{BlockID, String}
    left::Vector{String}
    right::Vector{String}
    partitionAlgorithmTime::Float64
    # default constructor
    ADMMBipartiteGraph() = new(Dict{String, ADMMNode}(), 
        Dict{String, ADMMEdge}(), 
        Dict{BlockID, String}(), 
        Vector{String}(), 
        Vector{String}(), 
        Inf)
end 

"""
    ADMMBipartiteGraph(graph::MultiblockGraph, mbp::MultiblockProblem, 
                      nodesAssignment::Dict{String, Int64}, 
                      edgesSplitting::Dict{String, Tuple{Int64, Int64}}) -> ADMMBipartiteGraph

Construct an ADMM bipartite graph from a multiblock graph and problem using provided node assignments and edge splitting decisions.

This is the core constructor that transforms a general multiblock optimization problem into a bipartite
graph suitable for ADMM decomposition. It handles both original constraints and edge splitting to
maintain bipartite structure while preserving the mathematical properties of the original problem.

# Arguments
- `graph::MultiblockGraph`: The original multiblock graph representation
- `mbp::MultiblockProblem`: The original multiblock optimization problem with objective functions and constraints
- `nodesAssignment::Dict{String, Int64}`: Dictionary mapping node IDs to partition assignments
  - Key: node ID from the MultiblockGraph
  - Value: 0 for left partition, 1 for right partition
- `edgesSplitting::Dict{String, Tuple{Int64, Int64}}`: Dictionary mapping edge IDs to splitting decisions
  - Key: edge ID from the MultiblockGraph
  - Value: (split_flag, partition_assignment) where:
    - split_flag: 0 = keep edge intact, 1 = split edge
    - partition_assignment: 0 = assign new auxiliary node to left, 1 = assign to right

# Returns
- A new ADMMBipartiteGraph with bipartite structure and all constraints preserved

# Algorithm Overview
1. **Index Mapping**: Create efficient mappings from block/constraint IDs to array indices
2. **Node Creation**: 
   - Transform variable blocks into ADMM nodes with original functions
   - Create constraint nodes for multi-block constraints with IndicatorSumOfNVariables
3. **Edge Processing**: For each original edge:
   - **No Split**: Create direct ADMM edge with appropriate mappings
   - **Split**: Create auxiliary node and replace original edge with two new edges
4. **Mapping Construction**: Set up linear mappings for each edge based on constraint structure
5. **Bipartite Validation**: Verify that no edges connect nodes within the same partition
6. **Partition Assignment**: Populate left and right partition vectors

# Node Creation Details
- **Variable Nodes**: Inherit f, g functions and initial values from original blocks
- **Constraint Nodes**: Use Zero() for f and IndicatorSumOfNVariables for g
- **Split Nodes**: Created with specific function configurations depending on split type

# Edge Splitting Cases
1. **TWO_BLOCK_EDGE Split**: Original constraint A₁x₁ + A₂x₂ = b becomes:
   - A₁x₁ - z₁ = 0 (edge from x₁ to auxiliary node)
   - A₂x₂ - z₂ = 0 (edge from x₂ to auxiliary node) 
   - z₁ + z₂ = b (constraint on auxiliary node)

2. **MULTIBLOCK_EDGE Split**: Original connection Aᵢxᵢ - zⱼ = 0 becomes:
   - Aᵢxᵢ - w = 0 (edge from xᵢ to auxiliary node)
   - w - zⱼ = 0 (edge from auxiliary node to constraint node)

# Mathematical Preservation
- All original constraints are preserved through edge representations
- Splitting maintains mathematical equivalence while ensuring bipartite structure
- Linear mappings correctly represent coefficient matrices from original problem
- Right-hand sides are properly distributed across split constraints

# Error Handling
- Validates bipartite property: ensures no edges connect nodes in same partition
- Checks block/constraint index consistency
- Verifies constraint structure matches graph representation

# Examples
```julia
# Typical usage after bipartization algorithm
graph = MultiblockGraph(mbp)
nodesAssignment, edgesSplitting = apply_bipartization_algorithm(graph, mbp)
admm_graph = ADMMBipartiteGraph(graph, mbp, nodesAssignment, edgesSplitting)

# Example assignment and splitting dictionaries
nodesAssignment = Dict(
    "VariableNode(Block1)" => 0,     # Left partition
    "VariableNode(Block2)" => 1,     # Right partition
    "ConstraintNode(Constr1)" => 1   # Right partition
)

edgesSplitting = Dict(
    "TwoBlockEdge(Constr1)" => (0, 0),     # Keep intact
    "MultiblockEdge(Constr2, Block1)" => (1, 1)  # Split, aux node to right
)
```

# Performance Notes
- Time complexity: O(V + E) where V = nodes, E = edges
- Space complexity: O(V + E) for the resulting bipartite graph
- Efficient index mappings minimize lookup overhead
- Sparse matrix operations used where appropriate

# Related Functions
- `ADMMBipartiteGraph(graph, mbp, algorithm)`: Higher-level constructor using bipartization algorithms
- Bipartization algorithms: `MilpBipartization`, `BfsBipartization`, etc.
"""
function ADMMBipartiteGraph(graph::MultiblockGraph, 
    mbp::MultiblockProblem, 
    nodesAssignment::Dict{String, Int64},              # indicates which partition the node belongs to; 0 for left, 1 for right
    edgesSplitting::Dict{String, Tuple{Int64, Int64}}, # (a,b) indicates how an edge is splitted; a=0 means no splitting; 
    partitionAlgorithmTime::Float64)                   # partitionAlgorithmTime is the time taken by the partition algorithm
    
    admmGraph = ADMMBipartiteGraph()
    admmGraph.partitionAlgorithmTime = partitionAlgorithmTime
    
    # create a mapping from block ID to block index in mbp.blocks
    blockID2Index = Dict{BlockID, Int64}()
    numberBlocks = length(mbp.blocks)
    for idx in 1:numberBlocks 
        blockID2Index[mbp.blocks[idx].id] = idx
    end 

    # create a mapping from constraint ID to constraint index in mbp.constraints
    constraintID2Index = Dict{BlockID, Int64}() 
    numberConstraints = length(mbp.constraints)
    for idx in 1:numberConstraints 
        constraintID2Index[mbp.constraints[idx].id] = idx
    end 

    # helper function to create an initial variable for an IndicatorSumOfNVariables instance
    function initialValueSumOfNVariables(numberVariables::Int64, rhs::NumericVariable)
        if isa(rhs, Number)
            return spzeros(numberVariables)
        else 
            dims = size(rhs)
            newDims = (dims[1] * numberVariables, dims[2:end]...)
            if length(newDims) <= 2
                return spzeros(newDims)
            else 
                return zeros(newDims)
            end
        end 
    end 

    # introduce an ADMM node for each variable node and each multiblock edge in MultiblockGraph
    for (nodeID, node) in graph.nodes 
        if node.type == VARIABLE_NODE 
            blockID = node.source 
            idx = blockID2Index[blockID]
            admmGraph.nodes[nodeID] = ADMMNode(
                mbp.blocks[idx].f, 
                mbp.blocks[idx].g, 
                deepcopy(mbp.blocks[idx].val),
                Set{String}(), 
                "", 
                nodesAssignment[nodeID])
            admmGraph.mbpBlockID2admmNodeID[blockID] = nodeID
        else 
            constrID = node.source 
            idx = constraintID2Index[constrID]
            numberInvolvedBlocks = length(mbp.constraints[idx].involvedBlocks)
            admmGraph.nodes[nodeID] = ADMMNode(
                Zero(), 
                IndicatorSumOfNVariables(numberInvolvedBlocks, mbp.constraints[idx].rhs), 
                initialValueSumOfNVariables(numberInvolvedBlocks, mbp.constraints[idx].rhs), 
                Set{String}(), 
                "", 
                nodesAssignment[nodeID])
        end 
    end 
    
    for (edgeID, edge) in graph.edges 
        constrID = edge.sourceBlockConstraint 
        constrIdx = constraintID2Index[constrID]

        if edgesSplitting[edgeID][1] == 0 
            if edge.type == TWO_BLOCK_EDGE
                nodeID1 = edge.nodeID1 
                nodeID2 = edge.nodeID2 

                blockID1 = graph.nodes[nodeID1].source 
                blockID2 = graph.nodes[nodeID2].source

                mappings = Dict{String, AbstractMapping}() 
                mappings[nodeID1] = mbp.constraints[constrIdx].mappings[blockID1]
                mappings[nodeID2] = mbp.constraints[constrIdx].mappings[blockID2]
                
                admmGraph.edges[edgeID] = ADMMEdge(
                    nodeID1, 
                    nodeID2,
                    mappings, 
                    mbp.constraints[constrIdx].rhs, 
                    "")
                
                push!(admmGraph.nodes[nodeID1].neighbors, edgeID)
                push!(admmGraph.nodes[nodeID2].neighbors, edgeID)
            else 
                nodeID1 = edge.nodeID1 
                nodeID2 = edge.nodeID2  # this is a constraint node 
                
                blockID = edge.sourceBlockVariable
                @assert(blockID == graph.nodes[nodeID1].source)

                blockPosInConstr = findfirst(isequal(blockID), mbp.constraints[constrIdx].involvedBlocks)
                @assert(blockPosInConstr != nothing, "ADMMBipartiteGraph: block $blockID not found in constraint $constrID")

                # A_ix_i - z_j = 0, where j = blockPosInConstr 
                mappings = Dict{String, AbstractMapping}() 
                mappings[nodeID1] = mbp.constraints[constrIdx].mappings[blockID]
                mappings[nodeID2] = LinearMappingExtraction(size(admmGraph.nodes[nodeID2].val), -1.0, 
                    (blockPosInConstr - 1) * size(mbp.constraints[constrIdx].rhs, 1) + 1, # start index of the block in the constraint
                    blockPosInConstr * size(mbp.constraints[constrIdx].rhs, 1)            # end index of the block in the constraint
                )
                
                admmGraph.edges[edgeID] = ADMMEdge(
                    nodeID1, 
                    nodeID2, 
                    mappings, 
                    zero(mbp.constraints[constrIdx].rhs),  
                    "")
                
                push!(admmGraph.nodes[nodeID1].neighbors, edgeID)
                push!(admmGraph.nodes[nodeID2].neighbors, edgeID)
            end 
        else 
            # add two edges 
            if edge.type == TWO_BLOCK_EDGE 
                # create a new aux node 
                newNodeID = createADMMNodeID(edgeID)
                admmGraph.nodes[newNodeID] = ADMMNode( 
                    Zero(), 
                    Zero(),
                    zero(mbp.constraints[constrIdx].rhs), 
                    Set{String}(), 
                    edgeID, 
                    edgesSplitting[edgeID][2])

                # add two new edges 
                nodeID1 = edge.nodeID1 
                nodeID2 = edge.nodeID2  

                blockID1 = graph.nodes[nodeID1].source 
                blockID2 = graph.nodes[nodeID2].source

                # A_ix_i - z = 0 
                newEdgeID1 = createADMMEdgeID(edgeID, nodeID1)
                mappings1 = Dict{String, AbstractMapping}()
                mappings1[nodeID1] = mbp.constraints[constrIdx].mappings[blockID1]
                mappings1[newNodeID] = LinearMappingIdentity(-1.0)

                admmGraph.edges[newEdgeID1] = ADMMEdge(
                    nodeID1, 
                    newNodeID, 
                    mappings1, 
                    zero(mbp.constraints[constrIdx].rhs), 
                    edgeID)

                push!(admmGraph.nodes[nodeID1].neighbors, newEdgeID1)
                push!(admmGraph.nodes[newNodeID].neighbors, newEdgeID1)
                
                # A_jx_j + z = b
                newEdgeID2 = createADMMEdgeID(edgeID, nodeID2)
                mappings2 = Dict{String, AbstractMapping}()
                mappings2[nodeID2] = mbp.constraints[constrIdx].mappings[blockID2]
                mappings2[newNodeID] = LinearMappingIdentity(1.0)

                admmGraph.edges[newEdgeID2] = ADMMEdge(
                    nodeID2, 
                    newNodeID, 
                    mappings2, 
                    mbp.constraints[constrIdx].rhs, 
                    edgeID)

                push!(admmGraph.nodes[nodeID2].neighbors, newEdgeID2)
                push!(admmGraph.nodes[newNodeID].neighbors, newEdgeID2)

            else   
                # create a new node; this is a aux node simply to break odd cycle 
                newNodeID = createADMMNodeID(edgeID)
                admmGraph.nodes[newNodeID] = ADMMNode( 
                    Zero(), 
                    Zero(), 
                    zero(mbp.constraints[constrIdx].rhs), 
                    Set{String}(),  
                    edgeID,  
                    edgesSplitting[edgeID][2])

                nodeID1 = edge.nodeID1 
                nodeID2 = edge.nodeID2 # this is a constriant node 

                blockID1 = graph.nodes[nodeID1].source 
                @assert(constrID == graph.nodes[nodeID2].source)

                newEdgeID1 = createADMMEdgeID(edgeID, nodeID1)
                mappings1 = Dict{String, AbstractMapping}()
                mappings1[nodeID1] = mbp.constraints[constrIdx].mappings[blockID1]
                mappings1[newNodeID] = LinearMappingIdentity(-1.0)

                admmGraph.edges[newEdgeID1] = ADMMEdge(
                    nodeID1, 
                    newNodeID, 
                    mappings1, 
                    zero(mbp.constraints[constrIdx].rhs), 
                    edgeID)

                push!(admmGraph.nodes[nodeID1].neighbors, newEdgeID1)
                push!(admmGraph.nodes[newNodeID].neighbors, newEdgeID1)

                blockPosInConstr = findfirst(isequal(blockID1), mbp.constraints[constrIdx].involvedBlocks)
                @assert(blockPosInConstr != nothing, "ADMMBipartiteGraph: block $blockID1 not found in constraint $constrID")

                newEdgeID2 = createADMMEdgeID(edgeID, nodeID2)
                mappings2 = Dict{String, AbstractMapping}() 
                mappings2[nodeID2] = LinearMappingExtraction(size(admmGraph.nodes[nodeID2].val), -1.0, 
                    (blockPosInConstr - 1) * size(mbp.constraints[constrIdx].rhs, 1) + 1, 
                    blockPosInConstr * size(mbp.constraints[constrIdx].rhs, 1))
                mappings2[newNodeID] = LinearMappingIdentity(1.0)
        
                admmGraph.edges[newEdgeID2] = ADMMEdge( 
                   nodeID2, 
                   newNodeID, 
                   mappings2, 
                   zero(mbp.constraints[constrIdx].rhs), 
                   edgeID)
                
                push!(admmGraph.nodes[nodeID2].neighbors, newEdgeID2)
                push!(admmGraph.nodes[newNodeID].neighbors, newEdgeID2)
            end 
        end 
    end 

    # check if the ADMM graph is bipartite
    for (edgeID, edge) in admmGraph.edges  
        nodeID1 = edge.nodeID1 
        nodeID2 = edge.nodeID2  
        if admmGraph.nodes[nodeID1].assignment == admmGraph.nodes[nodeID2].assignment 
            error("ADMMBipartiteGraph: The ADMM graph is not bipartite")
        end 
    end 

    # partition the nodes into left and right
    for (nodeID, node) in admmGraph.nodes 
        if node.assignment < 0.5 
            push!(admmGraph.left, nodeID)
        else 
            push!(admmGraph.right, nodeID)
        end 
    end

    return admmGraph 
end 

"""
    ADMMBipartiteGraph(graph::MultiblockGraph, mbp::MultiblockProblem, 
                      algorithm::BipartizationAlgorithm) -> ADMMBipartiteGraph

Construct an ADMM bipartite graph by automatically applying a bipartization algorithm to a multiblock graph.

This high-level constructor provides a convenient interface for creating ADMM bipartite graphs by
automatically handling the bipartization process. It selects and applies the specified algorithm,
handles edge splitting decisions, and constructs the final bipartite representation.

**Arguments**
- `graph::MultiblockGraph`: The original multiblock graph (may or may not be bipartite)
- `mbp::MultiblockProblem`: The original multiblock optimization problem
- `algorithm::BipartizationAlgorithm`: The bipartization algorithm to apply, one of:
  - `MILP_BIPARTIZATION`: Optimal MILP-based approach (slower but higher quality)
  - `BFS_BIPARTIZATION`: Fast BFS-based heuristic
  - `DFS_BIPARTIZATION`: Fast DFS-based heuristic  
  - `SPANNING_TREE_BIPARTIZATION`: Balanced spanning tree approach

**Returns**
- A new ADMMBipartiteGraph with proper bipartite structure suitable for ADMM decomposition

**Algorithm Selection Strategy**
- **Already Bipartite**: If the input graph is already bipartite, skips bipartization entirely
- **Performance Optimization**: Reports timing information for algorithm performance analysis
- **Error Handling**: Validates algorithm choice and provides meaningful error messages

**Workflow**
1. **Bipartite Check**: Test if the graph is already bipartite using existing coloring
2. **Algorithm Application**: If not bipartite, apply the selected bipartization algorithm
3. **Performance Monitoring**: Measure and report algorithm execution time
4. **Graph Construction**: Use the core constructor to build the final ADMM bipartite graph

**Algorithm Characteristics**
- **MILP_BIPARTIZATION**: 
  - Pros: Optimal solution considering operator norms
  - Cons: Slower, requires MILP solver
  - Best for: Small to medium problems where optimality is important
  
- **BFS_BIPARTIZATION**:
  - Pros: Fast, simple, handles disconnected graphs
  - Cons: May create more splits than necessary
  - Best for: Large problems where speed is critical
  
- **DFS_BIPARTIZATION**:
  - Pros: Fast, different traversal pattern than BFS
  - Cons: May create more splits than necessary
  - Best for: Alternative to BFS, may work better for certain graph structures
  
- **SPANNING_TREE_BIPARTIZATION**:
  - Pros: Balanced approach, fewer unnecessary splits
  - Cons: More complex than BFS/DFS
  - Best for: Good compromise between quality and speed

**Examples**
Create ADMM graph with MILP optimization:
```julia
mbp = MultiblockProblem()
graph = MultiblockGraph(mbp)
admm_graph = ADMMBipartiteGraph(graph, mbp, MILP_BIPARTIZATION)
```

Fast heuristic approach for large problems:
```julia
admm_graph_fast = ADMMBipartiteGraph(graph, mbp, BFS_BIPARTIZATION)
```

Balanced approach:
```julia
admm_graph_balanced = ADMMBipartiteGraph(graph, mbp, SPANNING_TREE_BIPARTIZATION)
```

**Performance Considerations**
- **Already Bipartite**: O(1) if graph is already bipartite (just copies coloring)
- **Bipartization Required**: Depends on chosen algorithm
  - MILP: Can be expensive for large graphs
  - BFS/DFS: O(V + E) linear time
  - Spanning Tree: O(V + E) linear time
- **Memory**: Additional memory for node assignments and edge splitting decisions

**Output Information**
- Logs whether bipartization was skipped (graph already bipartite)
- Reports algorithm name and execution time
- Enables performance profiling and algorithm comparison

**Error Handling**
- Validates that the algorithm enum value is recognized
- Ensures the resulting graph is truly bipartite
- Provides meaningful error messages for debugging

**Related Functions**
- `ADMMBipartiteGraph(graph, mbp, nodesAssignment, edgesSplitting)`: Core constructor
- `getBipartizationAlgorithmName`: Get human-readable algorithm names
- Bipartization algorithms: `MilpBipartization`, `BfsBipartization`, etc.
"""
function ADMMBipartiteGraph(graph::MultiblockGraph, 
    mbp::MultiblockProblem, 
    algorithm::BipartizationAlgorithm, 
    logLevel::Int64=1; 
    mipRelGap::Float64=0.01,
    mipTimeLimit::Float64=60.0,
    mipHeuristicEffort::Float64=0.2)
    if graph.isBipartite
        @PDMOInfo logLevel "ADMMBipartiteGraph: The graph is already bipartite; skip bipartization algorithm."
        edgesSplitting = Dict{String, Tuple{Int64, Int64}}(edgeID=>(0,0) for edgeID in keys(graph.edges))
        return ADMMBipartiteGraph(graph, mbp, graph.colors, edgesSplitting, 0.0)
    end 

    nodesAssignment = Dict{String, Int64}() 
    edgesSplitting = Dict{String, Tuple{Int64, Int64}}()

    timeStart = time()
    if algorithm == MILP_BIPARTIZATION 
        try
            MilpBipartization(graph, mbp, nodesAssignment, edgesSplitting; mipRelGap=mipRelGap, mipTimeLimit=mipTimeLimit, mipHeuristicEffort=mipHeuristicEffort) 
        catch e
            @PDMOError logLevel "ADMMBipartiteGraph: MILP bipartization failed. Error: $e. Use BFS bipartization instead."
            # @PDMOInfo logLevel "ADMMBipartiteGraph:  MILP bipartization failed. Use BFS bipartization instead."
            empty!(nodesAssignment)
            empty!(edgesSplitting)
            BfsBipartization(graph, mbp, nodesAssignment, edgesSplitting)
        end 
    elseif algorithm == BFS_BIPARTIZATION 
        BfsBipartization(graph, mbp, nodesAssignment, edgesSplitting)
    elseif algorithm == DFS_BIPARTIZATION 
        DfsBipartization(graph, mbp, nodesAssignment, edgesSplitting)
    elseif algorithm == SPANNING_TREE_BIPARTIZATION 
        SpanningTreeBipartization(graph, mbp, nodesAssignment, edgesSplitting)
    elseif algorithm == GNN_BIPARTIZATION
        GnnBipartization(graph, mbp, nodesAssignment, edgesSplitting)
    else 
        error("ADMMBipartiteGraph: Invalid bipartization algorithm")
    end 

    partitionAlgorithmTime = time() - timeStart
    msg = Printf.@sprintf("ADMMBipartiteGraph: %s took %.2f seconds. \n", 
        getBipartizationAlgorithmName(algorithm),  
        partitionAlgorithmTime) 
    @PDMOInfo logLevel msg 

    return ADMMBipartiteGraph(graph, mbp, nodesAssignment, edgesSplitting, partitionAlgorithmTime)
end

"""
    summary(admmGraph::ADMMBipartiteGraph)

Prints a comprehensive summary of the ADMM bipartite graph's structural properties and statistics.

This function provides essential information about the bipartite graph structure that is useful for
understanding the problem decomposition, algorithm performance analysis, and debugging ADMM implementations.

**Arguments**
- `admmGraph::ADMMBipartiteGraph`: The ADMM bipartite graph to analyze and summarize

**Output Information**
The function prints the following statistics to standard output:
- **Total Nodes**: Number of nodes in the bipartite graph (original + auxiliary)
- **Partition Sizes**: Number of nodes in left and right partitions
- **Total Edges**: Number of constraint edges in the bipartite graph
- **Partition Balance**: Ratio between left and right partition sizes

**Example Output**
```
Summary of ADMM Bipartitie Graph:
    Number of nodes             = 8
    Parition size (left, right) = (3, 5)
    Number of edges             = 12
```

**Analysis Value**
- **Problem Scale**: Total nodes and edges indicate computational complexity
- **ADMM Balance**: Partition sizes affect ADMM update step efficiency
- **Decomposition Quality**: Edge count relative to original problem shows splitting overhead
- **Memory Requirements**: Node and edge counts determine memory usage

**Usage Scenarios**
1. **Algorithm Comparison**: Compare different bipartization algorithms
2. **Problem Analysis**: Understand decomposition characteristics  
3. **Performance Monitoring**: Track graph properties across problem instances

**Interpretation Guidelines**
- **Balanced Partitions**: Similar left/right sizes often lead to better ADMM performance
- **Edge Density**: High edge count relative to nodes may indicate complex coupling
- **Split Overhead**: Compare edge count to original problem to assess bipartization cost
- **Scalability**: Large node/edge counts may require algorithm parameter tuning

# Implementation Notes
- Uses `@info` macro for consistent logging format
- Accesses graph fields directly for O(1) performance
- Prints to standard output for immediate visibility
- Compatible with logging redirection and capture

# Related Functions
- `summary(graph::MultiblockGraph)`: Summary of original graph before bipartization
- `summary(mbp::MultiblockProblem)`: Summary of original optimization problem
- `numberNodes`, `numberEdges`: Access individual statistics

# Performance
- **Time Complexity**: O(1) - only accesses pre-computed field lengths
- **Space Complexity**: O(1) - no additional memory allocation
- **Output**: Minimal console output suitable for logging and analysis
"""
function summary(admmGraph::ADMMBipartiteGraph, logLevel::Int64=1)
    if logLevel < 1
        return 
    end 
    @PDMOInfo logLevel "Summary of ADMM Bipartitie Graph:"
    println("    Number of nodes             = $(length(admmGraph.nodes))")
    println("    Parition size (left, right) = ($(length(admmGraph.left)), $(length(admmGraph.right)))")
    println("    Number of edges             = $(length(admmGraph.edges))")
end 

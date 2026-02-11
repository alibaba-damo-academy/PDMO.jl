"""
    ADMMNode

A node in the ADMM bipartite graph representation for the Alternating Direction Method of Multipliers.

ADMMNode represents either an original variable block from the multiblock problem or an auxiliary node
created from edge splitting during bipartization. Each node contains optimization functions and 
maintains connectivity information for the ADMM algorithm.

# Fields
- `f::AbstractFunction`: Primary objective function component (smooth part of the objective)
- `g::AbstractFunction`: Secondary objective function component (non-smooth part, often regularization or constraint indicators)
- `val::NumericVariable`: Current value/estimate of the variable associated with this node
- `neighbors::Set{String}`: Set of edge IDs connected to this node in the bipartite graph
- `convertedEdgeID::String`: Original edge ID from MultiblockGraph if this node was created by splitting an edge; empty string for original variable nodes
- `assignment::Int`: Partition assignment in the bipartite graph (0 for left partition, 1 for right partition)

# Node Types
1. **Original Variable Nodes**: Represent variable blocks from the original MultiblockProblem
   - Have non-trivial `f` and `g` functions from the original block
   - `convertedEdgeID` is empty string
   - `val` initialized from original block value

2. **Constraint Nodes**: Represent multi-block constraints from the original problem
   - Have `f = Zero()` and `g = IndicatorSumOfNVariables(...)`
   - Created for constraints involving more than 2 blocks
   - `convertedEdgeID` is empty string

3. **Split Edge Nodes**: Auxiliary nodes created during bipartization
   - Created when edges need to be split to maintain bipartite structure
   - Have specific function configurations depending on split type
   - `convertedEdgeID` contains the original edge ID

# Constructor
    ADMMNode(f, g, val, neighbors, convertedEdgeID, assignment)

# Usage in ADMM
- **Left Partition**: Typically contains variable nodes for x-update step
- **Right Partition**: Typically contains constraint nodes and auxiliary nodes for z-update step
- **Functions**: Used in proximal operators during ADMM iterations
- **Connectivity**: Determines the coupling structure in ADMM decomposition

# Examples
```julia
# Original variable node
node = ADMMNode(QuadraticFunction(...), IndicatorBox(...), x0, Set{String}(), "", 0)

# Constraint node  
node = ADMMNode(Zero(), IndicatorSumOfNVariables(...), z0, Set{String}(), "", 1)

# Split edge auxiliary node
node = ADMMNode(Zero(), IndicatorSumOfNVariables(2, rhs), aux_val, Set{String}(), "EdgeID123", 1)
```
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

An edge in the ADMM bipartite graph connecting two nodes and representing a constraint in the ADMM decomposition.

ADMMEdge represents a linear constraint between two nodes in the bipartite graph. Each edge corresponds
to either an original constraint from the multiblock problem or a constraint created during edge splitting
for bipartization. The edge defines the coupling between variables in the ADMM algorithm.

# Fields
- `nodeID1::String`: ID of the first node connected by this edge (typically from left partition)
- `nodeID2::String`: ID of the second node connected by this edge (typically from right partition)
- `mappings::Dict{String, AbstractMapping}`: Linear mappings for each node involved in the constraint
  - Key: node ID, Value: linear mapping applied to that node's variable
  - Represents the coefficient matrices in the linear constraint
- `rhs::NumericVariable`: Right-hand side vector/value of the constraint represented by this edge
- `splittedEdgeID::String`: Original edge ID from MultiblockGraph if this edge was created by splitting; empty string for original edges

# Edge Types
1. **Original Two-Block Edges**: Direct constraints between two variable blocks
   - Represent constraints of the form `A₁x₁ + A₂x₂ = b`
   - Connect two original variable nodes
   - `splittedEdgeID` is empty string

2. **Multi-Block Edges**: Constraints involving constraint nodes
   - Represent constraints of the form `Aᵢxᵢ - zⱼ = 0`
   - Connect a variable node to a constraint node
   - `splittedEdgeID` is empty string

3. **Split Edges**: Edges created from splitting original edges
   - Created during bipartization to maintain bipartite structure
   - Connect original nodes to auxiliary split nodes
   - `splittedEdgeID` contains the original edge ID

# Mathematical Representation
Each edge represents a constraint: `mapping[nodeID1] * x₁ + mapping[nodeID2] * x₂ = rhs`
- For node i with variable xᵢ, the constraint contribution is `mappings[nodeIDᵢ](xᵢ)`
- The complete constraint equation must equal the `rhs` value

# Constructor
    ADMMEdge(nodeID1, nodeID2, mappings, rhs, splittedEdgeID)

# Usage in ADMM
- **Constraint Coupling**: Defines how variables are coupled in the optimization problem
- **Dual Variables**: Each edge corresponds to dual variables (Lagrange multipliers) in ADMM
- **Update Steps**: Used in both primal and dual update steps of the ADMM algorithm
- **Convergence**: Residuals computed using these constraint definitions

# Examples
```julia
# Original two-block constraint: A₁x₁ + A₂x₂ = b
mappings = Dict("node1" => LinearMappingMatrix(A1), "node2" => LinearMappingMatrix(A2))
edge = ADMMEdge("node1", "node2", mappings, b, "")

# Multi-block constraint connection: Aᵢxᵢ - zⱼ = 0
mappings = Dict("var_node" => LinearMappingMatrix(A), "constr_node" => LinearMappingExtraction(...))
edge = ADMMEdge("var_node", "constr_node", mappings, zeros(m), "")

# Split edge constraint: Aᵢxᵢ - z₁ = 0 (from splitting)
mappings = Dict("original_node" => LinearMappingMatrix(A), "split_node" => LinearMappingExtraction(...))
edge = ADMMEdge("original_node", "split_node", mappings, zeros(m), "OriginalEdgeID")
```
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

Generate a unique node ID for an auxiliary node created from splitting an edge during bipartization.

When an edge needs to be split to maintain bipartite structure, this function creates a consistent
naming scheme for the new auxiliary node that will be inserted at the split point.

# Arguments
- `edgeID::String`: The original edge ID from the MultiblockGraph that is being split

# Returns
- A string ID for the new ADMM auxiliary node following the pattern "ADMMNodeConvertedFromEdge(edgeID)"

# Usage
This function is used internally during the construction of ADMMBipartiteGraph when:
- A TWO_BLOCK_EDGE needs to be split due to bipartization decisions
- A MULTIBLOCK_EDGE needs to be split to maintain bipartite structure
- The bipartization algorithm determines that an edge violates bipartiteness

# Examples
```julia
original_edge_id = "TwoBlockEdge(Constraint1)"
new_node_id = createADMMNodeID(original_edge_id)
# Returns: "ADMMNodeConvertedFromEdge(TwoBlockEdge(Constraint1))"

multiblock_edge_id = "MultiblockEdge(Constraint2, Block1)"  
aux_node_id = createADMMNodeID(multiblock_edge_id)
# Returns: "ADMMNodeConvertedFromEdge(MultiblockEdge(Constraint2, Block1))"
```

# Implementation Notes
- The generated ID uniquely identifies the auxiliary node
- The ID preserves traceability back to the original edge
- Used consistently across edge splitting operations
- Prevents ID conflicts with original variable and constraint nodes

# Related Functions
- `createADMMEdgeID`: Creates IDs for new edges connected to split nodes
- `ADMMBipartiteGraph`: Uses this function during edge splitting operations
"""
function createADMMNodeID(edgeID::String)
    return "ADMMNodeConvertedFromEdge($edgeID)"
end

"""
    createADMMEdgeID(edgeID::String, nodeID::String) -> String

Generate a unique edge ID for a new edge created when splitting an original edge during bipartization.

When an original edge is split to maintain bipartite structure, it is replaced by two or more new edges
that connect through an auxiliary node. This function creates consistent IDs for these new edges.

# Arguments
- `edgeID::String`: The original edge ID from the MultiblockGraph that was split
- `nodeID::String`: The ID of the node that this new edge connects to (either original node or auxiliary node)

# Returns
- A string ID for the new ADMM edge following the pattern "ADMMEdgeSplittedFrom(edgeID, nodeID)"

# Usage
This function is used internally during edge splitting operations when:
- An original TWO_BLOCK_EDGE is split into two edges through an auxiliary node
- A MULTIBLOCK_EDGE is split to resolve bipartite violations
- Multiple new edges need to be created with consistent, traceable naming

# Edge Splitting Scenarios
1. **TWO_BLOCK_EDGE Split**: Original edge (node1, node2) becomes:
   - Edge1: (node1, auxNode) with ID "ADMMEdgeSplittedFrom(originalEdgeID, node1)"
   - Edge2: (node2, auxNode) with ID "ADMMEdgeSplittedFrom(originalEdgeID, node2)"

2. **MULTIBLOCK_EDGE Split**: Original edge (varNode, constrNode) becomes:
   - Edge1: (varNode, auxNode) with ID "ADMMEdgeSplittedFrom(originalEdgeID, varNode)"
   - Edge2: (constrNode, auxNode) with ID "ADMMEdgeSplittedFrom(originalEdgeID, constrNode)"

# Examples
```julia
# Splitting a two-block edge
original_edge = "TwoBlockEdge(Constraint1)"
edge1_id = createADMMEdgeID(original_edge, "VariableNode(Block1)")
# Returns: "ADMMEdgeSplittedFrom(TwoBlockEdge(Constraint1), VariableNode(Block1))"

edge2_id = createADMMEdgeID(original_edge, "VariableNode(Block2)")  
# Returns: "ADMMEdgeSplittedFrom(TwoBlockEdge(Constraint1), VariableNode(Block2))"

# Connection to auxiliary node
aux_node_id = "ADMMNodeConvertedFromEdge(TwoBlockEdge(Constraint1))"
aux_edge_id = createADMMEdgeID(original_edge, aux_node_id)
# Returns: "ADMMEdgeSplittedFrom(TwoBlockEdge(Constraint1), ADMMNodeConvertedFromEdge(...))"
```

# Implementation Notes
- Preserves full traceability back to the original edge and connected node
- Ensures unique IDs even when multiple edges are created from the same original edge
- Used consistently across all edge splitting operations
- Enables reconstruction of the splitting history for debugging and analysis

# Related Functions
- `createADMMNodeID`: Creates IDs for auxiliary nodes in edge splits
- `ADMMBipartiteGraph`: Uses this function extensively during bipartization
"""
function createADMMEdgeID(edgeID::String, nodeID::String)
    return "ADMMEdgeSplittedFrom($edgeID, $nodeID)"
end

"""
    ADMMBipartiteGraph

A bipartite graph representation specifically designed for the Alternating Direction Method of Multipliers (ADMM) algorithm.

This structure transforms a general multiblock optimization problem into a bipartite graph that enables
efficient ADMM decomposition. The bipartite structure ensures that variables can be updated in alternating
fashion between the two partitions, which is essential for ADMM convergence properties.

# Fields
- `nodes::Dict{String, ADMMNode}`: Dictionary mapping node IDs to ADMMNode objects
  - Contains original variable nodes, constraint nodes, and auxiliary split nodes
  - Each node has associated optimization functions and partition assignment
- `edges::Dict{String, ADMMEdge}`: Dictionary mapping edge IDs to ADMMEdge objects
  - Represents linear constraints between nodes in the bipartite graph
  - Each edge defines coupling relationships for ADMM algorithm
- `mbpBlockID2admmNodeID::Dict{BlockID, String}`: Mapping from original MultiblockProblem block IDs to ADMM node IDs
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

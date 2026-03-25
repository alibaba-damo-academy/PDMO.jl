"""
    MultiblockGraph

A graph representation of a `MultiblockProblem`, with utilities to check
bipartiteness and connectivity.
"""

"""
    Graph Model for MultiblockProblem

Represents a graph structure for modeling multi-block optimization problems.
Contains nodes representing variables and constraints, and edges representing 
relationships between them.
"""

"""
    NodeType

Node category in `MultiblockGraph`.
"""
@enum NodeType begin 
    VARIABLE_NODE     # Node representing a variable block
    CONSTRAINT_NODE   # Node representing a constraint
end 

"""
    Node

A node in the multi-block graph representing either a variable block or a constraint.

# Fields
- `neighbors::Vector{String}`: List of edge IDs connected to this node
- `type::NodeType`: Type of node (VARIABLE_NODE or CONSTRAINT_NODE)
- `source::BlockID`: ID of the original variable block or constraint from the MultiblockProblem

# Constructor
    Node(neighbors::Vector{String}, type::NodeType, source::BlockID)

# Usage
Nodes are automatically created when constructing a MultiblockGraph from a MultiblockProblem.
Variable nodes are created for each block variable, and constraint nodes are created for
constraints involving more than two blocks.
"""
mutable struct Node 
    neighbors::Vector{String}    # Edge IDs of connected edges
    type::NodeType               # Type of node (variable or constraint)
    source::BlockID              # Variable or constraint index from original problem
end 


"""
    EdgeType

Edge category in `MultiblockGraph`.
"""
@enum EdgeType begin 
    TWO_BLOCK_EDGE     # Edge connecting exactly two variable nodes
    MULTIBLOCK_EDGE    # Edge connecting a variable node to a constraint node
end 

"""
    Edge

An edge in the multi-block graph connecting two nodes.

# Fields
- `nodeID1::String`: ID of the first node connected by this edge
- `nodeID2::String`: ID of the second node connected by this edge (constraint node for MULTIBLOCK_EDGE)
- `type::EdgeType`: Type of edge (TWO_BLOCK_EDGE or MULTIBLOCK_EDGE)
- `sourceBlockConstraint::BlockID`: ID of the constraint from the original problem that this edge represents
- `sourceBlockVariable::BlockID`: For MULTIBLOCK_EDGE, the variable block ID; empty string for TWO_BLOCK_EDGE

# Constructor
    Edge(nodeID1::String, nodeID2::String, type::EdgeType, sourceBlockConstraint::BlockID, sourceBlockVariable::BlockID)

# Usage
Edges are automatically created when constructing a MultiblockGraph from a MultiblockProblem.
- TWO_BLOCK_EDGE connects two variable nodes directly (constraint involves exactly 2 blocks)
- MULTIBLOCK_EDGE connects a variable node to a constraint node (constraint involves >2 blocks)
"""
mutable struct Edge 
    nodeID1::String     # ID of first node
    nodeID2::String     # ID of second node; if multiblock edge, this is the constraint node
    type::EdgeType      # Type of edge
    sourceBlockConstraint::BlockID    # Constraint ID from original problem (= nodes[nodeID2].source)
    sourceBlockVariable::BlockID      # if multiblock edge, variable ID from original problem (= nodes[nodeID1].source); "" otherwise
end 

"""
    createNodeID(id::BlockID; isConstraint::Bool=false) -> String

Create a stable node ID string from a block/constraint ID.

# Arguments
- `id::BlockID`: The block or constraint ID from the original MultiblockProblem
- `isConstraint::Bool=false`: If true, generates a constraint node ID; otherwise generates a variable node ID

# Returns
- `String`: `VariableNode(<id>)` or `ConstraintNode(<id>)`.
"""
function createNodeID(id::BlockID; isConstraint::Bool=false)
    if isConstraint
        return "ConstraintNode($id)" # id is the constraint index in the original problem
    else 
        return "VariableNode($id)"   # id is the variable index in the original problem
    end 
end 

"""
    createEdgeID(constrID::BlockID; variableID::BlockID="") -> String

Create a stable edge ID string from constraint/variable IDs.

# Arguments
- `constrID::BlockID`: The constraint ID from the original MultiblockProblem
- `variableID::BlockID=""`: The variable ID (empty for TWO_BLOCK_EDGE, specified for MULTIBLOCK_EDGE)

# Returns
- `String`: `TwoBlockEdge(<constrID>)` if `variableID == ""`, otherwise
  `MultiblockEdge(<constrID>, <variableID>)`.
"""
function createEdgeID(constrID::BlockID; variableID::BlockID="")
    if variableID == ""
        return "TwoBlockEdge($constrID)" # constrID is the constraint index in the original problem
    else 
        return "MultiblockEdge($constrID, $variableID)" #  MultiblockEdge: variableID is the variable index in the original problem
    end 
end 

"""
    MultiblockGraph

Graph representation of a multi-block optimization problem.
Contains nodes representing variables and constraints, connected by edges.

# Fields
- `nodes::Dict{String, Node}`: Dictionary mapping node IDs to Node objects
- `edges::Dict{String, Edge}`: Dictionary mapping edge IDs to Edge objects
- `colors::Dict{String, Int64}`: Node coloring used for bipartiteness testing (0 or 1)
- `isBipartite::Bool`: Whether the graph is bipartite (updated by analysis functions)
- `isConnected::Bool`: Whether the graph is connected (updated by analysis functions)

# Constructors
    MultiblockGraph()
    MultiblockGraph(mbp::MultiblockProblem)

# Graph Structure
The graph represents the constraint structure of a multiblock optimization problem:
- **Variable nodes**: Represent optimization variable blocks
- **Constraint nodes**: Represent constraints involving more than 2 blocks
- **Two-block edges**: Direct connections between variable nodes (2-block constraints)
- **Multi-block edges**: Connections from variable nodes to constraint nodes (>2-block constraints)

# Example
```julia
# Create graph from a multiblock problem
mbp = MultiblockProblem()
# ... add blocks and constraints to mbp ...
graph = MultiblockGraph(mbp)

# Analyze graph properties
is_bipartite = isMultiblockGraphBipartite(graph)
is_connected = isMultiblockGraphConnected(graph)
```

# Usage
This representation is particularly useful for:
- Algorithm selection (e.g., choosing between different decomposition methods)
- Problem analysis (connectivity, bipartiteness)
- Graph-based optimization algorithms like ADMM
"""
mutable struct MultiblockGraph
    nodes::Dict{String, Node}
    edges::Dict{String, Edge}
    colors::Dict{String, Int64}
    isBipartite::Bool
    isConnected::Bool
    # default constructor
    function MultiblockGraph()
        new(Dict{String, Node}(), Dict{String, Edge}(), Dict{String, Int64}(), false, false)
    end 
end 

"""
    getNodelNeighbors(graph::MultiblockGraph) -> Dict{String, Set{String}}

Constructs an adjacency list representation of the graph.

# Arguments
- `graph::MultiblockGraph`: The graph to analyze

# Returns
- A dictionary mapping each node ID to a set of its neighboring node IDs
"""
function getNodelNeighbors(graph::MultiblockGraph) 
    neighbors = Dict{String, Set{String}}(nodeID=>Set{String}() for nodeID in keys(graph.nodes))
    for (edgeID, edge) in graph.edges 
        push!(neighbors[edge.nodeID1], edge.nodeID2)
        push!(neighbors[edge.nodeID2], edge.nodeID1)
    end
    return neighbors
end 

"""
    isMultiblockGraphBipartite(graph::MultiblockGraph) -> Bool

Determines if the graph is bipartite using a breadth-first search coloring algorithm.
Updates the `colors` and `isBipartite` fields in the graph.

# Arguments
- `graph::MultiblockGraph`: The graph to analyze

# Returns
- `true` if the graph is bipartite, `false` otherwise
"""
function isMultiblockGraphBipartite(graph::MultiblockGraph)
    neighbors = getNodelNeighbors(graph)
    empty!(graph.colors)    

    for nodeID in keys(graph.nodes) 
        if haskey(graph.colors, nodeID) == false
            # BFS to color nodes
            graph.colors[nodeID] = 0
            queue = String[nodeID]
            while isempty(queue) == false 
                current = popfirst!(queue)
                currentColor = graph.colors[current]
                # Check all neighbors
                for neighborID in neighbors[current]
                    if haskey(graph.colors, neighborID) == false 
                        # Color neighbor with opposite color
                        graph.colors[neighborID] = 1 - currentColor 
                        push!(queue, neighborID)
                    elseif graph.colors[neighborID] == currentColor
                        # If neighbor has same color, graph is not bipartite
                        graph.isBipartite = false
                        empty!(graph.colors)
                        return graph.isBipartite
                    end
                end
            end
        end
    end
    graph.isBipartite = true
    return graph.isBipartite
end 

"""
    isMultiblockGraphConnected(graph::MultiblockGraph) -> Bool

Determines if the graph is connected using a breadth-first search traversal.
Updates the `isConnected` field in the graph.

# Arguments
- `graph::MultiblockGraph`: The graph to analyze

# Returns
- `true` if the graph is connected, `false` otherwise
"""
function isMultiblockGraphConnected(graph::MultiblockGraph)
    # Handle empty graph case
    if isempty(graph.nodes)
        graph.isConnected = true
        return graph.isConnected
    end

    neighbors = getNodelNeighbors(graph)
    visited = Dict{String, Bool}(nodeID => false for nodeID in keys(graph.nodes))
    
    # Start BFS from first node
    startNode = first(keys(graph.nodes))
    visited[startNode] = true 
    queue = String[startNode]

    while isempty(queue) == false 
        current = popfirst!(queue)

        for neighborID in neighbors[current]
            if visited[neighborID] == false 
                visited[neighborID] = true 
                push!(queue, neighborID)
            end
        end
    end
    graph.isConnected = all(values(visited))
    return graph.isConnected
end

"""
    MultiblockGraph(mbp::MultiblockProblem) -> MultiblockGraph

Constructs a graph representation from a MultiblockProblem.

# Arguments
- `mbp::MultiblockProblem`: The multi-block problem to convert to a graph

# Returns
- A new MultiblockGraph representing the problem structure with bipartiteness and connectivity analyzed

# Algorithm
The construction process follows these steps:
1. **Create variable nodes**: One node for each block variable in the problem
2. **Create constraint nodes**: One node for each constraint involving more than 2 blocks
3. **Create edges**: 
   - TWO_BLOCK_EDGE: Direct edge between two variable nodes (for 2-block constraints)
   - MULTIBLOCK_EDGE: Edge from variable node to constraint node (for >2-block constraints)
4. **Update node neighbors**: Each node maintains a list of connected edge IDs
5. **Analyze properties**: Automatically determines if the graph is bipartite and connected

# Graph Representation Rules
- **2-block constraint**: `A₁x₁ + A₂x₂ = b` → Direct edge between variable nodes
- **Multi-block constraint**: `A₁x₁ + A₂x₂ + A₃x₃ = b` → Constraint node with edges to each variable node

# Example
```julia
mbp = MultiblockProblem()
addBlockVariable!(mbp, BlockVariable("x1", f1, g1, x1_init))
addBlockVariable!(mbp, BlockVariable("x2", f2, g2, x2_init))
addBlockConstraint!(mbp, BlockConstraint("c1", ["x1", "x2"], mappings, rhs))

graph = MultiblockGraph(mbp)
println("Graph has \$(numberNodes(graph)) nodes and \$(numberEdges(graph)) edges")
println("Bipartite: \$(graph.isBipartite), Connected: \$(graph.isConnected)")
```

# Usage
This constructor is the primary way to create a graph representation for analysis and
algorithm selection in multiblock optimization problems.
"""
function MultiblockGraph(mbp::MultiblockProblem)
    graph = MultiblockGraph()
    
    for block in mbp.blocks
        nodeID = createNodeID(block.id)
        graph.nodes[nodeID] = Node(Vector{String}(), VARIABLE_NODE, block.id)
    end 

    # create constraint nodes for constraints with more than 2 involved blocks
    constrID2NodeID = Dict{BlockID, String}()
    for constr in mbp.constraints
        if length(constr.involvedBlocks) > 2 
            nodeID = createNodeID(constr.id; isConstraint=true)
            graph.nodes[nodeID] = Node(Vector{String}(), CONSTRAINT_NODE, constr.id) 
            constrID2NodeID[constr.id] = nodeID
        end 
    end 

    # create edges
    for constr in mbp.constraints
        if length(constr.involvedBlocks) == 2
            edgeID = createEdgeID(constr.id)
            nodeID1 = createNodeID(constr.involvedBlocks[1])
            nodeID2 = createNodeID(constr.involvedBlocks[2])
            graph.edges[edgeID] = Edge(nodeID1, nodeID2, TWO_BLOCK_EDGE, constr.id, "")
        else 
            for blockID in constr.involvedBlocks
                edgeID = createEdgeID(constr.id; variableID=blockID)
                nodeID = createNodeID(blockID)
                graph.edges[edgeID] = Edge(nodeID, constrID2NodeID[constr.id], MULTIBLOCK_EDGE, constr.id, blockID)
            end 
        end 
    end 

    for (edgeID, edge) in graph.edges
        push!(graph.nodes[edge.nodeID1].neighbors, edgeID) 
        push!(graph.nodes[edge.nodeID2].neighbors, edgeID)
    end

    isMultiblockGraphBipartite(graph)
    isMultiblockGraphConnected(graph)

    return graph
end 

"""
    numberNodes(graph::MultiblockGraph) -> Int

Returns the total number of nodes in the graph.

# Arguments
- `graph::MultiblockGraph`: The graph to analyze

# Returns
- The number of nodes
"""
function numberNodes(graph::MultiblockGraph)
    return length(graph.nodes)
end 

"""
    numberEdges(graph::MultiblockGraph) -> Int

Returns the total number of edges in the graph.

# Arguments
- `graph::MultiblockGraph`: The graph to analyze

# Returns
- The number of edges
"""
function numberEdges(graph::MultiblockGraph)
    return length(graph.edges)
end 

"""
    numberEdgesByTypes(graph::MultiblockGraph) -> Tuple{Int, Int}

Counts the number of edges of each type in the graph.

# Arguments
- `graph::MultiblockGraph`: The graph to analyze

# Returns
- A tuple containing (count of two-block edges, count of multi-block edges)
"""
function numberEdgesByTypes(graph::MultiblockGraph)
    count2BlockEdges = 0
    countMultiBlockEdges = 0
    for (edgeID, edge) in graph.edges 
        if edge.type == TWO_BLOCK_EDGE
            count2BlockEdges += 1 
        else 
            countMultiBlockEdges += 1 
        end 
    end 
    return count2BlockEdges, countMultiBlockEdges
end 

"""
    summary(graph::MultiblockGraph)

Prints a comprehensive summary of the graph's properties to standard output.
Includes information about the number of nodes, edges, connectivity, and bipartiteness.

# Arguments
- `graph::MultiblockGraph`: The graph to summarize

# Output
Prints the following information:
- Total number of nodes and edges
- Breakdown of edge types (TWO_BLOCK_EDGE vs MULTIBLOCK_EDGE)
- Connectivity status (connected or disconnected)
- Bipartiteness status (bipartite or non-bipartite)
- Special handling for empty graphs

# Example Output
```
Summary of Multiblock Graph:
    Number of nodes             = 5
    Number of edges             = 7
    Number of TWO_BLOCK_EDGE    = 3
    Number of MULTIBLOCK_EDGE   = 4
    The graph is connected.
    The graph is bipartite.
```

# Usage
This function is useful for:
- Debugging graph construction
- Understanding problem structure
- Algorithm selection based on graph properties
- Reporting problem characteristics

# Note
The function calls `isMultiblockGraphBipartite` and `isMultiblockGraphConnected` internally
to ensure the graph properties are up-to-date before printing.
"""
function summary(graph::MultiblockGraph, logLevel::Int64=1)
    if logLevel < 1
        return 
    end 
    @PDMOInfo logLevel "Summary of Multiblock Graph: "
    nNodes = numberNodes(graph)
    nEdges = numberEdges(graph)
    println("    Number of nodes             = $(nNodes)")
    println("    Number of edges             = $(nEdges)")
    if nNodes == 0 || nEdges == 0 
        println("    The graph is empty.")
        return 
    end 

    count2BlockEdges, countMultiBlockEdges = numberEdgesByTypes(graph)
    println("    Number of TWO_BLOCK_EDGE    = $count2BlockEdges")
    println("    Number of MULTIBLOCK_EDGE   = $countMultiBlockEdges")

    if graph.isConnected
        println("    The graph is connected.")
    else 
        println("    The graph is NOT connected; you may want to divide the problem into smaller ones.")
    end 
    
    if graph.isBipartite
        println("    The graph is bipartite.")
    else 
        println("    The graph is NOT bipartite.")
    end 
end 

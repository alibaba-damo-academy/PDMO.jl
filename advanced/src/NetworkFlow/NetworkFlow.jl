struct Arc
    src::Int
    dst::Int 
    capacity::Float64
    cost::Float64
end 

mutable struct NetworkFlowProblem 
    numberNodes::Int 
    arcs::Vector{Arc}
    supply::Vector{Float64}
    offset::Float64
end 

"""
    readDimacsMinCostFlowInstance(path::AbstractString; applyLowerBounds::Bool=true)

Read a min-cost flow instance in (a variant of) the DIMACS textual format and convert it
to a `NetworkFlowProblem`.

This is the most common format used by benchmark collections (including LEMON's
*MinCostFlowData*), typically containing lines like:

- `c ...` comments (ignored)
- `p min <num_nodes> <num_arcs>` problem line
- `n <node_id> <supply>` node supply/demand (positive = supply, negative = demand)
- `a <src> <dst> <lower> <upper> <cost>` arc line (lower bound may be omitted)

Some files omit the lower bound and use `a <src> <dst> <capacity> <cost>`.

# Keyword Arguments
- `applyLowerBounds::Bool=true`: If `true`, arc lower bounds are eliminated via the standard
  transformation `f = l + f'` with `0 <= f' <= u - l`. This modifies node supplies and
  produces an additional constant objective offset, stored in `NetworkFlowProblem.offset`.

# Returns
- `nfp::NetworkFlowProblem`: the converted problem. The field `nfp.offset` stores the constant
  objective offset due to lower bounds (0.0 if `applyLowerBounds=false` or no lower bounds).
"""
function readDimacsMinCostFlowInstance(path::AbstractString; applyLowerBounds::Bool=true)
    numNodes = nothing
    supply = Float64[]
    arcs = Arc[]
    constantCost = 0.0

    # In case some instances list arcs before the `p` line (uncommon but harmless),
    # keep them until we know the node count.
    pending_arc_tokens = Vector{Vector{String}}()

    open(path, "r") do io
        for rawline in eachline(io)
            line = strip(rawline)
            isempty(line) && continue

            head = first(line)
            head == 'c' && continue  # comment

            # Convert SubString tokens to String so downstream parsing has stable types.
            tokens = String.(split(line))
            isempty(tokens) && continue

            if tokens[1] == "p"
                # Typical: p min <n> <m> (we only need n)
                ints = Int[]
                for t in tokens
                    try
                        push!(ints, parse(Int, t))
                    catch
                    end
                end
                if isempty(ints)
                    error("Invalid DIMACS 'p' line (no integers found): $line")
                end
                numNodes = ints[1]
                supply = zeros(Float64, numNodes)

                # Flush any arcs that appeared before we knew `numNodes`.
                ref = Ref(constantCost)
                for atoks in pending_arc_tokens
                    # Re-run arc parsing now that `supply` exists.
                    _parse_dimacs_arc_tokens!(arcs, supply, atoks; applyLowerBounds=applyLowerBounds, constantCostRef=ref)
                end
                constantCost = ref[]
                empty!(pending_arc_tokens)

            elseif tokens[1] == "n"
                numNodes === nothing && error("Encountered 'n' line before 'p' line: $line")
                length(tokens) < 3 && error("Invalid DIMACS 'n' line: $line")
                i = parse(Int, tokens[2])
                b = parse(Float64, tokens[3])
                supply[i] = b

            elseif tokens[1] == "a"
                if numNodes === nothing
                    push!(pending_arc_tokens, tokens)
                else
                    ref = Ref(constantCost)
                    _parse_dimacs_arc_tokens!(arcs, supply, tokens; applyLowerBounds=applyLowerBounds, constantCostRef=ref)
                    constantCost = ref[]
                end
            else
                # Ignore unrecognized lines (some generators include extra metadata).
                continue
            end
        end
    end

    numNodes === nothing && error("No DIMACS 'p' line found in instance: $path")
    return NetworkFlowProblem(numNodes, arcs, supply, constantCost)
end

# Internal helper: parse an "a ..." line tokens into `arcs`, optionally applying lower bounds.
function _parse_dimacs_arc_tokens!(
    arcs::Vector{Arc},
    supply::Vector{Float64},
    tokens::Vector{String};
    applyLowerBounds::Bool,
    constantCostRef::Base.RefValue{Float64},
)
    # Supported variants:
    # - a src dst lower upper cost
    # - a src dst upper cost   (lower assumed 0)
    length(tokens) < 5 && error("Invalid DIMACS 'a' line: $(join(tokens, " "))")

    src = parse(Int, tokens[2])
    dst = parse(Int, tokens[3])

    lower = 0.0
    upper = 0.0
    cost = 0.0

    if length(tokens) >= 6
        lower = parse(Float64, tokens[4])
        upper = parse(Float64, tokens[5])
        cost = parse(Float64, tokens[6])
    else
        upper = parse(Float64, tokens[4])
        cost = parse(Float64, tokens[5])
    end

    if applyLowerBounds && lower != 0.0
        upper < lower && error("Invalid arc bounds: upper < lower for arc ($src -> $dst)")
        supply[src] -= lower
        supply[dst] += lower
        constantCostRef[] += cost * lower
        push!(arcs, Arc(src, dst, upper - lower, cost))
    else
        push!(arcs, Arc(src, dst, upper, cost))
    end
    return nothing
end


function generateRandomNetworkFlowProblem(
    numberNodes::Int,
    numberArcs::Int;
    fractionDegree2::Float64 = 0.3,
    maxCapacity::Float64 = 40.0,
    maxCost::Float64 = 10.0,
    supplyMagnitude::Int = 100,
)
    @assert numberNodes >= 2 "numberNodes must be >= 2"
    @assert numberArcs >= 1 "numberArcs must be >= 1"
    @assert 0.0 <= fractionDegree2 <= 1.0 "fractionDegree2 must be in [0,1]"
    @assert maxCapacity > 0.0 "maxCapacity must be positive"
    @assert maxCost >= 0.0 "maxCost must be nonnegative"
    numberArcs < numberNodes && error("numberArcs must be >= numberNodes so every node has at least 2 incident arcs (avoids 0/1-block constraints).")
    maxSimpleEdges = (numberNodes * (numberNodes - 1)) ÷ 2
    if numberArcs > maxSimpleEdges
        @warn "generateRandomNetworkFlowProblem: requested numberArcs=$numberArcs exceeds max simple undirected edges=$maxSimpleEdges; capping to $maxSimpleEdges."
        numberArcs = maxSimpleEdges
    end

    # Step 1: build a SIMPLE UNDIRECTED graph (no self-loops, no parallel edges).
    # We will enforce that a subset of nodes have UNDIRECTED degree exactly 2.
    k = min(numberNodes, round(Int, fractionDegree2 * numberNodes))
    degree2_nodes = k == 0 ? Int[] : sort!(randperm(numberNodes)[1:k])
    degree2_set = Set(degree2_nodes)

    core_nodes = [i for i in 1:numberNodes if !(i in degree2_set)]

    # Store undirected edges as ordered pairs (a,b) with a<b
    undirected = Set{Tuple{Int, Int}}()
    degree = zeros(Int, numberNodes)

    function add_undirected!(u::Int, v::Int)
        u == v && return false
        a, b = u < v ? (u, v) : (v, u)
        if (a, b) in undirected
            return false
        end
        push!(undirected, (a, b))
        degree[u] += 1
        degree[v] += 1
        return true
    end

    # Build a single cycle on all nodes first:
    # - guarantees the graph is connected
    # - guarantees every node has undirected degree >= 2 (exactly 2 at this point)
    for i in 1:(numberNodes - 1)
        add_undirected!(i, i + 1)
    end
    add_undirected!(numberNodes, 1)

    # After the cycle, every node has degree exactly 2. We now add extra edges.
    # Soft requirement: we TRY to keep `degree2_nodes` at degree 2 by avoiding them,
    # but if that's impossible (too many edges requested), we relax and allow touching them.
    remaining_edges = numberArcs - numberNodes
    if remaining_edges > 0
        preferred_nodes = length(core_nodes) >= 2 ? core_nodes : collect(1:numberNodes)
        relaxed_nodes = collect(1:numberNodes)

        added = 0
        # Phase 1: try to add among preferred_nodes (avoiding degree2_set if possible).
        for _ in 1:remaining_edges
            tries = 0
            while true
                tries += 1
                if tries > 50_000
                    break
                end
                u = rand(preferred_nodes)
                v = rand(preferred_nodes)
                u == v && continue
                if add_undirected!(u, v)
                    added += 1
                    break
                end
            end
            if added == remaining_edges
                break
            end
        end

        # Phase 2: relax and allow edges involving any nodes to reach the requested count.
        while added < remaining_edges
            tries = 0
            while true
                tries += 1
                if tries > 200_000
                    # Graph is saturated (should only happen if we hit maxSimpleEdges cap).
                    break
                end
                u = rand(relaxed_nodes)
                v = rand(relaxed_nodes)
                u == v && continue
                if add_undirected!(u, v)
                    added += 1
                    break
                end
            end
            # If we couldn't add after many tries, stop (we are at maximum).
            if tries > 200_000
                break
            end
        end
    end

    # Step 2: assign a random direction to each undirected edge to form a directed network.
    # Degree-2 nodes are NOT forced to have one in and one out; both arcs could point in/out.
    arcs = Arc[]
    for (a, b) in undirected
        if rand(Bool)
            src, dst = a, b
        else
            src, dst = b, a
        end
        capacity = rand() * maxCapacity + 1e-6
        cost = rand() * maxCost
        push!(arcs, Arc(src, dst, capacity, cost))
    end

    # Step 3 (feasibility by construction):
    # Sample a feasible flow within capacities, then DEFINE supplies from flow conservation:
    # supply[v] = (sum outflow) - (sum inflow).
    # This guarantees the instance is feasible (the sampled flow is a feasible solution).
    supply = zeros(Float64, numberNodes)
    for arc in arcs
        # Keep flows comfortably within capacity so supplies aren't extreme.
        f = rand() * arc.capacity
        supply[arc.src] += f
        supply[arc.dst] -= f
    end

    # Optionally scale supplies down (and hence the witness feasible flow) to keep them bounded.
    # Scaling by α ∈ [0,1] preserves feasibility because α*f <= capacity.
    maxabs = maximum(abs.(supply))
    if maxabs > 0
        α = min(1.0, Float64(supplyMagnitude) / maxabs)
        supply .*= α
    end

    # Numerical cleanup: enforce exact zero net supply (should already hold up to FP error).
    supply[end] -= sum(supply)

    offset = 0.0
    return NetworkFlowProblem(numberNodes, arcs, supply, offset)
end

function generateNetworkFlowProblem(networkFlowProblem::NetworkFlowProblem)
    incoming = [Vector{Int}() for _ in 1:networkFlowProblem.numberNodes] 
    outgoing = [Vector{Int}() for _ in 1:networkFlowProblem.numberNodes] 
    for (k, arc) in enumerate(networkFlowProblem.arcs)
        s = arc.src 
        d = arc.dst 
        push!(incoming[d], k)
        push!(outgoing[s], k)
    end

    mbp = MultiblockProblem() 

    for (k, arc) in enumerate(networkFlowProblem.arcs)
        block = BlockVariable(k) 
        block.f = AffineFunction(Float64[arc.cost], k == 1 ? networkFlowProblem.offset : 0.0)
        block.g = IndicatorBox(Float64[0.0], Float64[arc.capacity])
        block.val = proximalOracle(block.g, Float64[arc.capacity/2.0])
        addBlockVariable!(mbp, block)
    end 

    for i in 1:networkFlowProblem.numberNodes
        constr = BlockConstraint(i) 
        for k in outgoing[i]
            addBlockMappingToConstraint!(constr, k, LinearMappingIdentity(1.0))
        end 
        for k in incoming[i]
            addBlockMappingToConstraint!(constr, k, LinearMappingIdentity(-1.0))
        end 
        constr.rhs = Float64[networkFlowProblem.supply[i]]
        addBlockConstraint!(mbp, constr)
    end 
    return mbp 
end 


function generateNetworkFlowProblemLP(networkFlowProblem::NetworkFlowProblem)
    incoming = [Vector{Int}() for _ in 1:networkFlowProblem.numberNodes] 
    outgoing = [Vector{Int}() for _ in 1:networkFlowProblem.numberNodes] 
    for (k, arc) in enumerate(networkFlowProblem.arcs)
        s = arc.src 
        d = arc.dst 
        push!(incoming[d], k)
        push!(outgoing[s], k)
    end

    mbp = MultiblockProblem() 

    xBlock = BlockVariable("x")
    xBlock.f = AffineFunction(Float64[arc.cost for arc in networkFlowProblem.arcs], networkFlowProblem.offset)
    xBlock.g = IndicatorBox(Float64[0.0 for arc in networkFlowProblem.arcs], Float64[arc.capacity for arc in networkFlowProblem.arcs])
    xBlock.val = proximalOracle(xBlock.g, Float64[arc.capacity/2.0 for arc in networkFlowProblem.arcs])
    addBlockVariable!(mbp, xBlock)

    zBlock = BlockVariable("z")
    # z is fixed to the supply vector: z == supply
    zBlock.f = Zero()
    zBlock.g = IndicatorBox(networkFlowProblem.supply, networkFlowProblem.supply)
    zBlock.val = copy(networkFlowProblem.supply)
    addBlockVariable!(mbp, zBlock)

    constr = BlockConstraint()
    A = spzeros(length(networkFlowProblem.supply), length(networkFlowProblem.arcs))
    for (k, arc) in enumerate(networkFlowProblem.arcs)
        A[arc.src, k] = 1.0
        A[arc.dst, k] = -1.0
    end
    addBlockMappingToConstraint!(constr, "x", LinearMappingMatrix(A))
    addBlockMappingToConstraint!(constr, "z", LinearMappingIdentity(-1.0))
    constr.rhs = spzeros(length(networkFlowProblem.supply))
    addBlockConstraint!(mbp, constr)

    return mbp 
end 
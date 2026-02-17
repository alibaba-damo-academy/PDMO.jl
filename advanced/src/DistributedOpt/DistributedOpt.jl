"""
    generateSimpleGraph(n::Int64, kappa::Float64; ensure_connected::Bool = true) -> SimpleGraph

Create an undirected simple graph with `n` nodes and connectivity `kappa`, i.e.
`ne(g) / (n choose 2) ≈ kappa`, by constructing exactly:

    m = round(Int, kappa * (n*(n-1)/2))

Notes:
- `kappa` is clamped to `[0.0, 1.0]`.
- If `ensure_connected=true`, the returned graph is guaranteed connected (for `n ≥ 2`)
  by first constructing a random spanning tree, then adding/removing random edges.
  In this case, if the requested `kappa` would imply `m < n-1`, `m` is bumped up to `n-1`.
"""
function generateSimpleGraph(n::Int64, kappa::Float64; ensure_connected::Bool = true)
    @assert n ≥ 0 "n must be nonnegative"
    κ = clamp(kappa, 0.0, 1.0)

    g = SimpleGraph(n)
    max_edges = n * (n - 1) ÷ 2
    if max_edges == 0
        return g
    end

    m = round(Int, κ * max_edges)
    m = clamp(m, 0, max_edges)

    # If we require connectivity, we need at least n-1 edges (for n>=2).
    tree_edges = Set{Tuple{Int, Int}}()
    if ensure_connected && n ≥ 2 && m < (n - 1)
        m = n - 1
    end

    # Build a random spanning tree backbone (guarantees connectedness).
    if ensure_connected && n ≥ 2
        for v in 2:n
            p = rand(1:(v - 1))
            push!(tree_edges, (p, v))
            add_edge!(g, p, v)
        end
    end

    # Dense case: start from complete graph and remove random edges,
    # while never removing spanning-tree edges (if ensure_connected=true).
    if m > max_edges * 0.5
        for u in 1:(n - 1), v in (u + 1):n
            add_edge!(g, u, v)
        end

        to_remove = max_edges - m
        if to_remove > 0
            removable = Tuple{Int, Int}[]
            for e in edges(g)
                a, b = src(e) < dst(e) ? (src(e), dst(e)) : (dst(e), src(e))
                if ensure_connected && ((a, b) in tree_edges)
                    continue
                end
                push!(removable, (a, b))
            end

            # Safety: should always hold if m >= n-1 when ensure_connected=true.
            @assert to_remove ≤ length(removable)

            idxs = randperm(length(removable))[1:to_remove]
            for i in idxs
                a, b = removable[i]
                rem_edge!(g, a, b)
            end
        end
        return g
    end

    # Sparse case: sample unique edges via a set
    chosen = Set{Tuple{Int, Int}}()
    if ensure_connected
        union!(chosen, tree_edges)
    end
    while length(chosen) < m
        u = rand(1:n)
        v = rand(1:n)
        if u == v
            continue
        end
        a, b = u < v ? (u, v) : (v, u)
        if (a, b) in chosen
            continue
        end
        push!(chosen, (a, b))
        add_edge!(g, a, b)
    end

    return g
end


function generateDistributedOptInstance(
    numberNodes::Int64,
    kappa::Float64,
    n::Int64,
    m::Int64
)
    # generate a simple graph with numberNodes nodes and kappa connectivity
    g = generateSimpleGraph(numberNodes, kappa)

    # generate a random vector of length n
    x = randn(n)

    objFunctions = Vector{QuadraticFunction}() 
    for i in 1:numberNodes
        A = randn(m, n)
        b = A * x + 0.1 * randn(m)
        push!(objFunctions, QuadraticFunction(sparse(A'*A), 2.0*A'*b, b'*b))
    end 

    return g, objFunctions 
end 


function generateDistributedOptProblem(g::SimpleGraph, objFunctions::Vector{QuadraticFunction})
    numberNodes = length(objFunctions)
    n = length(objFunctions[1].q)

    mbp = MultiblockProblem() 
    for i in 1:numberNodes
        block = BlockVariable(i) 
        block.f = objFunctions[i]
        block.val = zeros(n)
        addBlockVariable!(mbp, block)
    end 

    for e in edges(g)
        i = src(e)
        j = dst(e)
        constr = BlockConstraint("($i,$j)")
        addBlockMappingToConstraint!(constr, i, LinearMappingIdentity(1.0))
        addBlockMappingToConstraint!(constr, j, LinearMappingIdentity(-1.0))
        constr.rhs = zeros(n)
        addBlockConstraint!(mbp, constr)
    end 

    return mbp 
end 


function generateClassicDistributedOptProblem(g::SimpleGraph, objFunctions::Vector{QuadraticFunction})
    numberNodes = length(objFunctions)
    n = length(objFunctions[1].q)

    mbp = MultiblockProblem() 
    for i in 1:numberNodes
        block = BlockVariable(i) 
        block.f = objFunctions[i]
        block.val = zeros(n)
        addBlockVariable!(mbp, block)
    end 

    for e in edges(g)
        i = src(e)
        j = dst(e)
        
        # add block variable z(i,j)
        block = BlockVariable("z($i,$j)")
        block.val = zeros(n)
        addBlockVariable!(mbp, block)

        constr1 = BlockConstraint("x($i)-z($i,$j) = 0")
        addBlockMappingToConstraint!(constr1, i, LinearMappingIdentity(1.0))
        addBlockMappingToConstraint!(constr1, "z($i,$j)", LinearMappingIdentity(-1.0))
        constr1.rhs = zeros(n)
        addBlockConstraint!(mbp, constr1)

        constr2 = BlockConstraint("x($j)-z($i,$j) = 0")
        addBlockMappingToConstraint!(constr2, j, LinearMappingIdentity(1.0))
        addBlockMappingToConstraint!(constr2, "z($i,$j)", LinearMappingIdentity(-1.0))
        constr2.rhs = zeros(n)
        addBlockConstraint!(mbp, constr2)
    end 

    return mbp 
end 
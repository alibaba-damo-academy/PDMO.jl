using JuMP 
using Ipopt

import PowerModels

include("CustomizedGeneratorCost.jl")
include("CustomizedZoneCost.jl")
include("OPFLib.jl")
include("IndicatorLocalDCConstraints.jl")


function distributedDCOPF(networkInfo, bus2Area::Dict{String, Int64};
    customizedGeneratorCost::Bool=false,
    mergeConstraints::Bool=true)
    numberAreas = length(unique(collect(values(bus2Area))))
    areaInfo, tieLines = createAreaNetworkInfo(networkInfo, bus2Area, numberAreas)
    mbp = MultiblockProblem() 

    # Create blocks for each area
    for area in 1:numberAreas
        block = BlockVariable(area) 
        block.g = IndicatorLocalDCConstraints(areaInfo[area])

        if customizedGeneratorCost
            # Gradient of the zone cost is a vector in the FULL block variable space.
            # Use the full dimension, not the number of generators, otherwise indices
            # like `p_g` positions can exceed `numberColumns` and cause BoundsError.
            numberColumns = block.g.numberVariables
            generatorIndices = Int64[] 
            generatorCosts = CustomizedGeneratorCost[]
            for g in keys(block.g.varToIdx["p_g"])
                idx = block.g.varToIdx["p_g"][g]
                push!(generatorIndices, idx)
                push!(generatorCosts, networkInfo.customizedGeneratorCost[g])
            end 
            block.f = CustomizedZoneCost(numberColumns, generatorIndices, generatorCosts)
            println("distributedDCOPF: area = $area has $(length(generatorIndices)) generators with customized costs")
        else 
            numberConstCost = 0
            numberLinearCost = 0
            numberQuadraticCost = 0
            # Add objective function if area has generators
            if !isempty(block.g.varToIdx["p_g"])
                Q = spzeros(Float64, block.g.numberVariables, block.g.numberVariables)
                q = zeros(block.g.numberVariables)
                r = 0.0
                
                for g in keys(block.g.varToIdx["p_g"])
                    ncost = networkInfo.data["gen"][g]["ncost"]
                    cost_coe = networkInfo.data["gen"][g]["cost"]
                    idx = block.g.varToIdx["p_g"][g]
                    
                    if ncost == 1 
                        r += cost_coe[1]
                        numberConstCost += 1
                    elseif ncost == 2 
                        q[idx] = cost_coe[1]
                        r += cost_coe[2]
                        numberLinearCost += 1
                    elseif ncost == 3
                        Q[idx, idx] = cost_coe[1]
                        q[idx] = cost_coe[2]
                        r += cost_coe[3]
                        numberQuadraticCost += 1
                    else 
                        @warn "distributedDCOPF: generator $g has $(ncost) pieces; skipped."
                    end 
                end 
                
                block.f = nnz(Q) > 0 ? QuadraticFunction(Q, q, r) : AffineFunction(q, r)
            end 
            println("distributedDCOPF: generator type for area = $area")
            println("   # constant cost generators = $numberConstCost")
            println("   # linear cost generators = $numberLinearCost")
            println("   # quadratic cost generators = $numberQuadraticCost")
        end 

        block.val = proximalOracle(block.g, zeros(block.g.numberVariables)) 
        addBlockVariable!(mbp, block)
    end 

    if mergeConstraints
        # Merge constraints by (areaA, areaB) pairs (unordered)
        first_mats = Dict{Tuple{Int,Int}, Vector{SparseMatrixCSC{Float64,Int}}}()
        second_mats = Dict{Tuple{Int,Int}, Vector{SparseMatrixCSC{Float64,Int}}}()
        for l in tieLines
            f_bus = string(networkInfo.data["branch"][l]["f_bus"])
            t_bus = string(networkInfo.data["branch"][l]["t_bus"])
            f_area = bus2Area[f_bus]
            t_area = bus2Area[t_bus]

            # Consistency constraints for voltage angles at boundary buses (3 rows per tie line)
            matrix_f_area = spzeros(Float64, 3, mbp.blocks[f_area].g.numberVariables)
            f_bus_idx_in_f_area = mbp.blocks[f_area].g.varToIdx["theta"][f_bus]
            t_bus_idx_in_f_area = mbp.blocks[f_area].g.varToIdx["theta"][t_bus]
            matrix_f_area[1, f_bus_idx_in_f_area] = 1.0
            matrix_f_area[2, t_bus_idx_in_f_area] = 1.0

            matrix_t_area = spzeros(Float64, 3, mbp.blocks[t_area].g.numberVariables)
            f_bus_idx_in_t_area = mbp.blocks[t_area].g.varToIdx["theta"][f_bus]
            t_bus_idx_in_t_area = mbp.blocks[t_area].g.varToIdx["theta"][t_bus]
            matrix_t_area[1, f_bus_idx_in_t_area] = -1.0
            matrix_t_area[2, t_bus_idx_in_t_area] = -1.0

            # Add power flow consistency
            p_from_idx_f = mbp.blocks[f_area].g.varToIdx["p_from"][l]
            p_from_idx_t = mbp.blocks[t_area].g.varToIdx["p_from"][l]
            matrix_f_area[3, p_from_idx_f] = 1.0
            matrix_t_area[3, p_from_idx_t] = -1.0

            # Unordered key for (areaA, areaB), but keep orientation for matrices
            a, b = f_area <= t_area ? (f_area, t_area) : (t_area, f_area)
            key = (a, b)
            if !haskey(first_mats, key)
                first_mats[key] = Vector{SparseMatrixCSC{Float64,Int}}()
                second_mats[key] = Vector{SparseMatrixCSC{Float64,Int}}()
            end
            if f_area <= t_area
                push!(first_mats[key], matrix_f_area)
                push!(second_mats[key], matrix_t_area)
            else
                push!(first_mats[key], matrix_t_area)
                push!(second_mats[key], matrix_f_area)
            end
        end

        # Emit one BlockConstraint per area pair by stacking rows
        idx = 0
        for (key, mats_first) in first_mats
            mats_second = second_mats[key]
            # stack rows (vcat) for all lines between the same pair
            Hfirst = mats_first[1]
            for i in 2:length(mats_first)
                Hfirst = vcat(Hfirst, mats_first[i])
            end
            Hsecond = mats_second[1]
            for i in 2:length(mats_second)
                Hsecond = vcat(Hsecond, mats_second[i])
            end
            idx += 1
            constr = BlockConstraint(idx)
            addBlockMappingToConstraint!(constr, key[1], LinearMappingMatrix(Hfirst))
            addBlockMappingToConstraint!(constr, key[2], LinearMappingMatrix(Hsecond))
            constr.rhs = zeros(size(Hfirst, 1))
            addBlockConstraint!(mbp, constr)
        end
    else
        # Add coupling constraints for tie lines (one constraint per line)
        for (idx, l) in enumerate(tieLines)
            constr = BlockConstraint(idx)
    
            f_bus = string(networkInfo.data["branch"][l]["f_bus"])
            t_bus = string(networkInfo.data["branch"][l]["t_bus"])
            f_area = bus2Area[f_bus]
            t_area = bus2Area[t_bus]
    
            # Consistency constraints for voltage angles at boundary buses
            matrix_f_area = spzeros(3, mbp.blocks[f_area].g.numberVariables)
            f_bus_idx_in_f_area = mbp.blocks[f_area].g.varToIdx["theta"][f_bus]
            t_bus_idx_in_f_area = mbp.blocks[f_area].g.varToIdx["theta"][t_bus]
            matrix_f_area[1, f_bus_idx_in_f_area] = 1.0 
            matrix_f_area[2, t_bus_idx_in_f_area] = 1.0 
            
            matrix_t_area = spzeros(3, mbp.blocks[t_area].g.numberVariables)
            f_bus_idx_in_t_area = mbp.blocks[t_area].g.varToIdx["theta"][f_bus]
            t_bus_idx_in_t_area = mbp.blocks[t_area].g.varToIdx["theta"][t_bus]
            matrix_t_area[1, f_bus_idx_in_t_area] = -1.0 
            matrix_t_area[2, t_bus_idx_in_t_area] = -1.0 
            
            # Add power flow consistency
            p_from_idx_f = mbp.blocks[f_area].g.varToIdx["p_from"][l]
            p_from_idx_t = mbp.blocks[t_area].g.varToIdx["p_from"][l]
            matrix_f_area[3, p_from_idx_f] = 1.0
            matrix_t_area[3, p_from_idx_t] = -1.0
    
            addBlockMappingToConstraint!(constr, f_area, LinearMappingMatrix(matrix_f_area))
            addBlockMappingToConstraint!(constr, t_area, LinearMappingMatrix(matrix_t_area))
            constr.rhs = zeros(3)
    
            addBlockConstraint!(mbp, constr)
        end 
    end

    return mbp 
end 


function generatePartitions(networkInfo, K)
    
    numberVertices = length(networkInfo.BusIDList)
    bus2vertex = Dict{String, Int64}(networkInfo.BusIDList[i]=>i for i in 1:numberVertices)

    edges = Vector{Tuple{Int64, Int64}}() 
    for l in networkInfo.BrchIDList
        f = string(networkInfo.data["branch"][l]["f_bus"])
        t = string(networkInfo.data["branch"][l]["t_bus"])
        f_idx = bus2vertex[f]
        t_idx = bus2vertex[t]
        push!(edges, (f_idx, t_idx))
    end 
    unique!(edges)

    # create a simple graph 
    g = LightGraphs.SimpleGraph(numberVertices)
    for (f,t) in edges 
        LightGraphs.add_edge!(g, f, t)
    end 

    # generate partitions
    partition = Metis.partition(g, K, alg = :KWAY)
    
    bus2Area = Dict{String, Int64}() 
    for i in 1:numberVertices 
        bus = networkInfo.BusIDList[i]
        area = partition[i]
        bus2Area[bus] = area 
    end 
    return bus2Area
end 


function testDistributedOPF(networkInfo::PowerNetworkInfo, 
    numberPartitions::Int64, 
    bus2Area::Dict{String, Int64},
    bipartAlgo::BipartizationAlgorithm, 
    admmSolver::AbstractADMMSubproblemSolver;
    dcObj::Float64 = Inf, 
    initialRho::Float64 = 10.0,
    tol::Float64=1.0e-4,
    maxIter::Int=1000000,
    timeLimit::Float64=7200.0,
    logInterval::Int=100,
    useCustomizedGeneratorCost::Bool=false, 
    mergeConstraints::Bool=true,
    seed::Int=123)

    # create a distributed DCOPF model
    mbp = distributedDCOPF(networkInfo, bus2Area; customizedGeneratorCost=useCustomizedGeneratorCost, mergeConstraints=mergeConstraints)

    # setup ADMM parameters 
    param = ADMMParam() 
    param.solver = admmSolver
    param.initialRho = initialRho
    param.logInterval = logInterval 
    param.maxIter = maxIter
    param.timeLimit = timeLimit
    param.presTolL2 = Inf 
    param.dresTolL2 = Inf 
    param.presTolLInf = tol
    param.dresTolLInf = tol
    # param.applyScaling = true 

    # run ADMM 
    result = runBipartiteADMM(mbp, param;
        bipartizationAlgorithm = bipartAlgo, 
        trueObj = dcObj, 
        tryJuMP = false)

    return result.iterationInfo 
end 


function printArea2Bus(bus2Area::Dict{String, Int64})
    area2Bus = Dict{Int64, Vector{String}}()
    for bus in keys(bus2Area)
        area = bus2Area[bus]
        if !haskey(area2Bus, area)
            area2Bus[area] = String[]
        end
        push!(area2Bus[area], bus)
    end
    for area in sort(collect(keys(area2Bus)))
        println("Area $area has $(length(area2Bus[area])) buses: $(area2Bus[area])")
    end
end


"""
OPF-specific serialization helpers for GNN bipartization.
"""

function toDict(f::IndicatorLocalDCConstraints)
    return Dict(
        "numberVariables" => f.numberVariables,
        "varToIdx" => f.varToIdx,
        "FunctionType" => "IndicatorLocalDCConstraints"
    )
end

# These functions are now replaced by the trait-based system using isSupportedByJuMP
# and the individual function implementations of JuMPAddProximableFunction and JuMPAddSmoothFunction

# unwrapFunction is no longer needed - wrapper functions handle their own transformations
# through their JuMP API implementations

"""
    addBlockVariableToJuMPModel!(model::JuMP.Model, 
                               f::AbstractFunction,
                               g::AbstractFunction,
                               x::NumericVariable,
                               blockID::BlockID, 
                               var::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}, 
                               objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}, 
                               nonlinearObjExpressions::Vector)

Add a block variable to a JuMP model.

# Arguments
- `model::JuMP.Model`: The JuMP model to which the variable will be added.
- `f::AbstractFunction`: The smooth function component of the block variable.
- `g::AbstractFunction`: The nonsmooth function component of the block variable.
- `x::NumericVariable`: The current value of the block variable.
- `blockID::BlockID`: The ID of the block variable.
- `var::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}`: Dictionary to store the created JuMP variables.
- `objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}`: Vector to collect linear/quadratic objective expressions.
- `nonlinearObjExpressions::Vector`: Vector to collect nonlinear objective expressions.

# Returns
- `Nothing`: The function modifies the collections in-place.

# Implementation Details
This function uses the new trait-based system to add functions to JuMP models:
1. Checks if functions support JuMP using `isSupportedByJuMP(f)` trait
2. Creates variables for the block using `JuMP.@variable`
3. Calls `JuMPAddProximableFunction(g, model, var)` to add constraints from proximable function g
4. Calls `JuMPAddSmoothFunction(f, model, var)` to add objective terms from smooth function f
5. Collects linear/quadratic objective expressions and detects nonlinear expressions

The trait-based approach allows easy extension by implementing the JuMP APIs for new function types
without modifying this central function.
""" 
function addBlockVariableToJuMPModel!(model::JuMP.Model, 
    f::AbstractFunction,
    g::AbstractFunction,
    x::NumericVariable,
    blockID::BlockID, 
    var::Union{Dict{BlockID, Vector{JuMP.VariableRef}}, Dict{String, Vector{JuMP.VariableRef}}}, 
    objExpressions::Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}, 
    nonlinearObjExpressions::Vector)

    @assert(length(size(x)) == 1, "addBlockVariableToJuMPModel!: only support vector variables for now.")

    # Check if function types are supported using new trait-based system
    @assert(isSupportedByJuMP(f), 
        "addBlockVariableToJuMPModel!: unsupported type of f = $(typeof(f))")

    @assert(isSupportedByJuMP(g),   
        "addBlockVariableToJuMPModel!: unsupported type of g = $(typeof(g))")

    dim = length(x)
    
    # Create variables for this block
    var[blockID] = JuMP.@variable(model, [k = 1:dim])
    # Use new trait-based system for proximable functions (constraints)
    try
        objTerm_g = JuMPAddProximableFunction(g, model, var[blockID])
        if isa(objTerm_g, Union{JuMP.AffExpr, JuMP.QuadExpr})
            push!(objExpressions, objTerm_g)
        elseif objTerm_g !== nothing
            push!(nonlinearObjExpressions, objTerm_g)
        end
    catch e
        error("addBlockVariableToJuMPModel!: Failed to add proximable function $(typeof(g)): $e")
    end
    
    # Use new trait-based system for smooth functions (objectives)
    try
        objTerm_f = JuMPAddSmoothFunction(f, model, var[blockID])
        if isa(objTerm_f, Union{JuMP.AffExpr, JuMP.QuadExpr})
            push!(objExpressions, objTerm_f)
        elseif objTerm_f !== nothing
            push!(nonlinearObjExpressions, objTerm_f)
        end
    catch e
        error("addBlockVariableToJuMPModel!: Failed to add smooth function $(typeof(f)): $e")
    end
end

"""
    solveMultiblockProblemByJuMP(mbp::MultiblockProblem)

Solve a multiblock problem using JuMP and Ipopt.

# Arguments
- `mbp::MultiblockProblem`: The multiblock problem to solve.

# Returns
- `Float64`: The optimal objective value found by the solver.

# Implementation Details
The function:
1. Converts the multiblock problem to a JuMP model
2. Sets up the Ipopt solver with HSL linear solver
3. Creates JuMP variables for each block variable
4. Adds constraints from the multiblock problem
5. Formulates the objective function (handling both linear/quadratic and nonlinear terms)
6. Solves the problem and returns the objective value

Currently, only LinearMappingMatrix and LinearMappingIdentity are supported for mappings.
"""
function solveMultiblockProblemByJuMP(mbp::MultiblockProblem, logLevel::Int64=1; 
    hasInteger::Bool=false, 
    silent::Bool=true)
    for constr in mbp.constraints
        for (id, L) in constr.mappings
            @assert(typeof(L) == LinearMappingMatrix || typeof(L) == LinearMappingIdentity, 
                "solveMultiblockProblemByJuMP: unsupported type of linear mapping = $(typeof(L))")
        end 
    end 

    if mbp.couplingFunction != nothing && isSupportedByJuMP(mbp.couplingFunction) == false 
        error("solveMultiblockProblemByJuMP: unsupported type of coupling function = $(typeof(mbp.couplingFunction))")
    end 

    model = hasInteger ? JuMP.Model(HiGHS.Optimizer) : JuMP.Model(Ipopt.Optimizer)
    
    if hasInteger == false && HSL_FOUND 
        JuMP.set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        JuMP.set_attribute(model, "linear_solver", "ma27")
    end 

    if silent
        JuMP.set_silent(model)
    end

    var = Dict{BlockID, Vector{JuMP.VariableRef}}() 
    objExpressions = Vector{Union{JuMP.AffExpr, JuMP.QuadExpr}}()
    nonlinearObjExpressions = Vector{Any}()

    for block in mbp.blocks 
        addBlockVariableToJuMPModel!(model, 
            block.f, 
            block.g, 
            block.val, 
            block.id, 
            var, 
            objExpressions,
            nonlinearObjExpressions)
    end 

    # add coupling function 
    if mbp.couplingFunction != nothing 
        varVector = Vector{Vector{JuMP.VariableRef}}()
        for block in mbp.blocks 
            push!(varVector, var[block.id])
        end 
        obj_expr = JuMPAddSmoothFunction(mbp.couplingFunction, model, varVector)
        if isa(obj_expr,  JuMP.QuadExpr) || isa(obj_expr, JuMP.AffExpr)
            push!(objExpressions, obj_expr)
        elseif obj_expr !== nothing
            push!(nonlinearObjExpressions, obj_expr)
        end
    end 

    # Add constraints
    for constr in mbp.constraints
        JuMP.@constraint(model, 
                sum((typeof(L) == LinearMappingMatrix ? L.A * var[id] : L.coe * var[id]) for (id, L) in constr.mappings) .== constr.rhs)  
    end 

    # add objective 
    quadraticObj = JuMP.QuadExpr()
    if !isempty(objExpressions) 
        for expr in objExpressions 
            JuMP.add_to_expression!(quadraticObj, expr)
        end 
    end 
    
    if !isempty(nonlinearObjExpressions)
        # Build nonlinear objective by combining all expressions
        nonlinearObj = quadraticObj  # Start with linear/quadratic terms
        for expr in nonlinearObjExpressions 
            nonlinearObj += expr
        end 
        JuMP.@NLobjective(model, Min, nonlinearObj)
    else 
        JuMP.@objective(model, Min, quadraticObj)
    end 

    # # Print the model to see what JuMP is actually solving
    # println("="^80)
    # println("JuMP MODEL BEING SOLVED:")
    # println("="^80)
    # println(model)
    # println("="^80)

    JuMP.optimize!(model)

    JuMP.solution_summary(model)

    # # Print solution values to understand why JuMP thinks it's solved
    # println("SOLUTION VALUES:")
    # for (blockID, variables) in var
    #     vals = JuMP.value.(variables)
    #     println("Block $blockID: $vals")
    # end
    # println("="^80)

    obj = JuMP.objective_value(model)
    time = JuMP.solve_time(model)
    status = JuMP.termination_status(model)
    
    if logLevel >= 1
        msg = Printf.@sprintf("MultiblockProblem: Objective value by JuMP = %.4e, time = %.2f seconds, status = %s\n", 
                         obj, time, status) 
        @PDMOInfo logLevel msg  
    end 

    return obj
end 

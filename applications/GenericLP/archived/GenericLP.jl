""" 
Generic LP 

    min_{x} <c, x> + offset 
       s.t. row_lower <= Ax <= row_upper
            col_lower <=  x <= col_upper
"""

struct GenericLP 
    number_cols::Int64 
    number_rows::Int64 
    obj::Vector{Float64}
    offset::Float64 
    A::SparseMatrixCSC
    row_lower::Vector{Float64}
    row_upper::Vector{Float64}
    col_lower::Vector{Float64}
    col_upper::Vector{Float64}
end 

function GenericLP(mps_dir::String)
    model = JuMP.read_from_file(mps_dir)
    vars = JuMP.all_variables(model)
    number_cols = length(vars)

    # obj 
    obj_expr = JuMP.objective_function(model)
    obj = zeros(number_cols)
    for (i, v) in enumerate(vars)
        obj[i] = JuMP.coefficient(obj_expr, v)
    end 
    offset = JuMP.constant(obj_expr)

    # constraint 
    constraints = JuMP.all_constraints(model; include_variable_in_set_constraints = false)
    number_rows = length(constraints)

    row_inds = Int[]
    col_inds = Int[]
    vals = Float64[]
    row_lower = zeros(number_rows)
    row_upper = zeros(number_rows)

    var2Idx = Dict(vars[i]=>i for i in 1:number_cols)

    for (i, con_ref) in enumerate(constraints)
        con = JuMP.constraint_object(con_ref)
        aff = con.func
        for (v, coeff) in aff.terms
            j = var2Idx[v]
            push!(row_inds, i)
            push!(col_inds, j)
            push!(vals, coeff)
        end 

        # Extract the bounds based on the type of the constraint set.
        s = con.set
        if s isa MathOptInterface.EqualTo{Float64}
            row_lower[i] = s.value
            row_upper[i] = s.value
        elseif s isa MathOptInterface.Interval{Float64}
            row_lower[i] = s.lower
            row_upper[i] = s.upper
        elseif s isa MathOptInterface.LessThan{Float64}
            row_lower[i] = -Inf
            row_upper[i] = s.upper
        elseif s isa MathOptInterface.GreaterThan{Float64}
            row_lower[i] = s.lower
            row_upper[i] = Inf
        else
            error("Unsupported constraint set: $s")
        end
    end 

    # Build the sparse matrix A.
    A = sparse(row_inds, col_inds, vals, number_rows, number_cols)

    # Extract variable bounds.
    # Helper functions that safely return variable bounds.
    safe_lower_bound(v::JuMP.VariableRef) = try
        JuMP.lower_bound(v)
    catch
        -Inf
    end

    safe_upper_bound(v::JuMP.VariableRef) = try
        JuMP.upper_bound(v)
    catch
        Inf
    end

    col_lower = Float64[safe_lower_bound(v) for v in vars]
    col_upper = Float64[safe_upper_bound(v) for v in vars]

    return GenericLP(number_cols, 
        number_rows, 
        obj, 
        offset, 
        A, 
        row_lower, row_upper, 
        col_lower, col_upper)
end 


# function generateGenericLP(lp::GenericLP)
    
#     number_slacks = 0
#     for r in 1:lp.number_rows
#         row_lower = lp.row_lower[r]
#         row_upper = lp.row_upper[r]
#         if abs(row_lower - row_upper) < FeasTolerance
#             continue 
#         end 

#         if row_upper < Inf 
#             number_slacks += 1
#         end 

#         if row_lower > -Inf
#             number_slacks += 1
#         end 
#     end 

#     for c in 1:lp.number_cols
#         col_lower = lp.col_lower[c]
#         col_upper = lp.col_upper[c]
#         if abs(col_lower - col_upper) < FeasTolerance 
#             continue 
#         end 
#         if col_upper < Inf 
#             number_slacks += 1
#         end 
#         if col_lower > -Inf 
#             number_slacks += 1
#         end 
#     end 

#     mbp = MultiblockProblem() 

#     xID = 1
#     block_x = BlockVariable(xID)
#     block_x.f = AffineFunction(lp.obj, lp.offset)
#     block_x.val = zeros(lp.number_cols)
#     push!(mbp.blocks, block_x)

#     sID = 2
#     block_s = BlockVariable(sID) 
#     block_s.g = IndicatorBox(zeros(number_slacks), ones(number_slacks) * Inf)
#     block_s.val = zeros(number_slacks)
#     push!(mbp.blocks, block_s)

#     constrID = 1
#     constr = BlockConstraint(constrID) 
#     push!(constr.involvedBlocks, xID)
#     push!(constr.involvedBlocks, sID)
    
#     count_slack_idx = 0 
#     for r in 1:lp.number_rows
#         row_lower = lp.row_lower[r]
#         row_upper = lp.row_upper[r]
#         if abs(row_lower - row_upper) < FeasTolerance
#             row = ScalarConstraint() 
#             row.coupling_coefficient[1] = lp.A[r, :]
#             row.coupling_coefficient[2] = spzeros(number_slacks)
#             row.rhs = row_lower
#             push!(constr.constraints, row)
#             continue 
#         end 

#         if row_upper < Inf 
#             count_slack_idx += 1
#             row = ScalarConstraint() 
#             row.coupling_coefficient[1] = lp.A[r,:]
#             row.coupling_coefficient[2] = spzeros(number_slacks)
#             row.coupling_coefficient[2][count_slack_idx] = 1.0
#             row.rhs = row_upper 
#             push!(constr.constraints, row)
#         end 

#         if row_lower > -Inf
#             count_slack_idx += 1
#             row = ScalarConstraint() 
#             row.coupling_coefficient[1] = lp.A[r,:]
#             row.coupling_coefficient[2] = spzeros(number_slacks)
#             row.coupling_coefficient[2][count_slack_idx] = -1.0
#             row.rhs = row_lower 
#             push!(constr.constraints, row)
#         end 
#     end 

#     for c in 1:lp.number_cols
#         col_lower = lp.col_lower[c]
#         col_upper = lp.col_upper[c]

#         if abs(col_lower - col_upper) < FeasTolerance 
#             row = ScalarConstraint() 
#             row.coupling_coefficient[1] = spzeros(lp.number_cols)
#             row.coupling_coefficient[1][c] = 1.0
#             row.coupling_coefficient[2] = spzeros(number_slacks)
#             row.rhs = col_lower
#             push!(constr.constraints, row)
#             continue 
#         end 

#         if col_upper < Inf 
#             count_slack_idx += 1
#             row = ScalarConstraint() 
#             row.coupling_coefficient[1] = spzeros(lp.number_cols)
#             row.coupling_coefficient[1][c] = 1.0
#             row.coupling_coefficient[2] = spzeros(number_slacks)
#             row.coupling_coefficient[2][count_slack_idx] = 1.0
#             row.rhs = col_upper 
#             push!(constr.constraints, row)
#         end 

#         if col_lower > -Inf 
#             count_slack_idx += 1
#             row = ScalarConstraint() 
#             row.coupling_coefficient[1] = spzeros(lp.number_cols)
#             row.coupling_coefficient[1][c] = 1.0
#             row.coupling_coefficient[2] = spzeros(number_slacks)
#             row.coupling_coefficient[2][count_slack_idx] = -1.0
#             row.rhs = col_lower
#             push!(constr.constraints, row)
#         end 

#     end 

#     constr.number_constraints = length(constr.constraints)

#     # println("count_slack_idx = $(count_slack_idx), number_slacks = $(number_slacks)")
#     push!(mbp.constraints, constr)

#     return mbp 
# end 


""" 
Two-block reformulation: 
    
    min_{x, s} <c, x> + offset 
          s.t.  Ax - s = 0 
                col_lower <= x <= col_upper 
                row_lower <= s <= row_upper 
"""
function generateGenericLP(lp::GenericLP)
    mbp = MultiblockProblem() 

    block_x = BlockVariable("x")
    block_x.f = AffineFunction(lp.obj, lp.offset)
    block_x.g = IndicatorBox(lp.col_lower, lp.col_upper)
    block_x.val = proximalOracle(block_x.g, zeros(lp.number_cols))
    addBlockVariable!(mbp, block_x)

    block_s = BlockVariable("s") 
    block_s.g = IndicatorBox(lp.row_lower, lp.row_upper)
    block_s.val = proximalOracle(block_s.g, zeros(lp.number_rows))
    addBlockVariable!(mbp, block_s)

    constr = BlockConstraint() 
    addBlockMappingToConstraint!(constr, "x", LinearMappingMatrix(lp.A))
    addBlockMappingToConstraint!(constr, "s", LinearMappingIdentity(-1.0))
    constr.rhs = zeros(lp.number_rows)
    addBlockConstraint!(mbp, constr)

    return mbp 
end 

""" 
    min <c, x> 
    s.t  Ax - s1 = 0 
          x - s2 = 0 
          
          row_lower <= s1 <= row_upper 
          col_lower <= s2 <= col_upper 
"""
function generateGenericLP2(lp::GenericLP)
    mbp = MultiblockProblem() 

    block_x = BlockVariable("x")
    block_x.f = AffineFunction(lp.obj, lp.offset)
    block_x.val = proximalOracle(block_x.g, zeros(lp.number_cols))
    addBlockVariable!(mbp, block_x)

    block_rowSlack = BlockVariable("rowSlack") 
    block_rowSlack.g = IndicatorBox(lp.row_lower, lp.row_upper)
    block_rowSlack.val = proximalOracle(block_rowSlack.g, zeros(lp.number_rows))
    addBlockVariable!(mbp, block_rowSlack)

    block_colSlack = BlockVariable("colSlack") 
    block_colSlack.g = IndicatorBox(lp.col_lower, lp.col_upper)
    block_colSlack.val = proximalOracle(block_colSlack.g, zeros(lp.number_cols))
    addBlockVariable!(mbp, block_colSlack)

    # Ax - s1 = 0
    constrRowSlack = BlockConstraint()  
    addBlockMappingToConstraint!(constrRowSlack, "x", LinearMappingMatrix(lp.A))
    addBlockMappingToConstraint!(constrRowSlack, "rowSlack", LinearMappingIdentity(-1.0))
    constrRowSlack.rhs = zeros(lp.number_rows)
    addBlockConstraint!(mbp, constrRowSlack)
    
    # x - s2 = 0
    constrColSlack = BlockConstraint()
    addBlockMappingToConstraint!(constrColSlack, "x", LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constrColSlack, "colSlack", LinearMappingIdentity(-1.0))
    constrColSlack.rhs = zeros(lp.number_cols)
    addBlockConstraint!(mbp, constrColSlack)

    return mbp 
end 

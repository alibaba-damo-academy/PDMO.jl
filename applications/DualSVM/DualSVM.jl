""" 
Dual SVM: 
    min_{x} 0.5 * <Qx, x> - <e, x>
       s.t. <b,x> = 0
            0 <= x_i <= C for all i 


Two-block reformulation:

    min_{x,z} 0.5 * <Qx, x> - <e,x>
         s.t. 
              x - z = 0
              0 <= z_i <= C for all i 
              x in {x: <b,x> = 0}   
"""

function generateDualSVM(Q::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, C::Float64)
    
    numberVars = length(b)
    @assert(numberVars == size(Q,1) == size(Q, 2), "DualSVM: input dimension mismatch. ")

    mbp = MultiblockProblem() 

    # x block
    block_x = BlockVariable() 
    block_x.f = QuadraticFunction(0.5 * Q, -ones(numberVars), 0.0)
    block_x.g = IndicatorHyperplane(b, 0.0)
    block_x.val = zeros(numberVars) # initial point
    xID = addBlockVariable!(mbp, block_x)

    # z block
    block_z = BlockVariable() 
    block_z.g = IndicatorBox(zeros(numberVars), ones(numberVars) * C)
    block_z.val = zeros(numberVars) # initial point
    zID = addBlockVariable!(mbp, block_z)

    # constraint: x - z = 0 
    constr = BlockConstraint() 
    addBlockMappingToConstraint!(constr, xID, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0))
    constr.rhs = spzeros(numberVars)
    addBlockConstraint!(mbp, constr)
    
    return mbp 
end 
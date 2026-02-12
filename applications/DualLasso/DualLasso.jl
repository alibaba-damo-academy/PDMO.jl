"""
Dual Lasso: 
    min_{x} (1/4) ||x||^2 - b'x 
        s.t. ||Ax||_{inf}  <= lambda 

Two-block reformulation: 
    min_{x,z} (1/4) ||x||^2 - b'x 
        s.t. 
            Ax - z = 0 
            ||z||_{inf}  <= lambda 
"""


function generateDualLasso(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, lambda::Float64)
    mbp = MultiblockProblem() 
    
    numberRows, numberCols = size(A)
    @assert(numberCols == length(b), "DualLasso: Dimension mismatch. ")

    # x block
    block_x = BlockVariable() 
    block_x.f = QuadraticFunction(0.25 * spdiagm(0 => ones(numberCols)), -b, 0.0)
    block_x.val = zeros(numberCols)
    xID = addBlockVariable!(mbp, block_x)

    # z block 
    block_z = BlockVariable() 
    block_z.g = IndicatorBox(-lambda * ones(numberRows), ones(numberRows) * lambda)
    block_z.val = zeros(numberRows)
    zID = addBlockVariable!(mbp, block_z)

    # constraint 
    constr = BlockConstraint() 
    addBlockMappingToConstraint!(constr, xID, LinearMappingMatrix(A))
    addBlockMappingToConstraint!(constr, zID, LinearMappingIdentity(-1.0))
    constr.rhs = zeros(numberRows)
    addBlockConstraint!(mbp, constr)

    return mbp
end 
"""
Dual Least Absolute Deviation: 
    min_{x} b'x 
        s.t. ||Ax||_{inf}  <= lambda
             ||x||_{inf}  <= 1

Two-block reformulation: 
    min_{x,z} b'x
        s.t. 
            Ax - z = 0 
            ||z||_{inf}  <= lambda 
            ||x||_{inf}  <= 1
"""


function generateDualLeastAbsoluteDeviation(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, lambda::Float64)
    mbp = MultiblockProblem() 
    
    numberRows, numberCols = size(A)
    @assert(numberCols == length(b), "DualLeastAbsoluteDeviation: Dimension mismatch. ")

    # x block
    block_x = BlockVariable() 
    block_x.f = AffineFunction(b, 0.0)
    block_x.g = IndicatorBox(-1 * ones(numberCols), ones(numberCols))
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
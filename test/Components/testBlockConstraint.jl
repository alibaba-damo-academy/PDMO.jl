using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../test_helper.jl")

# Import the required functions directly from Bipartization
import PDMO: blockConstraintViolation, addBlockMappingToConstraint!, addBlockMappingsToConstraint!,
                      blockConstraintViolation!, blockConstraintViolationL2Norm, blockConstraintViolationL2Norm!,
                      blockConstraintViolationLInfNorm, blockConstraintViolationLInfNorm!, checkBlockConstraintValidity

@testset "BlockConstraint Creation and Constructors" begin
    println("    ├─ Testing BlockConstraint constructors...")
    
    # Test basic construction with integer ID
    bc1 = BlockConstraint(1)
    @test bc1.id == 1
    @test isempty(bc1.involvedBlocks)
    @test isempty(bc1.mappings)
    @test bc1.rhs == 0.0
    println("    │  ├─ ✅ Integer ID constructor")
        
        # Test with string ID
        bc2 = BlockConstraint("test_constraint")
        @test bc2.id == "test_constraint"
    @test isempty(bc2.involvedBlocks)
    @test isempty(bc2.mappings)
    @test bc2.rhs == 0.0
    println("    │  ├─ ✅ String ID constructor")
    
    # Test default constructor (empty string)
    bc3 = BlockConstraint()
    @test bc3.id == ""
    @test isempty(bc3.involvedBlocks)
    @test isempty(bc3.mappings)
    @test bc3.rhs == 0.0
    println("    │  ├─ ✅ Default constructor")
    
    # Test error for negative integer ID
    @test_throws ErrorException BlockConstraint(-1)
    println("    │  ├─ ✅ Negative ID correctly rejected")
    
    # Test zero ID (should be allowed)
    bc4 = BlockConstraint(0)
    @test bc4.id == 0
    println("    │  ├─ ✅ Zero ID accepted")
    
    println("    └─ ✅ BlockConstraint creation tests completed")
end

@testset "BlockConstraint Single Mapping Management" begin
    println("    ├─ Testing single block mapping management...")
    
        bc = BlockConstraint(1)
        
    # Test adding LinearMappingMatrix
            A = sparse([1.0 2.0; 3.0 4.0])
    mapping_matrix = LinearMappingMatrix(A)
    addBlockMappingToConstraint!(bc, 1, mapping_matrix)
            
            @test 1 in bc.involvedBlocks
            @test haskey(bc.mappings, 1)
    @test bc.mappings[1] === mapping_matrix
    @test length(bc.involvedBlocks) == 1
    @test length(bc.mappings) == 1
    println("    │  ├─ ✅ LinearMappingMatrix addition")
    
    # Test adding LinearMappingIdentity to different block
    mapping_identity = LinearMappingIdentity(2.0)
    addBlockMappingToConstraint!(bc, "block2", mapping_identity)
    
    @test "block2" in bc.involvedBlocks
    @test haskey(bc.mappings, "block2")
    @test bc.mappings["block2"] === mapping_identity
    @test length(bc.involvedBlocks) == 2
    @test length(bc.mappings) == 2
    println("    │  ├─ ✅ LinearMappingIdentity addition")
    
    # Test error when adding mapping to existing block
    duplicate_mapping = LinearMappingIdentity(1.0)
    @test_throws ErrorException addBlockMappingToConstraint!(bc, 1, duplicate_mapping)
    println("    │  ├─ ✅ Duplicate mapping correctly rejected")
    
    println("    └─ ✅ Single mapping management tests completed")
end

@testset "BlockConstraint Multiple Mappings Management" begin
    println("    ├─ Testing multiple block mappings management...")
    
    bc = BlockConstraint("multi_constraint")
    
    # Create multiple mappings
    mappings_dict = Dict{Union{Int, String}, AbstractMapping}()
    mappings_dict[1] = LinearMappingMatrix(sparse([1.0 0.0; 0.0 1.0]))
    mappings_dict[2] = LinearMappingIdentity(3.0)
    mappings_dict["block3"] = LinearMappingMatrix(sparse([2.0 1.0]))
    
    # Add all mappings at once
    addBlockMappingsToConstraint!(bc, mappings_dict)
    
    @test length(bc.involvedBlocks) == 3
    @test length(bc.mappings) == 3
    @test 1 in bc.involvedBlocks
    @test 2 in bc.involvedBlocks
    @test "block3" in bc.involvedBlocks
    @test haskey(bc.mappings, 1)
    @test haskey(bc.mappings, 2)
    @test haskey(bc.mappings, "block3")
    println("    │  ├─ ✅ Multiple mappings added successfully")
    
    # Test that mappings are correctly stored
    @test bc.mappings[1] === mappings_dict[1]
    @test bc.mappings[2] === mappings_dict[2]
    @test bc.mappings["block3"] === mappings_dict["block3"]
    println("    │  ├─ ✅ Mappings correctly stored")
    
    println("    └─ ✅ Multiple mappings management tests completed")
end

@testset "BlockConstraint Violation Computation - Scalar RHS" begin
    println("    ├─ Testing constraint violation with scalar RHS...")
    
    bc = BlockConstraint("scalar_test")
    bc.rhs = 5.0
    
    # For scalar RHS, all mappings must return scalars
    # Use scalar inputs with LinearMappingIdentity to get scalar outputs
    addBlockMappingToConstraint!(bc, 1, LinearMappingIdentity(2.0))  # 2*x1
    addBlockMappingToConstraint!(bc, 2, LinearMappingIdentity(3.0))  # 3*x2
    
    # Create test data with scalars to get scalar outputs
    x_dict = Dict{Union{Int, String}, Union{Float64, AbstractArray{Float64}}}()
    x_dict[1] = 1.0  # scalar input -> scalar output: 2*1 = 2
    x_dict[2] = 2.0  # scalar input -> scalar output: 3*2 = 6
    
    # Expected: 2*1 + 3*2 - 5 = 2 + 6 - 5 = 3
    violation = blockConstraintViolation(bc, x_dict)
    @test violation isa Real
    @test violation ≈ 3.0
    @info "✅ Scalar RHS violation computed" expected=3.0 actual=violation
    
    # Test with different values
    x_dict[1] = 2.0
    x_dict[2] = 1.0
    # Expected: 2*2 + 3*1 - 5 = 4 + 3 - 5 = 2
    violation2 = blockConstraintViolation(bc, x_dict)
    @test violation2 ≈ 2.0
    println("    │  ├─ ✅ Scalar RHS violation computation verified")
    
    # Test edge case: what happens when we try to use vector-producing mappings with scalar RHS
    bc_problematic = BlockConstraint("problematic_test")
    bc_problematic.rhs = 1.0
    addBlockMappingToConstraint!(bc_problematic, 1, LinearMappingMatrix(sparse([1.0 1.0])))  # This returns a vector
    
    x_dict_problematic = Dict{Union{Int, String}, Union{Float64, AbstractArray{Float64}}}()
    x_dict_problematic[1] = [1.0, 2.0]  # This will produce a vector output [3.0]
    
    # This should fail because we're trying to add scalar to vector
    try
        violation_problematic = blockConstraintViolation(bc_problematic, x_dict_problematic)
        @test false  # Should not reach here
    catch e
        @test e isa MethodError
        println("    │  ├─ ✅ Vector-producing mapping with scalar RHS correctly fails")
    end
    
    println("    └─ ✅ Scalar RHS violation tests completed")
end

@testset "BlockConstraint Violation Computation - Vector RHS" begin
    println("    ├─ Testing constraint violation with vector RHS...")
    
    bc = BlockConstraint("vector_test")
    bc.rhs = [1.0, 2.0]  # Vector RHS
    
    # Add mappings that produce 2D output
    A1 = sparse([1.0 0.0; 0.0 2.0])  # [x1[1], 2*x1[2]]
    A2 = sparse([1.0 1.0; 0.0 1.0])  # [x2[1]+x2[2], x2[2]]
    addBlockMappingToConstraint!(bc, 1, LinearMappingMatrix(A1))
    addBlockMappingToConstraint!(bc, 2, LinearMappingMatrix(A2))
            
            # Create test data
            x_dict = Dict{Union{Int, String}, Union{Float64, AbstractArray{Float64}}}()
    x_dict[1] = [2.0, 3.0]  # A1*x1 = [2, 6]
    x_dict[2] = [1.0, 4.0]  # A2*x2 = [5, 4]
    
    # Expected violation: [2, 6] + [5, 4] - [1, 2] = [6, 8]
    violation = blockConstraintViolation(bc, x_dict)
    @test violation isa AbstractArray
    @test length(violation) == 2
    @test violation ≈ [6.0, 8.0]
    @info "✅ Vector RHS violation computed" expected=[6.0, 8.0] actual=violation
    
    # Test in-place version
    ret = similar(bc.rhs)
    blockConstraintViolation!(bc, x_dict, ret)
    @test ret ≈ [6.0, 8.0]
    println("    │  ├─ ✅ In-place vector RHS violation computation")
    
    # Test error for in-place with scalar RHS
    bc_scalar = BlockConstraint("scalar_error")
    bc_scalar.rhs = 1.0
    addBlockMappingToConstraint!(bc_scalar, 1, LinearMappingIdentity(1.0))
    ret_scalar = [0.0]
    @test_throws ErrorException blockConstraintViolation!(bc_scalar, x_dict, ret_scalar)
    println("    │  ├─ ✅ In-place error for scalar RHS correctly thrown")
    
    println("    └─ ✅ Vector RHS violation tests completed")
end

@testset "BlockConstraint Norm Computations" begin
    println("    ├─ Testing constraint violation norm computations...")
    
    bc = BlockConstraint("norm_test")
    bc.rhs = [3.0, 4.0]  # Vector RHS for norm computation
    
    # Add simple identity mapping
    addBlockMappingToConstraint!(bc, 1, LinearMappingIdentity(1.0))
    
    # Create test data that will give violation [1, 1] (so L2 norm = sqrt(2), Linf norm = 1)
    x_dict = Dict{Union{Int, String}, Union{Float64, AbstractArray{Float64}}}()
    x_dict[1] = [4.0, 5.0]  # violation = [4, 5] - [3, 4] = [1, 1]
    
    # Test L2 norm
    l2_norm = blockConstraintViolationL2Norm(bc, x_dict)
    @test l2_norm isa Float64
    @test l2_norm ≈ sqrt(2.0)
    @info "✅ L2 norm computed" expected=sqrt(2.0) actual=l2_norm
    
    # Test L2 norm in-place
    ret = similar(bc.rhs)
    l2_norm_inplace = blockConstraintViolationL2Norm!(bc, x_dict, ret)
    @test l2_norm_inplace ≈ sqrt(2.0)
    @test ret ≈ [1.0, 1.0]  # Should contain the violation
    println("    │  ├─ ✅ L2 norm in-place computation")
    
    # Test L-infinity norm
    linf_norm = blockConstraintViolationLInfNorm(bc, x_dict)
    @test linf_norm isa Float64
    @test linf_norm ≈ 1.0
    @info "✅ L-infinity norm computed" expected=1.0 actual=linf_norm
    
    # Test L-infinity norm in-place
    ret2 = similar(bc.rhs)
    linf_norm_inplace = blockConstraintViolationLInfNorm!(bc, x_dict, ret2)
    @test linf_norm_inplace ≈ 1.0
    @test ret2 ≈ [1.0, 1.0]  # Should contain the violation
    println("    │  ├─ ✅ L-infinity norm in-place computation")
    
    println("    └─ ✅ Norm computation tests completed")
end

@testset "BlockConstraint Validity Checking" begin
    println("    ├─ Testing constraint validity checking...")
    
    # Test valid constraint
    bc_valid = BlockConstraint(1)
    addBlockMappingToConstraint!(bc_valid, 1, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(bc_valid, 2, LinearMappingMatrix(sparse([1.0]')))  # 1x1 matrix
    
    @test checkBlockConstraintValidity(bc_valid) == true
    println("    │  ├─ ✅ Valid constraint correctly identified")
    
    # Test invalid constraint - negative ID
    bc_negative = BlockConstraint(1)
    bc_negative.id = -1  # Manually set to negative
    @test checkBlockConstraintValidity(bc_negative) == false
    println("    │  ├─ ✅ Negative ID correctly identified as invalid")
    
    # Test invalid constraint - mismatched lengths
    bc_mismatch = BlockConstraint(2)
    bc_mismatch.involvedBlocks = [1, 2]
    bc_mismatch.mappings[1] = LinearMappingIdentity(1.0)
    # Missing mapping for block 2, so lengths don't match
    @test checkBlockConstraintValidity(bc_mismatch) == false
    println("    │  ├─ ✅ Length mismatch correctly identified as invalid")
    
    # Test invalid constraint - less than 2 blocks
    bc_single = BlockConstraint(3)
    addBlockMappingToConstraint!(bc_single, 1, LinearMappingIdentity(1.0))
    @test checkBlockConstraintValidity(bc_single) == true
    println("    │  ├─ ✅ Single block constraint correctly identified as valid with a warning")
    
    # Test invalid constraint - missing mapping for involved block
    bc_missing = BlockConstraint(4)
    bc_missing.involvedBlocks = [1, 2]
    bc_missing.mappings[1] = LinearMappingIdentity(1.0)
    # Block 2 is in involvedBlocks but not in mappings
    @test checkBlockConstraintValidity(bc_missing) == false
    println("    │  ├─ ✅ Missing mapping correctly identified as invalid")
    
    # Test valid constraint with string IDs
    bc_string = BlockConstraint("valid_string")
    addBlockMappingToConstraint!(bc_string, "block1", LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(bc_string, "block2", LinearMappingMatrix(sparse([1.0]')))  # 1x1 matrix
    @test checkBlockConstraintValidity(bc_string) == true
    println("    │  ├─ ✅ Valid string ID constraint correctly identified")
    
    println("    └─ ✅ Validity checking tests completed")
end

@testset "BlockConstraint Complex Scenarios" begin
    println("    ├─ Testing complex constraint scenarios...")
    
    # Test constraint with multiple different mapping types
    bc_complex = BlockConstraint("complex_test")
    bc_complex.rhs = [10.0, 20.0, 30.0]
    
    # Add various mapping types
    addBlockMappingToConstraint!(bc_complex, 1, LinearMappingIdentity(2.0))
    addBlockMappingToConstraint!(bc_complex, 2, LinearMappingMatrix(sparse([1.0 1.0 1.0; 0.0 1.0 2.0; 1.0 0.0 1.0])))
    addBlockMappingToConstraint!(bc_complex, "block3", LinearMappingIdentity(-1.0))
    
    # Create test data
    x_dict = Dict{Union{Int, String}, Union{Float64, AbstractArray{Float64}}}()
    x_dict[1] = [1.0, 2.0, 3.0]  # 2*[1,2,3] = [2,4,6]
    x_dict[2] = [1.0, 2.0, 3.0]  # Matrix*[1,2,3] = [6,7,4]
    x_dict["block3"] = [1.0, 1.0, 1.0]  # -1*[1,1,1] = [-1,-1,-1]
    
    # Let's calculate step by step:
    # Block 1: 2*[1,2,3] = [2,4,6]
    # Block 2: [1 1 1; 0 1 2; 1 0 1] * [1,2,3] = [1+2+3, 0+2+6, 1+0+3] = [6,8,4]
    # Block 3: -1*[1,1,1] = [-1,-1,-1]
    # Sum: [2,4,6] + [6,8,4] + [-1,-1,-1] = [7,11,9]
    # Violation: [7,11,9] - [10,20,30] = [-3,-9,-21]
    violation = blockConstraintViolation(bc_complex, x_dict)
    @test violation ≈ [-3.0, -9.0, -21.0]
    @info "✅ Complex constraint violation computed" result=violation
    
    # Test validity
    @test checkBlockConstraintValidity(bc_complex) == true
    println("    │  ├─ ✅ Complex constraint is valid")
    
    # Test norms
    l2_norm = blockConstraintViolationL2Norm(bc_complex, x_dict)
    expected_l2 = sqrt(9 + 81 + 441)  # sqrt(3² + 9² + 21²)
    @test l2_norm ≈ expected_l2
    
    linf_norm = blockConstraintViolationLInfNorm(bc_complex, x_dict)
    @test linf_norm ≈ 21.0  # max(3, 9, 21)
    println("    │  ├─ ✅ Complex constraint norms computed")
    
    println("    └─ ✅ Complex scenarios tests completed")
end

@testset "BlockConstraint Edge Cases and Error Handling" begin
    println("    ├─ Testing edge cases and error handling...")
    
    # Test empty constraint (should be invalid)
    bc_empty = BlockConstraint("empty")
    @test checkBlockConstraintValidity(bc_empty) == false
    println("    │  ├─ ✅ Empty constraint correctly identified as invalid")
    
    # Test constraint with zero coefficient mapping
    bc_zero = BlockConstraint("zero_test")
    # Note: LinearMappingIdentity doesn't allow zero coefficient, so this tests the assertion
    @test_throws AssertionError LinearMappingIdentity(0.0)
    println("    │  ├─ ✅ Zero coefficient mapping correctly rejected")
    
    # Test constraint with very large values
    bc_large = BlockConstraint("large_test")
    bc_large.rhs = [1e10, -1e10]
    addBlockMappingToConstraint!(bc_large, 1, LinearMappingIdentity(1e6))
    addBlockMappingToConstraint!(bc_large, 2, LinearMappingIdentity(-1e6))
    
    x_dict_large = Dict{Union{Int, String}, Union{Float64, AbstractArray{Float64}}}()
    x_dict_large[1] = [1e4, 2e4]
    x_dict_large[2] = [1e4, 2e4]
    
    violation_large = blockConstraintViolation(bc_large, x_dict_large)
    @test violation_large isa AbstractArray
    @test isfinite(violation_large[1]) && isfinite(violation_large[2])
    println("    │  ├─ ✅ Large value constraint handled correctly")
    
    # Test constraint with mixed ID types
    bc_mixed = BlockConstraint("mixed_ids")
    addBlockMappingToConstraint!(bc_mixed, 1, LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(bc_mixed, "string_block", LinearMappingIdentity(1.0))
    addBlockMappingToConstraint!(bc_mixed, 42, LinearMappingIdentity(1.0))
    
    @test length(bc_mixed.involvedBlocks) == 3
    @test 1 in bc_mixed.involvedBlocks
    @test "string_block" in bc_mixed.involvedBlocks
    @test 42 in bc_mixed.involvedBlocks
    @test checkBlockConstraintValidity(bc_mixed) == true
    println("    │  ├─ ✅ Mixed ID types handled correctly")
    
    println("    └─ ✅ Edge cases and error handling tests completed")
end 
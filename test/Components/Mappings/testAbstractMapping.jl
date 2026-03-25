using Test
using PDMO
using LinearAlgebra
using SparseArrays
include("../../test_helper.jl")

# Import the adjoint! function from PDMO
import PDMO: adjoint!
import PDMO: adjoint

@testset "AbstractMapping Tests" begin
    
    @testset "Abstract Type Interface" begin
        # Test that AbstractMapping is defined
        @test AbstractMapping isa Type
        
        # Test that all concrete mappings are subtypes
        @test NullMapping <: AbstractMapping
        @test LinearMappingIdentity <: AbstractMapping
        @test LinearMappingExtraction <: AbstractMapping
        @test LinearMappingMatrix <: AbstractMapping
    end
    
    @testset "NullMapping Tests" begin
        # Test constructor
        null_map = NullMapping()
        @test null_map isa NullMapping
        @test null_map isa AbstractMapping
        
        # NullMapping is just a placeholder, so basic construction is the main test
        @test typeof(null_map) == NullMapping
    end
    
    @testset "LinearMappingIdentity Tests" begin
        @testset "Constructor" begin
            # Test default constructor
            id_map = LinearMappingIdentity()
            @test id_map isa LinearMappingIdentity
            @test id_map.coe == 1.0
            
            # Test with custom coefficient
            id_map2 = LinearMappingIdentity(2.5)
            @test id_map2.coe == 2.5
            
            # Test with negative coefficient
            id_map3 = LinearMappingIdentity(-1.0)
            @test id_map3.coe == -1.0
            
            # Test error for zero coefficient
            @test_throws AssertionError LinearMappingIdentity(0.0)
        end
        
        @testset "Forward Mapping - Out-of-place" begin
            id_map = LinearMappingIdentity(2.0)
            
            # Test with vector
            x_vec = [1.0, 2.0, 3.0]
            y_vec = id_map(x_vec)
            @test y_vec ≈ [2.0, 4.0, 6.0]
            @test size(y_vec) == size(x_vec)
            
            # Test with matrix
            x_mat = [1.0 2.0; 3.0 4.0]
            y_mat = id_map(x_mat)
            @test y_mat ≈ [2.0 4.0; 6.0 8.0]
            @test size(y_mat) == size(x_mat)
            
            # Test with coefficient 1.0 (optimization case)
            id_map_one = LinearMappingIdentity(1.0)
            y_one = id_map_one(x_vec)
            @test y_one ≈ x_vec

            # Allocating API should not alias input for arrays.
            y_alias_check = id_map_one(x_vec)
            y_alias_check[1] = 999.0
            @test x_vec[1] == 1.0
        end
        
        @testset "Forward Mapping - In-place" begin
            id_map = LinearMappingIdentity(3.0)
            
            # Test with vector - overwrite
            x_vec = [1.0, 2.0, 3.0]
            ret_vec = similar(x_vec)
            id_map(x_vec, ret_vec, false)
            @test ret_vec ≈ [3.0, 6.0, 9.0]
            
            # Test with vector - add
            ret_vec = [1.0, 1.0, 1.0]
            id_map(x_vec, ret_vec, true)
            @test ret_vec ≈ [4.0, 7.0, 10.0]  # [1,1,1] + [3,6,9]
            
            # Test with matrix
            x_mat = [1.0 2.0; 3.0 4.0]
            ret_mat = similar(x_mat)
            id_map(x_mat, ret_mat, false)
            @test ret_mat ≈ [3.0 6.0; 9.0 12.0]
            
            # Test error for scalar input (in-place not supported)
            @test_throws ErrorException id_map(1.0, 1.0, false)
            
            # Test dimension mismatch error
            x_wrong = [1.0, 2.0]
            ret_wrong = [1.0, 2.0, 3.0]
            @test_throws AssertionError id_map(x_wrong, ret_wrong, false)
        end
        
        @testset "Adjoint Operations" begin
            id_map = LinearMappingIdentity(2.0)
            
            # Test adjoint out-of-place using the mapping's adjoint! method
            y = [2.0, 4.0, 6.0]
            x_adj = similar(y)
            adjoint!(id_map, y, x_adj, false)
            @test x_adj ≈ [4.0, 8.0, 12.0]  # 2.0 * y
            
            # Test adjoint in-place - add
            ret_adj = [1.0, 1.0, 1.0]
            adjoint!(id_map, y, ret_adj, true)
            @test ret_adj ≈ [5.0, 9.0, 13.0]  # [1,1,1] + [4,8,12]

            # Allocating adjoint should not alias input for arrays.
            id_map_one = LinearMappingIdentity(1.0)
            y_src = [2.0, 4.0, 6.0]
            y_adj = adjoint(id_map_one, y_src)
            y_adj[1] = -123.0
            @test y_src[1] == 2.0
        end
        
        @testset "Adjoint Mapping Creation" begin
            id_map = LinearMappingIdentity(3.0)
            adj_map = createAdjointMapping(id_map)
            
            @test adj_map isa LinearMappingIdentity
            @test adj_map.coe == 3.0  # For identity, adjoint has same coefficient
            
            # Test that adjoint mapping works correctly
            x = [1.0, 2.0]
            y_original = id_map(x)
            y_adjoint = adj_map(x)
            @test y_original ≈ y_adjoint
        end
        
        @testset "Operator Norm" begin
            id_map = LinearMappingIdentity(2.0)
            norm_val = operatorNorm2(id_map)
            @test norm_val ≈ 2.0  # For scaled identity, norm is |coefficient|
            
            id_map_neg = LinearMappingIdentity(-3.0)
            norm_neg = operatorNorm2(id_map_neg)
            @test norm_neg ≈ 3.0  # Absolute value
        end
    end
    
    @testset "LinearMappingExtraction Tests" begin
        @testset "Constructor" begin
            # Test valid construction
            dim = (5, 3)
            coe = 2.0
            indexStart = 2
            indexEnd = 4
            ext_map = LinearMappingExtraction(dim, coe, indexStart, indexEnd)
            
            @test ext_map isa LinearMappingExtraction
            @test ext_map.dim == dim
            @test ext_map.coe == coe
            @test ext_map.indexStart == indexStart
            @test ext_map.indexEnd == indexEnd
            
            # Test error cases
            @test_throws AssertionError LinearMappingExtraction((5, 3), 2.0, 0, 3)  # indexStart < 1
            @test_throws AssertionError LinearMappingExtraction((5, 3), 2.0, 3, 6)  # indexEnd > dim[1]
            @test_throws AssertionError LinearMappingExtraction((5, 3), 2.0, 4, 3)  # indexStart > indexEnd
            @test_throws AssertionError LinearMappingExtraction((5, 3), 0.0, 2, 4)  # zero coefficient
        end
        
        @testset "Forward Mapping - Vector Input" begin
            dim = (4,)
            ext_map = LinearMappingExtraction(dim, 2.0, 2, 3)
            
            # Test extraction from vector
            x = [1.0, 2.0, 3.0, 4.0]
            y = ext_map(x)
            @test y ≈ [4.0, 6.0]  # 2.0 * [2.0, 3.0]
            @test length(y) == 2
            
            # Test in-place version
            ret = similar(y)
            ext_map(x, ret, false)
            @test ret ≈ [4.0, 6.0]
            
            # Test add version
            ret = [1.0, 1.0]
            ext_map(x, ret, true)
            @test ret ≈ [5.0, 7.0]  # [1,1] + [4,6]
        end
        
        @testset "Adjoint Operations" begin
            dim = (4,)
            ext_map = LinearMappingExtraction(dim, 2.0, 2, 3)
            
            # Test adjoint - should place values back in original positions
            y = [4.0, 6.0]
            x_adj = zeros(4)
            adjoint!(ext_map, y, x_adj, false)
            @test x_adj ≈ [0.0, 8.0, 12.0, 0.0]  # 2.0 * y placed at positions 2:3
            @test length(x_adj) == 4
            
            # Test adjoint add
            ret_adj = ones(4)
            adjoint!(ext_map, y, ret_adj, true)
            @test ret_adj ≈ [1.0, 9.0, 13.0, 1.0]  # [1,1,1,1] + [0,8,12,0]
        end
        
        @testset "Operator Norm" begin
            dim = (4,)
            ext_map = LinearMappingExtraction(dim, 3.0, 2, 4)
            norm_val = operatorNorm2(ext_map)
            @test norm_val ≈ 3.0  # For extraction, norm is |coefficient|
        end
    end

    @testset "Mapping Composition Adjoint Tests" begin
        # Regression test: A' * A for extraction mappings must be an operator
        # on the original space (not another extraction mapping).
        A = LinearMappingExtraction((4,), 2.0, 2, 3)
        A_adj_A = adjoint(A, A)
        @test A_adj_A isa LinearMappingMatrix

        x = [1.0, 2.0, 3.0, 4.0]
        y = A_adj_A(x)
        @test y ≈ [0.0, 8.0, 12.0, 0.0]
    end
    
    @testset "LinearMappingMatrix Tests" begin
        @testset "Constructor" begin
            # Test with sparse matrix
            A = sparse([1.0 2.0 0.0; 0.0 3.0 4.0])
            mat_map = LinearMappingMatrix(A)
            
            @test mat_map isa LinearMappingMatrix
            @test mat_map.A == A
            @test mat_map.inputDim == 3
            @test mat_map.outputDim == 2
        end
        
        @testset "Forward Mapping - Vector Input" begin
            A = sparse([1.0 2.0; 3.0 4.0])
            mat_map = LinearMappingMatrix(A)
            
            # Test matrix-vector multiplication
            x = [1.0, 2.0]
            y = mat_map(x)
            expected = A * x  # [5.0, 11.0]
            @test y ≈ expected
            @test length(y) == 2
            
            # Test in-place version
            ret = similar(y)
            mat_map(x, ret, false)
            @test ret ≈ expected
            
            # Test add version
            ret = [1.0, 1.0]
            mat_map(x, ret, true)
            @test ret ≈ expected + [1.0, 1.0]
        end
        
        @testset "Forward Mapping - Matrix Input" begin
            A = sparse([1.0 2.0; 3.0 4.0])
            mat_map = LinearMappingMatrix(A)
            
            # Test matrix-matrix multiplication
            X = [1.0 2.0; 3.0 4.0]  # 2x2 input
            Y = mat_map(X)
            expected = A * X
            @test Y ≈ expected
            @test size(Y) == (2, 2)
            
            # Test in-place version
            ret = similar(Y)
            mat_map(X, ret, false)
            @test ret ≈ expected
        end
        
        @testset "Adjoint Operations" begin
            A = sparse([1.0 2.0; 3.0 4.0])
            mat_map = LinearMappingMatrix(A)
            
            # Test adjoint using adjoint! method
            y = [1.0, 2.0]
            x_adj = zeros(2)
            adjoint!(mat_map, y, x_adj, false)
            expected = A' * y  # [7.0, 10.0]
            @test x_adj ≈ expected
            
            # Test adjoint add
            ret_adj = [1.0, 1.0]
            adjoint!(mat_map, y, ret_adj, true)
            @test ret_adj ≈ expected + [1.0, 1.0]
        end
        
        @testset "Adjoint Mapping Creation" begin
            A = sparse([1.0 2.0; 3.0 4.0])
            mat_map = LinearMappingMatrix(A)
            adj_map = createAdjointMapping(mat_map)
            
            @test adj_map isa LinearMappingMatrix
            @test adj_map.A ≈ sparse(A')  # Should work now with the fixed createAdjointMapping
            @test adj_map.inputDim == mat_map.outputDim
            @test adj_map.outputDim == mat_map.inputDim
        end
        
        @testset "Operator Norm" begin
            # Test with simple matrix
            A = sparse([3.0 0.0; 0.0 4.0])  # Diagonal matrix
            mat_map = LinearMappingMatrix(A)
            norm_val = operatorNorm2(mat_map)
            @test norm_val ≈ 4.0  # Largest singular value
            
            # Test with more complex matrix
            A2 = sparse([1.0 1.0; 1.0 1.0])  # Rank 1 matrix
            mat_map2 = LinearMappingMatrix(A2)
            norm_val2 = operatorNorm2(mat_map2)
            @test norm_val2 ≈ 2.0  # sqrt(2) * sqrt(2) = 2
        end
        
        @testset "Edge Cases" begin
            # Test with rectangular matrix
            A_rect = sparse([1.0 2.0 3.0; 4.0 5.0 6.0])  # 2x3
            mat_map_rect = LinearMappingMatrix(A_rect)
            
            x_rect = [1.0, 2.0, 3.0]
            y_rect = mat_map_rect(x_rect)
            @test y_rect ≈ A_rect * x_rect
            @test length(y_rect) == 2
            
            # Test with zero matrix
            A_zero = spzeros(2, 2)
            mat_map_zero = LinearMappingMatrix(A_zero)
            x_zero = [1.0, 2.0]
            y_zero = mat_map_zero(x_zero)
            @test y_zero ≈ [0.0, 0.0]
        end
    end

    @testset "LinearMappingIdentityStacking Tests" begin
        L = LinearMappingIdentityStacking(2.0, 3)
        x = [1.0, 2.0]

        y = L(x)
        @test y ≈ [2.0, 4.0, 2.0, 4.0, 2.0, 4.0]

        y_inplace = zeros(6)
        L(x, y_inplace, false)
        @test y_inplace ≈ y

        y_add = ones(6)
        L(x, y_add, true)
        @test y_add ≈ ones(6) .+ y

        x_adj = adjoint(L, y)
        @test x_adj ≈ [12.0, 24.0]

        x_adj_inplace = zeros(2)
        adjoint!(L, y, x_adj_inplace, false)
        @test x_adj_inplace ≈ x_adj

        @test operatorNorm2(L) ≈ 2.0 * sqrt(3.0)
    end

    @testset "LinearMappingStacking Tests" begin
        mappings = AbstractMapping[LinearMappingIdentity(1.0), LinearMappingIdentity(2.0)]
        L = LinearMappingStacking(mappings, [2, 2])
        x = [1.0, 2.0]

        y = L(x)
        @test y ≈ [1.0, 2.0, 2.0, 4.0]

        y_inplace = zeros(4)
        L(x, y_inplace, false)
        @test y_inplace ≈ y

        y_add = ones(4)
        L(x, y_add, true)
        @test y_add ≈ ones(4) .+ y

        x_adj = adjoint(L, y)
        @test x_adj ≈ [5.0, 10.0]  # [1,2] + 2*[2,4]
        @test y ≈ [1.0, 2.0, 2.0, 4.0]  # allocating adjoint must not mutate input

        x_adj_inplace = zeros(2)
        adjoint!(L, y, x_adj_inplace, false)
        @test x_adj_inplace ≈ x_adj

        @test operatorNorm2(L) ≈ sqrt(1.0^2 + 2.0^2)
    end
    
    @testset "Mathematical Properties" begin
        # Test linearity property for all mappings
        @testset "Linearity Tests" begin
            # Test LinearMappingIdentity
            id_map = LinearMappingIdentity(2.0)
            x1 = [1.0, 2.0]
            x2 = [3.0, 4.0]
            α = 0.3
            
            # L(αx1 + (1-α)x2) = αL(x1) + (1-α)L(x2)
            lhs = id_map(α * x1 + (1-α) * x2)
            rhs = α * id_map(x1) + (1-α) * id_map(x2)
            @test lhs ≈ rhs
            
            # Test LinearMappingMatrix
            A = sparse([1.0 2.0; 3.0 4.0])
            mat_map = LinearMappingMatrix(A)
            
            lhs_mat = mat_map(α * x1 + (1-α) * x2)
            rhs_mat = α * mat_map(x1) + (1-α) * mat_map(x2)
            @test lhs_mat ≈ rhs_mat
        end
        
        @testset "Adjoint Properties" begin
            # Test that (L*)* = L for identity mapping
            id_map = LinearMappingIdentity(3.0)
            adj_map = createAdjointMapping(id_map)
            adj_adj_map = createAdjointMapping(adj_map)
            
            x = [1.0, 2.0, 3.0]
            @test id_map(x) ≈ adj_adj_map(x)
            
            # Test adjoint property: <Lx, y> = <x, L*y> using adjoint! method
            A = sparse([1.0 2.0; 3.0 4.0])
            mat_map = LinearMappingMatrix(A)
            
            x = [1.0, 2.0]
            y = [1.0, 1.0]
            
            Lx = mat_map(x)
            L_star_y = zeros(2)
            adjoint!(mat_map, y, L_star_y, false)
            
            lhs = dot(Lx, y)
            rhs = dot(x, L_star_y)
            @test lhs ≈ rhs atol=1e-10
        end
    end
end 
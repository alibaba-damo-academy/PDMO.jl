using Test
using PDMO
include("test_helper.jl")

println("="^80)
println("🧪 Starting PDMO.jl Test Suite")
println("="^80)

# Include test files
@testset verbose=true "PDMO.jl Tests" begin
    
    # COMPONENTS TESTS
    @info "📦 COMPONENTS TESTS" description="Testing core components: BlockVariable, BlockConstraint, Functions, Mappings"
    components_testset = @testset verbose=true "Components Tests" begin
        
        @info "Testing BlockVariable functionality..."
        @testset "BlockVariable Tests" begin
            include("Components/testBlockVariable.jl")
        end
        
        @info "Testing BlockConstraint functionality..."
        @testset "BlockConstraint Tests" begin
            include("Components/testBlockConstraint.jl")
        end
        
        @info "Testing Abstract Function interface..."
        @testset "AbstractFunction Tests" begin
            include("Components/Functions/testAbstractFunction.jl")
        end
        
        @info "Testing Abstract Multiblock Function interface..."
        @testset "AbstractMultiblockFunction Tests" begin
            include("Components/Functions/testAbstractMultiblockFunction.jl")
        end
        
        @info "Testing Quadratic Multiblock Function implementation..."
        @testset "QuadraticMultiblockFunction Tests" begin
            include("Components/Functions/testQuadraticMultiblockFunction.jl")
        end
        
        @info "Testing Abstract Mapping interface..."
        @testset "AbstractMapping Tests" begin
            include("Components/Mappings/testAbstractMapping.jl")
        end
    end
    
    # Count and report components tests
    test_counter.component_tests = count_tests_in_testset(components_testset)
    @info "✅ COMPONENTS TESTS COMPLETED" tests_executed=test_counter.component_tests
    
    # FORMULATIONS TESTS
    @info "🏗️ FORMULATIONS TESTS" description="Testing problem formulations: MultiblockProblem, Graph, Bipartization Algorithms"
    formulations_testset = @testset verbose=true "Formulations Tests" begin
        
        @info "Testing MultiblockProblem formulation..."
        @testset "MultiblockProblem Tests" begin
            include("Formulations/testMultiblockProblem.jl")
        end
        
        @info "Testing Constraint to Quadratic Transformation..."
        @testset "ConstraintToQuadratic Tests" begin
            include("Formulations/testConstraintToQuadraticTransformation.jl")
        end
        
        @info "Testing MultiblockGraph representation..."
        @testset "MultiblockGraph Tests" begin
            include("Formulations/testMultiblockGraph.jl")
        end
        
        @info "Testing Bipartization Algorithms..."
        @testset "BipartizationAlgorithms Tests" begin
            include("Formulations/testBipartizationAlgorithms.jl")
        end
        
        @info "Testing ADMM Bipartite Graph..."
        @testset "AdmmBipartiteGraph Tests" begin
            include("Formulations/testAdmmBipartiteGraph.jl")
        end
    end
    
    # Count and report formulations tests
    test_counter.formulation_tests = count_tests_in_testset(formulations_testset)
    @info "✅ FORMULATIONS TESTS COMPLETED" tests_executed=test_counter.formulation_tests
    
    # ALGORITHMS TESTS
    @info "🔧 ALGORITHMS TESTS" description="Testing optimization algorithms: BCD, ADMM, AdaPDM"
    algorithms_testset = @testset verbose=true "Algorithms Tests" begin
        
        @info "Testing ADMM Algorithm..."
        @testset "ADMM Tests" begin
            include("Algorithms/ADMM/testBipartiteADMM.jl")
        end

        @info "Testing BCD Algorithm..."
        @testset "BCD Tests" begin
            include("Algorithms/BCD/testBCDTwoBlockQP.jl")
        end
        
        @info "Testing AdaPDM Algorithm..."
        @testset "AdaPDM Tests" begin
            include("Algorithms/AdaPDM/testAdaPDM.jl")
        end
    end
    
    # Count and report algorithms tests
    test_counter.algorithm_tests = count_tests_in_testset(algorithms_testset)
    @info "✅ ALGORITHMS TESTS COMPLETED" tests_executed=test_counter.algorithm_tests
    
    # UTILITIES TESTS
    @info "🧰 UTILITIES TESTS" description="Testing utility modules: DEC parser and I/O helpers"
    utilities_testset = @testset verbose=true "Utilities Tests" begin
        @info "Testing DEC parser utility..."
        @testset "DEC Parser Tests" begin
            include("Util/testDEC.jl")
        end
    end
    
    # Count and report utilities tests
    test_counter.utility_tests = count_tests_in_testset(utilities_testset)
    @info "✅ UTILITIES TESTS COMPLETED" tests_executed=test_counter.utility_tests
end

# Print final comprehensive summary
# print_final_summary()

println("\n" * "="^80)
println("✅ PDMO.jl Test Suite Complete")
println("="^80) 
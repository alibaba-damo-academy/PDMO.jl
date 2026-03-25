using Test
using PDMO
using SparseArrays

@testset "DEC Parser Tests" begin
    @testset "Infers block count without NBLOCKS" begin
        A = sparse([1.0 0.0 2.0; 0.0 3.0 0.0])
        rowNames = ["r1", "r2"]

        dec_content = """
        PRESOLVED
        1
        BLOCK 1
        r1
        BLOCK 2
        r2
        """

        filename = tempname() * ".dec"
        open(filename, "w") do io
            write(io, dec_content)
        end

        dec = parseDEC(A, rowNames, filename; logLevel=0)
        @test dec !== nothing
        @test dec.numberBlocks == 2
        @test get(dec.mapBlock2Rows, 1, Int[]) == [1]
        @test get(dec.mapBlock2Rows, 2, Int[]) == [2]
        @test get(dec.mapBlock2Columns, 1, Int[]) == [1, 3]
        @test get(dec.mapBlock2Columns, 2, Int[]) == [2]
    end

    @testset "Remaps non-consecutive block IDs" begin
        A = sparse([1.0 0.0; 0.0 1.0])
        rowNames = ["r1", "r2"]

        dec_content = """
        NBLOCKS
        2
        BLOCK 3
        r1
        BLOCK 7
        r2
        """

        filename = tempname() * ".dec"
        open(filename, "w") do io
            write(io, dec_content)
        end

        dec = parseDEC(A, rowNames, filename; logLevel=0)
        @test dec !== nothing
        @test dec.numberBlocks == 2
        @test haskey(dec.mapBlock2Rows, 1)
        @test haskey(dec.mapBlock2Rows, 2)
        @test get(dec.mapBlock2Rows, 1, Int[]) == [1]
        @test get(dec.mapBlock2Rows, 2, Int[]) == [2]
    end
end

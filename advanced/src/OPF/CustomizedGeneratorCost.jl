struct CustomizedGeneratorCost <: AbstractFunction 
    # original cost is quadratic ax^2 + bx + c
    a::Float64 
    b::Float64 
    c::Float64 

    function CustomizedGeneratorCost(a::Float64, b::Float64, c::Float64)
        new(0, 1000.0 * a, c)
    end 
end 
    
isSmooth(::Type{CustomizedGeneratorCost}) = true
isConvex(::Type{CustomizedGeneratorCost}) = true
isSupportedByJuMP(::Type{CustomizedGeneratorCost}) = true

# function evaluation
function (f::CustomizedGeneratorCost)(x::Float64, enableParallel::Bool=false)
    return  f.a * x^2 + f.b * x + f.c
end 

# gradient oracle
function gradientOracle(f::CustomizedGeneratorCost, x::Float64, enableParallel::Bool=false)
    return 2.0 * f.a * x + f.b
end 

# how to model this function in JuMP
function JuMPAddSmoothFunction(f::CustomizedGeneratorCost, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    @assert length(var) == 1 "CustomizedGeneratorCost: dimension must be 1, got $(length(var))"
    return abs(f.a) == 0.0 ? (f.b * var[1] + f.c) : (f.a * var[1]^2 + f.b * var[1] + f.c)
end 


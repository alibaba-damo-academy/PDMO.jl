struct CustomizedZoneCost <: AbstractFunction 
    numberColumns::Int64 
    generatorIndices::Vector{Int64}
    generatorCosts::Vector{CustomizedGeneratorCost}

    function CustomizedZoneCost(numberColumns::Int64, generatorIndices::Vector{Int64}, generatorCosts::Vector{CustomizedGeneratorCost})
        new(numberColumns, generatorIndices, generatorCosts)
    end 
end 


isSmooth(::Type{CustomizedZoneCost}) = true
isConvex(::Type{CustomizedZoneCost}) = true
isSupportedByJuMP(::Type{CustomizedZoneCost}) = true

function (f::CustomizedZoneCost)(x::Vector{Float64}, enableParallel::Bool=false)
    return length(f.generatorIndices) == 0 ? 0.0 : sum(f.generatorCosts[i](x[f.generatorIndices[i]]) for i in 1:length(f.generatorIndices))
end 

function gradientOracle!(grad::Vector{Float64}, f::CustomizedZoneCost, x::Vector{Float64}, enableParallel::Bool=false)
    fill!(grad, 0.0)
    for i in 1:length(f.generatorIndices)
       grad[f.generatorIndices[i]] = gradientOracle(f.generatorCosts[i], x[f.generatorIndices[i]])
    end 
end 

function gradientOracle(f::CustomizedZoneCost, x::Vector{Float64}, enableParallel::Bool=false)
    grad = zeros(f.numberColumns)
    gradientOracle!(grad, f, x, enableParallel)
    return grad
end 

function JuMPAddSmoothFunction(f::CustomizedZoneCost, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    return length(f.generatorIndices) == 0 ? 0.0 : sum(JuMPAddSmoothFunction(f.generatorCosts[i], model, [var[f.generatorIndices[i]]]) for i in 1:length(f.generatorIndices))
end
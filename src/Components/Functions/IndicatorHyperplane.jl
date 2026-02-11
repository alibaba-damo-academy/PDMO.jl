"""
    IndicatorHyperplane(slope::Vector{Float64}, intercept::Float64)

Represents the indicator function of a hyperplane defined by ⟨slope, x⟩ = intercept.

# Mathematical Definition
The indicator function of the hyperplane H = {x : ⟨a,x⟩ = b}:

```math
I_H(x) = \\begin{cases}
0 & \\text{if } \\langle a,x \\rangle = b \\\\
+\\infty & \\text{otherwise}
\\end{cases}
```

where a is the normal vector (slope) and b is the intercept.

# Arguments
- `slope::Vector{Float64}`: Normal vector of the hyperplane (must be non-zero)
- `intercept::Float64`: Right-hand side of the hyperplane equation

# Properties
- **Smooth**: No, not differentiable (indicator function)
- **Convex**: Yes, indicator functions of convex sets are convex
- **Proximal**: Yes, proximal operator is projection onto the hyperplane
- **Set Indicator**: Yes, represents the constraint set ⟨a,x⟩ = b

# Mathematical Properties
- **Hyperplane equation**: ⟨slope, x⟩ = intercept
- **Normal vector**: slope vector points orthogonal to the hyperplane
- **Proximal operator (projection)**: P_H(x) = x - (⟨a,x⟩ - b) · a/||a||²

# Geometric Interpretation
- **Hyperplane**: (n-1)-dimensional affine subspace in ℝⁿ
- **Normal vector**: slope defines the orientation of the hyperplane
- **Distance from origin**: |intercept|/||slope|| when slope is normalized
- **Projection**: Orthogonal projection onto the hyperplane

# Projection Formula
The projection of point x onto hyperplane H = {z : ⟨a,z⟩ = b} is:
```
P_H(x) = x - ((⟨a,x⟩ - b) / ||a||²) · a
```

This formula:
1. Computes the signed distance from x to the hyperplane: (⟨a,x⟩ - b) / ||a||
2. Moves x by this distance in the negative normal direction: a / ||a||²

# Special Cases
- **Origin-centered hyperplane**: intercept = 0 gives ⟨a,x⟩ = 0
- **Coordinate hyperplane**: slope = eᵢ gives xᵢ = intercept
- **45-degree line (2D)**: slope = [1,1] gives x₁ + x₂ = intercept

# Numerical Considerations
- **Non-zero slope**: Constructor validates ||slope|| > ZeroTolerance
- **Scaling invariance**: (α·slope, α·intercept) represents the same hyperplane
- **Numerical stability**: Pre-computed scaledSlope avoids numerical issues
- **Tolerance**: Uses FeasTolerance for feasibility checking

# Relationship to Other Constraints
- **Linear subspace**: IndicatorLinearSubspace for Ax = b (multiple equations)
- **Half-space**: Related to ⟨a,x⟩ ≤ b constraints
- **Box constraints**: Coordinate hyperplanes form box constraint boundaries
- **Polytopes**: Intersection of multiple hyperplanes and half-spaces
"""
struct IndicatorHyperplane <: AbstractFunction 
    slope::Vector{Float64}
    intercept::Float64
    scaledSlope::Vector{Float64}  # Precomputed slope/‖slope‖²

    function IndicatorHyperplane(slope::Vector{Float64}, intercept::Float64)
        @assert(length(slope) > 0, "IndicatorHyperplane: slope vector cannot be empty")
        
        # Compute norm squared
        norm_sq = dot(slope, slope)
        
        @assert(norm_sq > ZeroTolerance, "IndicatorHyperplane: slope vector cannot be zero")
        
        # Precompute scaled slope for efficiency
        scaledSlope = slope ./ norm_sq
        
        new(slope, intercept, scaledSlope)
    end 
end 

# Override traits for IndicatorHyperplane
isProximal(::Type{IndicatorHyperplane}) = true 
isConvex(::Type{IndicatorHyperplane}) = true
isSet(::Type{IndicatorHyperplane}) = true
isSupportedByJuMP(f::Type{<:IndicatorHyperplane}) = true

# function value
function (f::IndicatorHyperplane)(x::NumericVariable, enableParallel::Bool=false)
    @assert(length(x) == length(f.slope), "IndicatorHyperplane: dimension mismatch.")
    residual = dot(f.slope, x) - f.intercept
    return abs(residual) <= FeasTolerance * 100.0 ? 0.0 : Inf
end 

# proximal oracle
function proximalOracle!(y::NumericVariable, f::IndicatorHyperplane, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    @assert(length(x) == length(f.slope), "IndicatorHyperplane: dimension mismatch.")

    # Compute projection: y = x - (<a,x> - b)a/‖a‖²
    # Use precomputed scaled_slope to avoid division
    residual = dot(f.slope, x) - f.intercept
    y .= x .- residual * f.scaledSlope
end

# proximal oracle (out-of-place)
function proximalOracle(f::IndicatorHyperplane, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    y = similar(x)
    proximalOracle!(y, f, x, gamma, enableParallel)
    return y
end

# JuMP support
function JuMPAddProximableFunction(g::IndicatorHyperplane, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    @assert length(var) == length(g.slope) "Variable dimension must match hyperplane slope dimension"
    
    # Add hyperplane constraint: slope' * x = intercept
    JuMP.@constraint(model, dot(g.slope, var) == g.intercept)
    
    return nothing  # Hyperplane constraints don't contribute to objective
end



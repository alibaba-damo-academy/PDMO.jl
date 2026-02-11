"""
    IndicatorBox(lb::NumericVariable, ub::NumericVariable)

Indicator function of a box constraint set [lb, ub].

# Mathematical Definition
The indicator function of the box constraint set:
{x : lb ≤ x ≤ ub} (element-wise inequalities)

f(x) = 0    if lb ≤ x ≤ ub (element-wise)
f(x) = +∞   otherwise

# Arguments
- `lb::NumericVariable`: Lower bound vector/scalar
- `ub::NumericVariable`: Upper bound vector/scalar

# Properties
- **Smooth**: No, not differentiable on the boundary
- **Convex**: Yes, indicator functions of convex sets are convex
- **Proximal**: Yes, has explicit proximal operator (projection onto box)
- **Set Indicator**: Yes, this is an indicator function

# Mathematical Properties
- **Proximal Operator**: Element-wise projection onto the box
  - prox_f(x) = clamp(x, lb, ub) = max(lb, min(x, ub))

"""
struct IndicatorBox <: AbstractFunction 
    lb::NumericVariable
    ub::NumericVariable

    function IndicatorBox(lb::NumericVariable, ub::NumericVariable)
        if size(lb) != size(ub)
            error("IndicatorBox: lb and ub have different shapes.")
        end
        if any(lb .> ub)
            error("IndicatorBox: infeasible domain; some lb > ub. ")
        end 
        new(lb, ub)
    end 
end 

# Override traits for IndicatorBox
isProximal(f::Type{<:IndicatorBox}) = true 
isConvex(f::Type{<:IndicatorBox}) = true
isSet(f::Type{<:IndicatorBox}) = true
isSupportedByJuMP(f::Type{<:IndicatorBox}) = true

# function value
function (f::IndicatorBox)(x::NumericVariable, enableParallel::Bool=false)
    # if size(x) != size(f.lb)
    #     error("IndicatorBox: function evaluation encountered dimension mismatch.")
    # end
    
    if isa(x, Number)
        if x < f.lb - FeasTolerance * 100.0 || x > f.ub + FeasTolerance * 100.0
            return Inf
        else
            return 0.0
        end
    else
        for k in eachindex(x)
            if x[k] < f.lb[k] - FeasTolerance * 100.0 || x[k] > f.ub[k] + FeasTolerance * 100.0
                return Inf
            end
        end
        return 0.0
    end
end

# proximal oracle
function proximalOracle!(y::NumericVariable, f::IndicatorBox, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    # if size(x) != size(f.lb) || size(y) != size(f.lb)
    #     error("IndicatorBox: proximal oracle encountered dimension mismatch.")
    # end
    
    if isa(x, Number)
        error("IndicatorBox: proximal oracle does not support in-place operations for scalar inputs.")
    end
    y .= clamp.(x, f.lb, f.ub)
end

function proximalOracle(f::IndicatorBox, x::NumericVariable, gamma::Float64 = 1.0, enableParallel::Bool=false)
    # if size(x) != size(f.lb)
    #     error("IndicatorBox: proximal oracle encountered dimension mismatch.")
    # end
    
    return isa(x, Number) ? clamp(x, f.lb, f.ub) : clamp.(x, f.lb, f.ub)
end

# JuMP support
function JuMPAddProximableFunction(g::IndicatorBox, model::JuMP.Model, var::Vector{<:JuMP.VariableRef})
    # Set box bounds on the provided variables
    if isa(g.lb, Number) && isa(g.ub, Number)
        # Scalar bounds - apply to all variables
        for k in 1:length(var)
            if g.lb > -Inf 
                JuMP.set_lower_bound(var[k], g.lb)
            end 
            if g.ub < Inf 
                JuMP.set_upper_bound(var[k], g.ub)
            end 
        end
    else
        # Vector bounds - apply element-wise
        @assert length(var) == length(g.lb) "IndicatorBox: variable dimension must match bounds dimension"
        for k in 1:length(var)
            if g.lb[k] > -Inf 
                JuMP.set_lower_bound(var[k], g.lb[k])
            end 
            if g.ub[k] < Inf 
                JuMP.set_upper_bound(var[k], g.ub[k])
            end 
        end
    end
    
    return nothing  # Box constraints don't contribute to objective
end

 
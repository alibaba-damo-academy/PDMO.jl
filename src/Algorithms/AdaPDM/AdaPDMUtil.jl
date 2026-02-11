"""
    AdaPDMLog(iter, info::AdaPDMIterationInfo, param::AbstractAdaPDMParam; final::Bool = false)

Log iteration information for the Adaptive Primal-Dual Method algorithms.

This function provides formatted output of algorithm progress including objective values,
residuals, and timing information. It handles both regular iteration logging and final
summary logging.

# Arguments
- `iter::Int`: Current iteration number (0 for initialization)
- `info::AdaPDMIterationInfo`: Current iteration information containing residuals and objective
- `param::AbstractAdaPDMParam`: Algorithm parameters containing logging interval
- `final::Bool=false`: Whether this is the final log entry

# Returns
- `Bool`: `true` if logging was performed, `false` if skipped due to logging interval

# Behavior
- **Header Display**: Shows column headers every 20 log intervals and at iteration 0
- **Selective Logging**: Only logs at specified intervals unless `final=true`
- **Formatted Output**: Displays iteration number, Lagrangian objective, primal/dual residuals, and timing

# Output Format
```
ITERATION    LAGRANGIAN     PRES(l2)     PRES(lInf)    DRES(l2)    DRES(lInf)     TIME
        0    1.2345e+02    1.23e-04     1.23e-04     1.23e-04     1.23e-04     1.23
```

# Examples
```julia
# Regular iteration logging
logged = AdaPDMLog(100, info, param)

# Force final logging regardless of interval
AdaPDMLog(150, info, param; final=true)
```

See also: `AdaPDMIterationInfo`, `AbstractAdaPDMParam`
"""
function AdaPDMLog(iter, info::AdaPDMIterationInfo, param::AbstractAdaPDMParam; final::Bool = false) 
    if param.logLevel < 1
        return false  
    end 

    if (final == false && iter > 0 && iter % param.logInterval != 0)
        return false 
    end 

    header = false 
    if (iter == 0)
        header = true 
    elseif (iter > param.logInterval && (iter/param.logInterval) % 20 == 1)
        header = true 
    end 

    if (header)
        Printf.@printf("%10s ", "ITERATION") 
        Printf.@printf("%12s ", "LAGRANGIAN") 
        Printf.@printf("%12s ", "OBJ") 
        Printf.@printf("%12s ", "PRES(l2)")
        Printf.@printf("%12s ", "PRES(lInf)")  
        Printf.@printf("%12s ", "DRES(l2)")
        Printf.@printf("%12s ", "DRES(lInf)")
        Printf.@printf("%9s\n", "TIME")
    end 
    
    obj = info.lagrangianObj[end]
    presL2 = info.presL2[end]
    presLInf = info.presLInf[end]
    dresL2 = info.dresL2[end]
    dresLInf = info.dresLInf[end]
    time = info.totalTime
    primalObj = info.objectiveValue[end]

    Printf.@printf("%10d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %9.2f\n", 
            iter, 
            obj, 
            primalObj,
            presL2, presLInf, 
            dresL2, dresLInf,
            time)

    return true 
end 

"""
    AdaPDMLog(info::AdaPDMIterationInfo, trueObj::Float64=Inf)

Display a comprehensive summary of the Adaptive Primal-Dual Method algorithm results.

This function prints a formatted summary of the algorithm's final state, including
convergence status, final objective and residual values, iteration count, and timing.
Optionally compares against a known true objective value.

# Arguments
- `info::AdaPDMIterationInfo`: Final iteration information from the algorithm
- `trueObj::Float64=Inf`: True objective value for comparison (if known)

# Output Information
- **Solver Status**: Termination reason (optimal, iteration limit, time limit, etc.)
- **Lagrangian**: Final Lagrangian objective value
- **Primal Residuals**: Final L2 and L∞ primal residuals
- **Dual Residuals**: Final L2 and L∞ dual residuals  
- **Iteration Count**: Total iterations performed
- **Total Time**: Algorithm execution time in seconds
- **Objective Difference**: Difference from true objective (if provided)

# Examples
```julia
# Basic summary without true objective
AdaPDMLog(info)

# Summary with true objective comparison
AdaPDMLog(info, 42.0)
```

# Sample Output
```
AdaPDM Summary: 
    Solver Status   =   ADA_PDM_TERMINATION_OPTIMAL
    Lagrangian      =   4.2345e+01
    Pres (L2)       =   9.8765e-07
    Pres (LInf)     =   1.2345e-06
    Dres (L2)       =   8.7654e-07
    Dres (LInf)     =   1.1111e-06
    Stop. Iter      =          150
    Total Time      =        12.34
    True Obj. Diff  =         0.23
```

See also: `AdaPDMIterationInfo`, `getTerminationStatus`
"""
function AdaPDMLog(info::AdaPDMIterationInfo, logLevel::Int64, trueObj::Float64=Inf)
    if logLevel < 1
        return  
    end 

    @PDMOInfo logLevel "AdaPDM Summary: "
    Printf.@printf("    Solver Status   =   %s\n", getTerminationStatus(info.terminationStatus))
    Printf.@printf("    Lagrangian      = %12.4e\n", info.lagrangianObj[end])
    Printf.@printf("           Obj      = %12.4e\n", info.objectiveValue[end])
    Printf.@printf("    Pres (L2)       = %12.4e\n", info.presL2[end])
    Printf.@printf("    Pres (LInf)     = %12.4e\n", info.presLInf[end])
    Printf.@printf("    Dres (L2)       = %12.4e\n", info.dresL2[end])
    Printf.@printf("    Dres (LInf)     = %12.4e\n", info.dresLInf[end])
    Printf.@printf("    Stop. Iter      = %12d\n",  info.stopIter) 
    Printf.@printf("    Total Time      = %12.2f\n", info.totalTime)
    if (trueObj < Inf)
        diff = abs(trueObj - info.lagrangianObj[end])
        Printf.@printf("    True Obj. Diff  = %12.2f\n", diff)
    else 
        Printf.@printf("    True Obj. Diff  = Unknown\n")
    end 
end 

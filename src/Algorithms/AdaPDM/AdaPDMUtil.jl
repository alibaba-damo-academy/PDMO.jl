"""
    AdaPDMLog(iter, info::AdaPDMIterationInfo, param::AbstractAdaPDMParam; final::Bool = false)

Print one iteration log line for AdaPDM methods.

# Arguments
- `iter::Int`: Current iteration number (0 for initialization)
- `info::AdaPDMIterationInfo`: Current iteration information containing residuals and objective
- `param::AbstractAdaPDMParam`: Algorithm parameters containing logging interval
- `final::Bool=false`: Whether this is the final log entry

# Returns
- `Bool`: `true` if a log line is printed, else `false`.

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

Print final AdaPDM summary statistics.

# Arguments
- `info::AdaPDMIterationInfo`: Final iteration information from the algorithm
- `trueObj::Float64=Inf`: True objective value for comparison (if known)

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

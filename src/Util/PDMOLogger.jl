# PDMOLogger.jl - Simple logging wrapper for PDMO

using Logging

"""
    @PDMOInfo level msg [key=value ...]

Log info message if level >= 1. Wrapper around @info.
"""
macro PDMOInfo(level, msg, args...)
    return esc(:(
        if $level >= 1
            @info $msg $(args...)
        end
    ))
end

"""
    @PDMOWarn level msg [key=value ...]

Log warning message if level >= 2. Wrapper around @warn.
"""
macro PDMOWarn(level, msg, args...)
    return esc(:(
        if $level >= 2
            @warn $msg $(args...)
        end
    ))
end

"""
    @PDMOError level msg [key=value ...]

Log error message if level >= 3. Wrapper around @error.
"""
macro PDMOError(level, msg, args...)
    return esc(:(
        if $level >= 3
            @error $msg $(args...)
        end
    ))
end

"""
    @PDMODebug level msg [key=value ...]

Log debug message if level >= 3. Wrapper around @debug.
"""
macro PDMODebug(level, msg, args...)
    return esc(:(
        if $level >= 3
            @debug $msg $(args...)
        end
    ))
end

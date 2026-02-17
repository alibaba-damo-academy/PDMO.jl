using LinearAlgebra 
using SparseArrays
using Random 
Random.seed!(123)

using Base.Threads 
using Test 

using PDMO

# import these functions so that they can be overloaded 
import PDMO: isProximal, isConvex, isSet, isSmooth, isSupportedByJuMP, JuMPAddSmoothFunction, JuMPAddProximableFunction
import PDMO: proximalOracle, proximalOracle!
import PDMO: gradientOracle, gradientOracle!
import PDMO: adjoint, adjoint!, operatorNorm2

import HSL_jll 
# Check if HSL is properly linked by looking for override/lib/ followed by any system path and libhsl.so or libhsl.dylib
const HSL_FOUND = occursin(r"override/lib/[^/]+/libhsl\.(so|dylib)", HSL_jll.libhsl_path)
# println("HSL_FOUND: $HSL_FOUND; hsl path: $(HSL_jll.libhsl_path)")
if HSL_FOUND == false 
    @warn "Advanced Applications: HSL is not properly linked. IPOPT will use default linear solver."
end 
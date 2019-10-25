module KDESieve

import Distributions
import KernelDensity
import LinearAlgebra

include("comparator.jl")
include("sieve.jl")

export Sieve, predict

end # module

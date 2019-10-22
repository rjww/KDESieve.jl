module KDESieve

import Distributions
import KernelDensity

include("kde_comparator.jl")
include("sieve_layer.jl")
include("sieve.jl")

export Sieve, predict

end # module

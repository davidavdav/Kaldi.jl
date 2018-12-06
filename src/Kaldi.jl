## Kaldi.jl Julia support for the Kaldi speech recognition toolkit
## (c) 2016 David A. van Leeuwen

module Kaldi

using DataStructures

export load_ark_matrix, save_ark_matrix, load_nnet_am, load_ark_matrices, components, __version__

__version__ = "t2"

include("types.jl")
include("io.jl")
include("nnet-propagate.jl")
include("misc.jl")

end

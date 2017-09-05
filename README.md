# Kaldi.jl
Julia interface to the Kaldi speech recognition suite.

Currently, we can only read and write Kaldi `.ark` matrices, and read a nnet2 binary acoustic model.  Sorry---I have to start somewhere.

## Install

```julia
Pkg.clone("https://github.com/davidavdav/Kaldi.jl.git")
```

## Reading a Kaldi `.ark` file

```julia
using Kaldi
for (id, matrix) in load_ark_matrix(fd)
  println("Key ", id, " matrix ", matrix)
end
```
Here `fd` is an object of type `::IO`, e.g., `open("file.ark")`.  `load_ark_matrix()` is a generator (coroutine), and returns a `(key, value)` pair on every iteration.  There is also `load_ark_matrices()` which reads the entire file and produces an `OrderedDict`, with the matrix IDs as keys in the order as they occur in the `.ark` file:
```julia
matrices = load_ark_matrices("file.ark")
```
Matrices have the same direction sense as in the C++ library, i.e., features are like row vectors.  However, this is a different memory layout, because Julia unfortunately represents matrices column-major.  We may change this in the future, if we would directly interface to the kaldi C++ libraries.

Currently only matrices of type float, double and compressed (version 1) are supported.   Compressed matrices are expanded to `Float32`.

## Writing a Kaldi `.ark` file

```julia
save_ark_matrix(f, id::AbstractString, mat::Matrix)
## or
save_ark_matrix(f, d::Associative)
## or
save_ark_matrix(f, keys::Vector{AbstractString}, values::Vector{Matrix{AbstractFloat}})
```

This is the reverse of loading a `.ark` matrix.  `d` can be a normal (unordered) dict, but this leads to an arbitrary storage order of the matrices in the `.ark` file.  Kaldi often works with (promises) of lexicographically ordered keys.  The second version allows explicit control of the order of the matrices without having to use an `OrderedDict`.

Only matirces of type `Float32` and `Float64` are supported.

## Reading a nnet2 neural net acoustical model

We now have rudimentary support for reading (binary) `nnet-am` files in Dan Povey's nnet2 implementation.
```julia
nnetam = open("final.mdl") do fd
    load_nnet_am(fd)
end
```
This reads a tuple `(transition_model, nnet2)` into `nnetam`.

## Plans

Nothing concrete, but it would be kind-of cool to be able to run the `online/nnet2` pipeline of Kaldi natively in Julia.  Come to think of it, it would be better work on nnet3 support instead.  But then, there is not even a small chance we'll be able to reproduce the nnet3 computation.

# Kaldi.jl
Julia interface to the Kaldi speech recognition suite. 

Currently, we can only read and write Kaldi `.ark` matrices.  Sorry---I have to start somewhere. 

## Install

```julia
Pkg.clone("https://github.com/davidavdav/Kaldi.jl.git")
```

## Reading a Kaldi `.ark` file

```julia
using Kaldi
dict = load_ark_matrix(f)
for (id, matrix) in dict
  println("Key ", id, " matrix ", matrix)
end
```
Here `f` is either a file name `::AbstractString` or an object of type `::IO`.  `load_ark_matrix()` returns an `OrderedDict`, with the matrix IDs as keys in the order as they occur in the `.ark` file.  Matrices have the same sense as in the C++ library, i.e., features are like row vectors.  However, this is a different memory layout, because Julia unfortunately represents matrices column-major.  We may change this in the future, if we would directly interface to the kaldi C++ libraries.  

Currently only matrices of type float, double and compressed (version 1) are supported.   Compressed matrices are expanded to `Float32`. 

## Writing a Kaldi `.ark` file

```julia
save_ark_matrix(f, d::Associative)
## or
save_ark_matrix(f, keys::Vector{AbstractString}, values::Vector{Matrix{AbstractFloat}})
```

This is the reverse of loading a `.ark` matrix.  `d` can be a normal (unordered) dict, but this leads to an arbitrary storage order of the matrices in the `.ark` file.  Kaldi often works with (promises) of lexicographically ordered keys.  The second version allows explicit control of the order of the matrices without having to use an `OrderedDict`. 

Only matirces of type `Float32` and `Float64` are supported.  

## Plans

Nothing concrete, but it would be kind-of cool to be able to run the `online/nnet2` pipeline of Kaldi natively in Julia. 

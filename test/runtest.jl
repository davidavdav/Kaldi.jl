include("../src/Kaldi.jl")

nnam = open("6.mdl") do fd
	Kaldi.load_nnet_am(fd);
end

components = nnam.nnet.components

x = Kaldi.load_ark_matrices("dearmrgabriel.feats.ark")["0"]'
yref = Kaldi.load_ark_matrices("out.ark")["0"]'

y = copy(x)
for c in components
	y = Kaldi.propagate(c, y, true)
end

if false

x = rand(Float32, 140, 113)
y = Array(Any, length(nn.components)+1)
y[1] = x

function prop(nn, y, i)
	y[i+1] = Kaldi.propagate(nn.components[i], y[i])
end

for i=1:length(nn.components)
	prop(nn, y, i)
end

z = Array(Any, length(nn.components)+1)
z[1] = x

function propv(nn, z, i)
	z[i+1] = hcat([Kaldi.propagate(nn.components[i], z[i][:,j])] for j in 1:size(z[i], 2))
end

Kaldi.init!(nn)
for i=1:length(nn.components)
	prop(nn, z, i)
end

end

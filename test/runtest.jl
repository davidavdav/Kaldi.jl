include("../src/Kaldi.jl")

trans, nn = open("final.mdl") do fd
	Kaldi.load_nnet_am(fd);
end

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

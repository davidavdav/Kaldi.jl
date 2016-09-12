include("../src/Kaldi.jl")

trans, nn = Kaldi.load_nnet_am(open("final.mdl"));

x = rand(140, 10)
y = Array(Any, 20)
y[1] = x

function prop(nn, y, i)
	y[i+1] = Kaldi.propagate(nn.components[i], y[i])
end

for i=1:length(nn.components)
	prop(nn, y, i)
end

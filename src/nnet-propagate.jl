import StatsFuns

## catch-all rules
## most components don't need an init!()
init!(c::NnetComponent) = c
## a sequence of frames (Matrix) is processed on-by-one by default
propagate(c::NnetComponent, x::Matrix) = mapslices(s -> propagate(c, s), x, 1)

## splice
output_dim(c::SpliceComponent) = (c.input_dim - c.const_component_dim) * length(c.context) + c.const_component_dim

function init!(c::SpliceComponent)
	fill!(c.buffer, zero(eltype(c.buffer)))
	return c
end

function propagate{T}(c::SpliceComponent, x::Vector{T})
	length(x) == c.input_dim || error("Dimension mismatch")
	var_dim = c.input_dim - c.const_component_dim
	y = Array(T, output_dim(c))
	for j in 1:size(c.buffer, 2)-1
		c.buffer[:,j] = c.buffer[:, j+1]
	end
	c.buffer[:, end] = x[1:var_dim]
	offset = 1 - first(c.context)
	dest = 0
	for t in c.context
		y[dest+(1:var_dim)]	= c.buffer[:, offset + t]
		dest += var_dim
	end
	y[dest+1:end] = x[var_dim+1:end]
	return y
end

## affine
## we need to define two definitions to prevent ambiguities with the catch-all mapslices...
propagate(c::AbstractAffineComponent, x::Vector) = c.bias_params .+ c.linear_params * x
propagate(c::AbstractAffineComponent, x::Matrix) = c.bias_params .+ c.linear_params * x
#function propagate{T}(c::AbstractAffineComponent, x::Matrix{T})
#	res = zeros(T, length(c.bias_params), size(x, 2))
#	BLAS.gemm!('N', 'N', one(T), c.linear_params, zero(T), res)
#	broadcast!(+, res, res, c.bias_params)
#	return res
#end

## (group)pnorm
function propagate{T}(c::PnormComponent, x::Vector{T})
	group = c.input_dim ÷ c.output_dim
	return T[norm(view(x, i:i+group-1), c.P) for i in 1:group:length(x)]
end

## normalize
propagate(c::NormalizeComponent, x::Vector) = normalize(x)

## fixed scale
propagate(c::FixedScaleComponent, x::Vector) = c.scales .* x
propagate(c::FixedScaleComponent, x::Matrix) = c.scales .* x

## softmax
propagate(c::SoftmaxComponent, x::Vector) = StatsFuns.softmax(x)

## processing of the entire net
function init!(nn::Nnet)
	for c in nn.components
		init!(c)
	end
	return nn
end

function propagate(nnet::Nnet, x::VecOrMat)
	timing = Float64[]
	for c in nnet.components
		t = time()
		x = propagate(c, x)
		push!(timing, time()-t)
	end
	return x, timing
end

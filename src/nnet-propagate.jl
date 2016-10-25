import StatsFuns

## catch-all rules
## most components don't need an init!()
init!(c::NnetComponent) = c
## a sequence of frames (Matrix) is processed on-by-one by default
propagate(c::NnetComponent, x::Matrix) = mapslices(s -> propagate(c, s), x, 1)

## splice
output_dim(c::SpliceComponent) = (c.input_dim - c.const_component_dim) * length(c.context) + c.const_component_dim
delay(c::SpliceComponent) = maximum(c.context)

## get a frame from a delay line with index i (∈ d.context), given current vector x
function getframe(d::Delay, index::Integer, inframe::Vector)
	mini, maxi = extrema(d.context)
	d.i < maxi && return similar(inframe, length(inframe), 0) ## dim x 0, not type stable!
	if index >= maxi
		return inframe
	else
		return d.buffer[:, clamp(index - mini + 1, 1, d.i)]
	end
end

function pushframe(d::Delay, inframe::Vector)
	dim, nbuf = size(d.buffer)
	length(inframe) == dim || throw(DimensionMismatch("Input frame dimension $(length(inframe)) does not match delay buffer $dim"))
	if d.i < nbuf
		d.i += 1
	else
		d.buffer[:, 1:end-1] = d.buffer[:, 2:end]
	end
	nbuf > 0 && (d.buffer[:, d.i] = inframe)
end

function init!(c::SpliceComponent)
	fill!(c.buffer, zero(eltype(c.buffer)))
	c.cursor = 0
	return c
end

## Features are column vectors stacked in a matrix (for now)

## This might very well be the hardest function to program
function propagate{T}(c::SpliceComponent, x::Vector{T})
	din = length(x)
	din == c.input_dim || error("Dimension mismatch")
	dvar = c.input_dim - c.const_component_dim
	yvar = vcat([getframe(c.delay, i, x[1:dvar]) for i in c.delay.context]...)
	yconst = getframe(c.const_delay, c.delay.context[1], x[dvar+1:end])
	pushframe(c.delay, x[1:dvar])
	pushframe(c.const_delay, x[dvar+1:end])
	return vcat(yvar, yconst)
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

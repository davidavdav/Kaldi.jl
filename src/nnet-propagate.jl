 import StatsFuns

## catch-all rules
## most components don't need an init!() or a flush
init!(c::NnetComponent) = c

## splice
output_dim(c::SpliceComponent) = (c.input_dim - c.const_component_dim) * length(c.delay.context) + c.const_component_dim
delay(c::SpliceComponent) = maximum(c.delay.context)

## get a frame from a delay line with index i (∈ d.context), given current vector x. Now returns a matrix, because
## there can be 0 frames returned, this needs to be encoded somehow.
function getframe(d::Delay, index::Integer, inframe::AbstractVector)
	mini, maxi = extrema(d.context)
	d.i < maxi && return similar(inframe, length(inframe), 0)
	if index >= maxi
		return reshape(inframe, length(inframe), 1)
	else
		i = clamp(index - mini + 1, 1, d.i)
		return d.buffer[:, i:i] ## effective reshape
	end
end

## push a frame onto the delay line buffer
function pushframe(d::Delay, inframe::AbstractVector)
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
	c.delay.i = 0
	return c
end

## Features are column vectors stacked in a matrix (for now)

## This might very well be the hardest function to program
function propagate(c::SpliceComponent, x::AbstractVector{T}) where T
	din = length(x)
	din == c.input_dim || error("Dimension mismatch")
	dvar = c.input_dim - c.const_component_dim
	yvar = vcat([getframe(c.delay, i, x)[1:dvar, :] for i in c.delay.context]...)
	yconst = getframe(c.delay, c.delay.context[1], x)[dvar+1:end, :]
	pushframe(c.delay, x)
	return vcat(yvar, yconst)
end

## We should define flush() for any component that has a delay
function flush(c::SpliceComponent)
	dvar = c.input_dim - c.const_component_dim
	# delay(c) > 0 || return
	lastframe = c.delay.buffer[:,end]
	y = similar(c.delay.buffer, output_dim(c), delay(c))
	for j in 1:delay(c)
		yvar = vcat([getframe(c.delay, j-1 + i, lastframe)[1:dvar, :] for i in c.delay.context]...)
		yconst = getframe(c.delay, j-1 + c.delay.context[1], lastframe)[dvar+1:end, :]
		y[:,j] = vcat(yvar, yconst)
	end
	init!(c)
	return y
end

function propagate(c::NnetComponent, x::AbstractMatrix, flush=false)
	y = [propagate(c, x[:, i]) for i in 1:size(x, 2)]
	if flush && method_exists(Kaldi.flush, (typeof(c),))
		return hcat(y..., Kaldi.flush(c))
	else
		return hcat(y...)
	end
end

## affine
## we need to define two definitions to prevent ambiguities with the catch-all mapslices...
propagate(c::AbstractAffineComponent, x::Vector) = c.bias_params .+ c.linear_params * x
propagate(c::AbstractAffineComponent, x::Matrix) = c.bias_params .+ c.linear_params * x

## (group)pnorm
function propagate(c::PnormComponent, x::Vector{T}) where T
	group = c.input_dim ÷ c.output_dim
	return T[norm(view(x, i:i+group-1), c.P) for i in 1:group:length(x)]
end

## normalize
propagate(c::NormalizeComponent, x::Vector) = normalize(x) * √length(x)

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

function propagate(nnet::Nnet, x::AbstractMatrix)
	timing = Float64[]
	for c in nnet.components
		t = time()
		x = propagate(c, x)
		push!(timing, time()-t)
	end
	return x, timing
end

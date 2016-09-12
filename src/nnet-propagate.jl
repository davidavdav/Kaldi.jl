import StatsFuns

output_dim(c::SpliceComponent) = (c.input_dim - c.const_component_dim) * length(c.context) + c.const_component_dim

## splice
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
# propagate(c::SpliceComponent, x::Matrix) = mapslices(s -> propagate(c, s), x, 1)

## catch-all mapslices
propagate(c::NnetComponent, x::Matrix) = mapslices(s -> propagate(c, s), x, 1)

## affine
## we need to define two definitions to prevent ambiguities with the catch-all mapslices...
propagate(c::AbstractAffineComponent, x::Vector) = c.bias_params .+ c.linear_params * x
propagate(c::AbstractAffineComponent, x::Matrix) = c.bias_params .+ c.linear_params * x

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

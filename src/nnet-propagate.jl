output_dim(c::SpliceComponent) = (c.input_dim - c.const_component_dim) * length(c.context) + c.const_component_dim

function propagate{T}(c::SpliceComponent, x::Vector{T})
	length(x) == c.input_dim || error("Dimension mismatch")
	var_dim = c.input_dim - c.const_component_dim
	y = Array(T, output_dim(c))
	for j in 1:size(c.buffer, 2)-1
		c.buffer[:,j] = c.buffer[:, j+1]
	end
	c.buffer[:, end] = x[1:var_dim]
	offset = last(c.context) + 1
	dest = 0
	for t in c.context
		y[dest+(1:var_dim)]	= c.buffer[:, offset + t]
		dest += var_dim
	end
	y[dest+1:end] = x[var_dim+1:end]
	return y
end

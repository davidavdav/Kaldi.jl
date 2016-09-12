type Transition{T<:AbstractFloat}
	index::Int32
	prob::T
end
Base.eltype{T}(t::Transition{T}) = T

type HmmState{TT<:Transition}
	pdf_class::Int32
	transitions::Vector{TT}
end
# HmmState{TT<:Transition}(p::Int32, t::Vector{TT}) = HmmState{eltype(TT), TT}(p, t)

type TopologyEntry
	entry::Vector{HmmState}
end

type Triple
	phone::Int32
	hmm_state::Int32
	df::Int32
end

type TransitionModel{T<:AbstractFloat}
	topo::Vector{TopologyEntry}
	triples::Vector{Triple}
	log_probs::Vector{T}
end

abstract NnetComponent{T}

type Nnet{T}
	components::Array{NnetComponent}
	priors::Vector{T}
end

type NnetAM{T<:AbstractFloat}
	trans_model::TransitionModel{T}
	nnet::Nnet
end

type SpliceComponent <: NnetComponent
	input_dim::Int32
	context::Vector{Int32}
	const_component_dim::Int32
	buffer::Matrix
	function SpliceComponent(input_dim, context, const_component_dim, ftype=Float32)
		var_dim = input_dim - const_component_dim
		history_dim = 1 - -(extrema(context)...)
		new(input_dim, context, const_component_dim, zeros(ftype, var_dim, history_dim)) ## TODO weasle in the type of the buffer, somehow
	end
end

abstract AbstractAffineComponent <: NnetComponent

type FixedAffineComponent{T} <: AbstractAffineComponent
	linear_params::Matrix{T}
	bias_params::Vector{T}
end

type AffineComponentPreconditionedOnline{T} <: AbstractAffineComponent
	learning_rate::T
	linear_params::Matrix{T}
	bias_params::Vector{T}
	rank_in::Int32
	rank_out::Int32
	update_period::Int32
	num_samples_history::T
	alpha::T
	max_change_per_sample::T
end

type PnormComponent{T} <: NnetComponent
	input_dim::Int32
	output_dim::Int32
	P::T
end

type NormalizeComponent{T} <: NnetComponent
	dim::Int32
	value_sum::Vector{T}
	deriv_sum::Vector{T}
	count::Int64
end

type FixedScaleComponent{T} <: NnetComponent
	scales::Vector{T}
end

type SoftmaxComponent{T} <: NnetComponent
	dim::Int32
	value_sum::Vector{T}
	deriv_sum::Vector{T}
	count::Int64
end

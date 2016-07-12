type HmmState
	pdf_class::Int32
	transitions::Vector{Tuple{Int32, Float32}}
end

type TopologyEntry
	entry::Vector{HmmState}
end

type Triple
	phone::Int32
	hmm_state::Int32
	df::Int32
end

type TransitionModel
	topo::Vector{TopologyEntry}
	triples::Vector{Triple}
	log_probs::Vector{Float32}
end

abstract NnetComponent

type Nnet{T}
	components::Array{NnetComponent}
	priors::Vector{T}
end

type SpliceComponent <: NnetComponent
	input_dim::Int32
	context::Vector{Int32}
	const_component_dim::Int32
end

type FixedAffineComponent{T} <: NnetComponent
	linear_params::Matrix{T}
	bias_params::Vector{T}
end

type AffineComponentPreconditionedOnline{T} <: NnetComponent
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

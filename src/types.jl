struct Transition{T<:AbstractFloat}
	index::Int32
	prob::T
end
Base.eltype{T}(t::Transition{T}) = T

struct HmmState{TT<:Transition}
	pdf_class::Int32
	transitions::Vector{TT}
end
# HmmState{TT<:Transition}(p::Int32, t::Vector{TT}) = HmmState{eltype(TT), TT}(p, t)

struct TopologyEntry
	entry::Vector{HmmState}
end

## old kaldi transition info
struct Triple
	phone::Int32
	hmm_state::Int32
	forward_pdf::Int32
end

## new kaldi transition info
struct Tuple4
	phone::Int32
	hmm_state::Int32
	forward_pdf::Int32
	self_pdf::Int32
end
Tuple4(phone::Int32, hmm_state::Int32, pdf_class::Int32) = Tuple4(phone, hmm_state, pdf_class, pdf_class)


struct TransitionModel{T<:AbstractFloat}
	topo::Vector{TopologyEntry}
	tuples::Vector{Tuple4}
	log_probs::Vector{T}
end

abstract type NnetComponent{T}
end

struct Nnet{T}
	components::Array{NnetComponent}
	priors::Vector{T}
end

struct NnetAM{T<:AbstractFloat}
	trans_model::TransitionModel{T}
	nnet::Nnet
end

mutable struct Delay{T}
	context::Vector{Int32}
	buffer::AbstractMatrix{T}
	i::Int
end
function Delay{T}(context, dim::Integer) where T
	nbuf = maximum(context) - min(minimum(context), 0)
	Delay(context, zeros(T, dim, nbuf), 0)
end
Delay(context, dim::Integer, ftype=Float32) = Delay{ftype}(context, dim)

struct SpliceComponent{T} <: NnetComponent{T}
	input_dim::Int32
	const_component_dim::Int32
	delay::Delay{T}
	# const_delay::Delay
end
SpliceComponent{T}(input_dim, const_component_dim, context::Vector) where T = SpliceComponent(input_dim, const_component_dim, Delay{T}(context, input_dim))
SpliceComponent(input_dim, const_component_dim, context, T::Type) = SpliceComponent{T}(input_dim, const_component_dim, context)

abstract type AbstractAffineComponent{T} <: NnetComponent{T} end

struct FixedAffineComponent{T} <: AbstractAffineComponent{T}
	linear_params::Matrix{T}
	bias_params::Vector{T}
end

struct AffineComponentPreconditionedOnline{T} <: AbstractAffineComponent{T}
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

struct PnormComponent{T} <: NnetComponent{T}
	input_dim::Int32
	output_dim::Int32
	P::T
end

struct NormalizeComponent{T} <: NnetComponent{T}
	dim::Int32
	value_sum::Vector{T}
	deriv_sum::Vector{T}
	count::Int64
end

struct FixedScaleComponent{T} <: NnetComponent{T}
	scales::Vector{T}
end

type SoftmaxComponent{T} <: NnetComponent{T}
	dim::Int32
	value_sum::Vector{T}
	deriv_sum::Vector{T}
	count::Int64
end

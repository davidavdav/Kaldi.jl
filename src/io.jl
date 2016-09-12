## io.jl input/output functions for Kaldi

## inspired by Kaldi source
uint16tofloat(x::UInt16, minvalue, range) = minvalue + range * x / typemax(UInt16)
function uint8tofloat(x::UInt8, quantiles)
    p0, p25, p75, p100 = quantiles
    if x ≤ 0x40
        return p0 + (p25 - p0) * x / 0x40
    elseif x ≤ 0xc0
        return p25 + (p75 - p25) * (x - 0x40) / 0x80
    else
        return p75 + (p100 - p75) * (x - 0xc0) / 0x3f
    end
end

readtoken(io::IO) = ascii(readuntil(io, ' ')[1:end-1])
expecttoken(io::IO, token) = (t = readtoken(io)) == token || error("Expected ", token, ", saw ", t)

#function expectoneortwotokens(io::IO, token1, token2)
#    token = readtoken(io)
#    if token == token1
#        return expecttoken(io, token2)
#    else
#        return token == token2 || error("Expected ", token1, " or ", token2, ", saw ", token)
#    end
#end

is_binary(io::IO) = read(io, UInt8) == 0 && read(io, Char) == 'B'

## This loads Kaldi matrices from an ark stream, one at the time
## we could probably also use code below
load_ark_matrix(io::IO) = @task while !eof(io)
    ## parse the index
    id = readtoken(io)
    binary = is_binary(io)
    binary || error("Only binary format is supported yet")
    token = readtoken(io)
    if token == "CM" ## compressed matrix
        minvalue, range = read(io, Float32, 2)
        nrow, ncol = read(io, Int32, 2)
        ret = Array(Float32, nrow, ncol)
        quantiles = reshape([uint16tofloat(x, minvalue, range) for x in read(io, UInt16, 4*ncol)], (4, Int64(ncol)))
        for j in 1:ncol
            bytes = read(io, UInt8, nrow)
            for i in 1:nrow
                ret[i, j] = uint8tofloat(bytes[i], sub(quantiles, :, j))
            end
        end
        produce(id, ret)
    else
        if token == "FM"
            datatype = Float32
        elseif token == "DM"
            datatype = Float64
        else
            error("Unknown token ", token)
        end
        nrow = Int64(readint(io))
        ncol = Int64(readint(io))
        produce(id, reshape(read(io, datatype, nrow*ncol), (ncol, nrow))')
    end
end

function load_ark_matrices(fd::IO)
    data = OrderedDict{ASCIIString, Matrix}()
    for (id, matrix) in load_ark_matrix(fd)
        data[id] = matrix
    end
    return data
end

## This does no longer work.  Must be the macro that kills it
load_ark_matrix(s::AbstractString) = open(s) do fd
    load_ark_matrix(fd)
end

## save a single matrix with a key
function save_ark_matrix{T<:AbstractFloat}(fd::IO, key::String, value::Matrix{T})
    write(fd, key * " \0B")
    nrow, ncol = size(value)
    if T == Float32
        write(fd, "FM ")
    elseif T == Float64
        write(fd, "DM ")
    else
        error("Unknown floating point type ", T)
    end
    write(fd, UInt8(4), Int32(nrow), UInt8(4), Int32(ncol))
    write(fd, value')
end

## save multiple matrices as dict
save_ark_matrix{K<:AbstractString,V}(fd::IO, dict::Associative{K,V}) = for (k,v) in dict
    save_ark_matrix(fd, k, v)
end

function save_ark_matrix{K<:AbstractString,T<:AbstractFloat}(fd::IO, keys::Vector{K}, values::Vector{Matrix{T}})
    length(keys) == length(values) || error("Vector length mismatch")
    for (k,v) in zip(keys, values)
        save_ark_matrix(fd, k, v)
    end
end

save_ark_matrix(s::AbstractString, args...) = open(s, "w") do fd
    save_ark_matrix(fd, args...)
end

## nnet2 reading, we might want to rewrite some of the above code using routine below...

function load_nnet_am(io::IO)
    is_binary(io) || error("Expected binary header, sorry")
    tm = load_transition_model(io)
    nnet = load_nnet(io)
    return tm, nnet
end

function load_transition_model(io::IO)
    expecttoken(io, "<TransitionModel>")
    topo = load_hmm_topology(io)
    expecttoken(io, "<Triples>")
    triples = [Triple(readint(io), readint(io), readint(io)) for i in 1:readint(io)]
    expecttoken(io, "</Triples>")
    expecttoken(io, "<LogProbs>")
    log_probs = read_kaldi_array(io)
    expecttoken(io, "</LogProbs>")
    expecttoken(io, "</TransitionModel>")
    return TransitionModel(topo, triples, log_probs)
end

function load_hmm_topology(io::IO)
    expecttoken(io, "<Topology>")
    phones = readvector(io, Int32)
    phone2idx = readvector(io, Int32)
    len = readint(io)
    topo = Array(TopologyEntry, len)
    for i in 1:len
        n = readint(io)
        e = Array(HmmState, n)
        T = Any
        for j in 1:n
            pdf_class = readint(io)
            t = [Transition(readint(io), readfloat(io)) for k in 1:readint(io)]
            ## we have to be carefull about the type, not sure if this is in any
            if j == 1
                T = eltype(t[1])
            end
            e[j] = HmmState(pdf_class, Transition{T}[x for x in t])
        end
        topo[i] = TopologyEntry(e)
    end
    expecttoken(io, "</Topology>")
    return topo
end

## recursive list of subtypes, I may be doing this not so efficiently
function recursivesubtypes(t)
    res = []
    s = subtypes(t)
    for tt in s
        if length(subtypes(tt)) == 0
            push!(res, tt)
        else
            for ttt in recursivesubtypes(tt)
                push!(res, ttt)
            end
        end
    end
    return res
end

function load_nnet(io::IO)
    ## nnet
    expecttoken(io, "<Nnet>")
    expecttoken(io, "<NumComponents>")
    n = readint(io)
    components = NnetComponent[]
    expecttoken(io, "<Components>")
    ## take care of type names, strip off "Kalid." prefix and type parameters
    componentdict = Dict(replace(split(string(t),".")[end], r"{\w+}", "")  => t for t in recursivesubtypes(NnetComponent))
    for i in 1:n
        kind = readtoken(io)[2:end-1] ## remove < >
        kind ∈ keys(componentdict) || error("Unknown Nnet component ", kind)
        push!(components, load_nnet_component(io, componentdict[kind]))
        expecttoken(io, "</$kind>")
    end
    expecttoken(io, "</Components>")
    expecttoken(io, "</Nnet>")
    ## priors
    priors = read_kaldi_array(io)
    return Nnet(components, priors)
end

function load_nnet_component(io::IO, ::Type{SpliceComponent})
    input_dim = readint(io, "<InputDim>")
    token = readtoken(io)
    if token == "<LeftContext>"
        leftcontext = readint(io)
        rightcontext = readint(io, "<RightContext>")
        context = collect(-leftcontext:rightcontext)
    elseif token == "<Context>"
        context = readvector(io, Int32)
    else
        error("Unexpected token ", token)
    end
    const_component_dim = readint(io, "<ConstComponentDim>")
    return SpliceComponent(input_dim, context, const_component_dim)
end

function load_nnet_component(io::IO, ::Type{FixedAffineComponent})
    linear_params = read_kaldi_array(io, "<LinearParams>")
    bias_params = read_kaldi_array(io, "<BiasParams>")
    t = promote_type(eltype(linear_params), eltype(bias_params))
    return FixedAffineComponent{t}(linear_params, bias_params)
end

function load_nnet_component(io::IO, ::Type{AffineComponentPreconditionedOnline})
    learning_rate = readfloat(io, "<LearningRate>")
    linear_params = read_kaldi_array(io, "<LinearParams>")
    bias_params = read_kaldi_array(io, "<BiasParams>")
    token = readtoken(io)
    if token == "<Rank>"
        rank_out = rank_in = readint(io)
    elseif token == "<RankIn>"
        rank_in = readint(io)
        rank_out = readint(io, "<RankOut>")
    else
        error("Unexpected token ", token)
    end
    token = readtoken(io)
    if token == "<UpdatePeriod>"
        update_period = readint(io)
        expecttoken(io, "<NumSamplesHistory>")
    elseif token == "<NumSamplesHistory>"
        update_period = 1
    else
        error("Unexpected token ", token)
    end
    num_samples_history = readfloat(io)
    alpha = readfloat(io, "<Alpha>")
    max_change_per_sample = readfloat(io, "<MaxChangePerSample>")
    return AffineComponentPreconditionedOnline(learning_rate, linear_params, bias_params, rank_in, rank_out, update_period, num_samples_history, alpha, max_change_per_sample)
end

function load_nnet_component(io::IO, ::Type{PnormComponent})
    input_dim = readint(io, "<InputDim>")
    output_dim = readint(io, "<OutputDim>")
    p = readfloat(io, "<P>")
    return PnormComponent(input_dim, output_dim, p)
end

function load_nnet_component(io::IO, ::Type{NormalizeComponent})
    dim = readint(io, "<Dim>")
    value_sum = read_kaldi_array(io, "<ValueSum>")
    deriv_sum = read_kaldi_array(io, "<DerivSum>")
    count = readint(io, "<Count>")
    return NormalizeComponent(dim, value_sum, deriv_sum, count)
end

function load_nnet_component(io::IO, ::Type{FixedScaleComponent})
    scales = read_kaldi_array(io, "<Scales>")
    return FixedScaleComponent(scales)
end

function load_nnet_component(io::IO, ::Type{SoftmaxComponent})
    dim = readint(io, "<Dim>")
    value_sum = read_kaldi_array(io, "<ValueSum>")
    deriv_sum = read_kaldi_array(io, "<DerivSum>")
    count = readint(io, "<Count>")
    return SoftmaxComponent(dim, value_sum, deriv_sum, count)
end

function load_nnet_component{C<:NnetComponent}(io::IO, ::Type{C})
    println(readtoken(io))
end


function readint(io, token="")
    if token != ""
        expecttoken(io, token)
    end
    s = read(io, UInt8)
    if s == 4
        return read(io, Int32)
    elseif s == 8
        return read(io, Int64)
    else
        error("Unknown int size ", s)
    end
end

function readfloat(io, token="")
    if token != ""
        expecttoken(io, token)
    end
    s = read(io, UInt8)
    if s == 4
        return read(io, Float32)
    elseif s == 8
        return read(io, Float64)
    else
        error("Unknown float size ", s)
    end
end

## only used for Int32?
function readvector(io::IO, t::Type)
    s = read(io, UInt8)
    len = read(io, Int32)
    s == sizeof(t) || error("Type size check failed: ", s, " ", sizeof(t))
    return read(io, t, len)
end

## This reads a Kaldi-encoded vector or matrix
function read_kaldi_array(io::IO, token="")
    if token != ""
        expecttoken(io, token)
    end
    token = readtoken(io)
    length(token) == 2 || error("Unexpected token length ", length(token))
    if token[1] == 'F'
        datatype = Float32
    elseif token[1] == 'D'
        datatype = Float64
    else
        error("Unknown element type ", token[1])
    end
    if token[2] == 'V'
        len = readint(io)
        return read(io, datatype, len)
    elseif token[2] == 'M'
        nrow = Int(readint(io))
        ncol = Int(readint(io))
        return reshape(read(io, datatype, nrow*ncol), (ncol, nrow))'
    else
        error("Unknown array type")
    end
end

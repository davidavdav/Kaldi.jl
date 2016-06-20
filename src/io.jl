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

## This loads Kaldi matrices from an ark stream.
function load_ark_matrix(fd::IO)
    data = OrderedDict{ASCIIString, Matrix}()
    while !eof(fd)
        ## parse the index
        id = readuntil(fd, ' ')[1:end-1]
        token = ascii(readuntil(fd, ' ')[2:end-1]) ## skip '\0'
        if token == "BCM"
            minvalue, range = read(fd, Float32, 2)
            nrow, ncol = read(fd, Int32, 2)
            ret = Array(Float32, nrow, ncol)
            quantiles = reshape([uint16tofloat(x, minvalue, range) for x in read(fd, UInt16, 4*ncol)], (4, Int64(ncol)))
            for j in 1:ncol
                bytes = read(fd, UInt8, nrow)
                for i in 1:nrow
                    ret[i, j] = uint8tofloat(bytes[i], sub(quantiles, :, j))
                end
            end
            data[id] = ret
        else
            if token == "BFM"
                datatype = Float32
            elseif token == "BDM"
                datatype = Float64
            else
                error("Unknown token ", token)
            end
            readbytes(fd, 1) == [UInt(4)] || error("Expected \\'", nbytes, "' for ", datatype)
            nrow = Int64(read(fd, Int32))
            readbytes(fd, 1) == [UInt(4)] || error("Expected \\'", nbytes, "' for ", datatype)
            ncol = Int64(read(fd, Int32))
            data[id] = reshape(read(fd, datatype, nrow*ncol), (ncol, nrow))'
        end
    end
    return data
end

load_ark_matrix(s::AbstractString) = open(s) do fd
    load_ark_matrix(fd)
end

## save a single matrix with a key
function save_ark_matrix{T<:AbstractFloat}(fd::IO, key::ASCIIString, value::Matrix{T})
    write(fd, key * " \0")
    nrow, ncol = size(value)
    if T == Float32
        write(fd, "BFM ")
    elseif T == Float64
        write(fd, "BDM ")
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

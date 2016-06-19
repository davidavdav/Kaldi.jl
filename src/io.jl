## io.jl input/output functions for Kaldi

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

function load_ark_matrix(fd)
    data = Dict{ASCIIString, Matrix}()
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
                nbytes, datatype = 4, Float32
            elseif token == "BDM"
                nbytes, datatype = 8, Float64
            else
                error("Unknown token ", token)
            end
            readbytes(fd, 1) == [UInt(nbytes)] || error("Expected \\'", nbytes, "' for ", datatype)
            nrow = Int64(read(fd, Int32))
            readbytes(fd, 1) == [UInt(nbytes)] || error("Expected \\'", nbytes, "' for ", datatype)
            ncol = Int64(read(fd, Int32))
            data[id] = reshape(read(fd, Float32, nrow*ncol), (ncol, nrow))'
        end
    end
    return data
end

function save_ark_matrix
end

using LinearAlgebra

# Get the closest normalized floating point

function getDigits(decimalNum, base :: Int64, digits :: Int64)
    """
    A function that takes a decimal representation of a number and
    returns its representation in a specific base up to a certain
    number of digits.
    """
    base = float(base)
    e = floor(Int64,log(base,decimalNum))
    d = zeros(Int64,digits)
    num = decimalNum/(base^e)
    for j = 1:digits
        d[j] = floor(Int64,num)
        num = (num - d[j])*base
    end

    return d, e
end

# List all normalized floating point numbers that can be represented.
function recurr(digits :: Vector{Int64}, base :: Int64, curr :: Int64, precision :: Int64, exponent :: Int64)
    if curr == precision
        for i = 0:(base-1)
            digits[curr] = i
            println("Digits: $digits\tExponent:$exponent")
        end
    else
        for i = 0:(base-1)
            digits[curr] = i
            recurr(digits, base, curr+1, precision, exponent)
        end
    end
end

function listAllNormalizedFp(base :: Int64, precision :: Int64, eMin :: Int64, eMax :: Int64)
    digits = zeros(Int64, precision)
    for e = eMin:eMax
        for lead = 1: (base -1)
            digits[1] = lead
            recurr(digits, base, 2, precision, e)
        end
    end
end

listAllNormalizedFp(3,2,-1,1)

base = 3
p = 5

# List all normalized floating point numbers that can be represented.

function listAllNormalizedFlP(base, precision, emin, emax)
    outcome = zeros(Int64, precision + 1)'
    digits = zeros(Int64, precision)
    allComb = collect(Iterators.product(Iterators.repeated(collect(0:base-1), precision-1)...))
    for e = emin:emax
        for i = 1:length(allComb)
            digits = vcat(1,collect(allComb[i]))
            println("Digits: $digits\tExponent:$e")
            outcome = vcat(outcome, hcat(digits',e) )
        end
    end
    return outcome[2:end,:]
end


Out = listAllNormalizedFlP(2,2,0,1)


# Back substitution algorithm
# solving the inverse of a upper triangular matrix

function backSub(R, b)

    # n should equal to m and should equal the length of b
    n, m = size(R)
    x = zeros(n)
    x[n] = b[n] / R[n,n]
    for k = n-1:-1:1
        x[k] = (b[k] - R[k,k+1:end]' * x[k+1:end]) / R[k,k]
    end
    return x
end

R = [
    4 -1  2  3;
    0 -2  7 -4;
    0  0  6  5;
    0  0  0  3;]
b = [20, -7, 4, 6]

x = backSub(R, b)
println("Answer should be: [3, -4, -1, 2]")
println(x)


all_normalized_fp = function(base::Int64, prec::Int64, emin::Int64, emax::Int64)
    ## Number of possible values for each e:
    N = (base-1)*base^(prec-1)#*(emax-emin+1)

    out=zeros(Int64, N, prec, emax-emin+1)

    es = emin:emax


    for e=1:length(es)
        for b0=1:(base-1)
            for i=1:(base^(prec-1))
                out[(b0-1)*(base^(prec-1))+i,1,e] = b0
                for j=1:(prec-1)
                    out[(b0-1)*(base^(prec-1))+i,prec-j+1,e] = floor((i-1)/base^(j-1))%base
                end
            end
        end
    end

    return(out)
end

test=all_normalized_fp(2, 3, -1, 1)

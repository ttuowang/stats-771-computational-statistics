using LinearAlgebra

# Implementation of Jacobi without using '\'
function jacobiIteration(A, b, x; iterMax = 100)
    """Implements the Jacobi iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess."""
    n = length(x)
    for i = 1:iterMax
        xk = x
        for j = 1:n
            x[j] = 1/A[j,j]*(b[j] - vcat(A[j,1:j-1],A[j,j+1:end])' * vcat(xk[1:j-1],xk[j+1:end]) )
        end
    end
    return x
end

# Implementation of Gauss-Seidel without using '\'
function gsIteration(A, b, x; iterMax = 100)
    """Implements the Gauss-Seidel iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess."""
    n = length(x)
    for i = 1:iterMax
        xk = x
        for j = 1:n
            x[j] = 1/A[j,j]*(b[j] - A[j,1:j-1]'*x[1:j-1] - A[j,j+1:end]'*xk[j+1:end])
        end
    end
    return x
end


function generateProblem(n)
    A = randn(n,n)  + sqrt(n)*Matrix{Float64}(I,n,n)
    x = randn(n)
    x0 = randn(n)
    b = A*x
    return A, b, x, x0
end

A, b, x, x0 = generateProblem(10)

x00 = copy(x0)
z_j = jacobiIteration(A,b,x00; iterMax = 10000)
println(norm(z_j-x))

#x00 = copy(x0)
#z_j_2, G = jacobiIteration2(A,b,x00; iterMax = 10000)
#println(norm(z_j_2-x))


x00 = copy(x0)
z_gs = gsIteration(A,b,x00)

println(norm(z_gs-x))

# Implement SOR without using '\'
function sorIteration(A, b, x, w; iterMax = 100)
    """Implements the SOR iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess.
    w is a number between (0,1]"""
    n = length(x)
    for i = 1:iterMax
        xk = x
        for j = 1:n
            x[j] = w/A[j,j]*(b[j] - A[j,1:j-1]'*x[1:j-1] - A[j,j+1:end]'*xk[j+1:end]) + (1-w)*xk[j]
        end
    end
    return x
end

# Implement SSOR without using '\'
function ssorIteration(A, b, x, w; iterMax = 100)
    """Implements the SSOR iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess.
    w is a number between (0,1]"""
    n = length(x)
    for i = 1:iterMax
        xk = x
        for j = 1:n
            x[j] = w/A[j,j]*(b[j] - A[j,1:j-1]'*x[1:j-1] - A[j,j+1:end]'*xk[j+1:end]) + (1-w)*xk[j]
        end
        xk = x
        for j = 1:n
            x[j] = w/A[j,j]*(b[j] - A[j,1:j-1]'*xk[1:j-1] - A[j,j+1:end]'*x[j+1:end]) + (1-w)*xk[j]
        end
    end
    return x
end


function generateProblem(n)
    A = randn(n,n)  + sqrt(n)*Matrix{Float64}(I,n,n)
    x = randn(n)
    x0 = randn(n)
    b = A*x
    return A, b, x, x0
end

using Random
Random.seed!(1223);
A, b, x, x0 = generateProblem(10)

x00 = copy(x0)
z_j = jacobiIteration(A,b,x00; iterMax = 10000)

x00 = copy(x0)
z_gs = gsIteration(A,b,x00; iterMax = 10000)

w = 0.3
x00 = copy(x0)
z_sor = sorIteration(A,b,x00,w; iterMax = 10000)

x00 = copy(x0)
z_ssor = ssorIteration(A,b,x00,w; iterMax = 10000)

println(norm(z_j-x))
println(norm(z_gs-x))
println(norm(z_sor-x))
println(norm(z_ssor-x))

# Implementation of the Cyclical Kaczmarz and Random Kaczmarz

# Kaczmarz Update

using Pkg
Pkg.add("Distributions")
using Distributions

function kaczmarzUpdate(b,a,x)
    """
    The Kaczmarz update, where b is a scalar,
    a is vector, and x is the current iterate.
    Returns the next iterate.
    """
    return x + ((b - dot(a,x))/dot(a,a))*a
end


# Cyclical Kaczmarz
function cyclicalKaczmarz(b, A, x;maxIter=5)
    """
    Estimates solution to Ax=b using Kaczmarz
    update by cycling through the rows of A.
    b is constant vector.
    A is the coefficient matrix.
    x is an initial iterate.
    maxIter is the number of cycles over A.
    """
    n = length(b)
    for i = 0:(maxIter*n)-1
        k = rem(i,n)+1
        x = kaczmarzUpdate(b[k],A[k,:],x)
    end
    return x
end

# mode function
rem(101, 100)

#Randomized Kaczmarz
using Distributions
function randomizedKaczmarz(b, A, x; maxIter=1000)
    """
    Estimates solution to Ax=b using randomized
    Kaczmarz, where probability of sampling a row
    depends on the sum of squares of the row.
    Samples are taken with replacement.
    b is a constant vector.
    A is the coefficient matrix.
    x is an initial iterate.
    maxIter is number of rows sampled.
    """
    n = length(b)
    p = map(λ -> sum(A[λ,:].^2), 1:n)/sum(A.^2)
    dist = Categorical(p)
    indices = rand(dist,maxIter)
    for k in indices
        x = kaczmarzUpdate(b[k],A[k,:],x)
    end
    return x
end

# Randomized Kaczmarz with row permutation
using Random
rng = MersenneTwister(1234);
A = [1 2 3; 4 5 6; 7 8 9]
shuffle(rng, Vector(1:10))

function permutationKaczmarz(b, A, x;maxIter=5)
    """
    Estimates solution to Ax=b using Kaczmarz
    update by cycling through the rows of A.
    b is constant vector.
    A is the coefficient matrix.
    x is an initial iterate.
    maxIter is the number of cycles over A.
    """
    n = length(b)
    rng = MersenneTwister(1234);
    for i = 1:maxIter
        for j = 1:n
            x = kaczmarzUpdate(b[j],A[j,:],x)
        end
        rowId = shuffle(rng, Vector(1:size(A)[1]))
        A = A[rowId,:]
    end
    return x
end


n = 100
d = n
A = randn(n,d)
x₊ = randn(d)
b = A*x₊

x₀ = randn(d)
x_cyc = cyclicalKaczmarz(b, A, x₀, maxIter=100)
x_ran = randomizedKaczmarz(b, A, x₀, maxIter=100*n)
x_per = permutationKaczmarz(b, A, x₀, maxIter=100)

norm(A*x_cyc-b)
norm(A*x_ran-b)
norm(A*x_per-b)





################################################################
################################################################
################################################################
# Implementation of Jacobi without using '\'
function jacobiIteration2(A, b, x, xreal; iterMax = 100)
    """Implements the Jacobi iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess.
    xreal is the true value, i.e. A * xreal = b"""
    n = length(x)
    r = zeros(iterMax)
    for i = 1:iterMax
        r[i] = norm(x - xreal)
        xk = x
        for j = 1:n
            x[j] = 1/A[j,j]*(b[j] - vcat(A[j,1:j-1],A[j,j+1:end])' * vcat(xk[1:j-1],xk[j+1:end]) )
        end
    end
    return x, r
end

# Implementation of Gauss-Seidel without using '\'
function gsIteration2(A, b, x,xreal; iterMax = 100)
    """Implements the Gauss-Seidel iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess."""
    n = length(x)
    r = zeros(iterMax)
    for i = 1:iterMax
        r[i] = norm(x - xreal)
        xk = x
        for j = 1:n
            x[j] = 1/A[j,j]*(b[j] - A[j,1:j-1]'*x[1:j-1] - A[j,j+1:end]'*xk[j+1:end])
        end
    end
    return x,r
end

# Implement SOR without using '\'
function sorIteration2(A, b, x,xreal,w; iterMax = 100)
    """Implements the SOR iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess.
    w is a number between (0,1]"""
    n = length(x)
    r = zeros(iterMax)
    for i = 1:iterMax
        r[i] = norm(x - xreal)
        xk = x
        for j = 1:n
            x[j] = w/A[j,j]*(b[j] - A[j,1:j-1]'*x[1:j-1] - A[j,j+1:end]'*xk[j+1:end]) + (1-w)*xk[j]
        end
    end
    return x,r
end

# Implement SSOR without using '\'
function ssorIteration2(A, b, x,xreal, w; iterMax = 100)
    """Implements the SSOR iteration without using backslash.
    A is the coefficient matrix.
    b is the constant matrix.
    x is an initial guess.
    w is a number between (0,1]"""
    n = length(x)
    r = zeros(iterMax)
    for i = 1:iterMax
        r[i] = norm(x - xreal)
        xk = x
        for j = 1:n
            x[j] = w/A[j,j]*(b[j] - A[j,1:j-1]'*x[1:j-1] - A[j,j+1:end]'*xk[j+1:end]) + (1-w)*xk[j]
        end
        xk = x
        for j = 1:n
            x[j] = w/A[j,j]*(b[j] - A[j,1:j-1]'*xk[1:j-1] - A[j,j+1:end]'*x[j+1:end]) + (1-w)*xk[j]
        end
    end
    return x,r
end



using Random

Random.seed!(2031);
A, b, xtrue, x0 = generateProblem(10)

x00 = copy(x0)
z_j, r_j = jacobiIteration2(A,b,x00,xtrue)

x00 = copy(x0)
z_gs, r_gs = gsIteration2(A,b,x00,xtrue)

w = 0.5
x00 = copy(x0)
z_sor, r_sor = sorIteration2(A,b,x00,xtrue,w)

x00 = copy(x0)
z_ssor, r_ssor = ssorIteration2(A,b,x00,xtrue,w)

println(norm(z_j-xtrue))
println(norm(z_gs-xtrue))
println(norm(z_sor-xtrue))
println(norm(z_ssor-xtrue))

# Plotting in Julia
# Reference: https://julialang.org/downloads/plotting.html
using Pkg
Pkg.add("Gadfly")
Pkg.add("DataFrames")
using Gadfly
using DataFrames
Pkg.add("Plots")
Pkg.add("PyPlot")
using PyPlot
using Plots

R = hcat(r_j,r_gs,r_sor,r_ssor)

fig, ax = subplots()
ax[:plot](1:100, R[:,1], label="Jacobi")
ax[:plot](1:100, R[:,2],alpha=0.4, label="GS")
ax[:plot](1:100, R[:,3], label="SOR")
ax[:plot](1:100, R[:,4], label="SSOR")
legend()


# Another useful example
A = [0.9 0 0; 0 0.8 0; 0 0 0.7]
xtrue = randn(3)
b = A*xtrue
x0 = randn(3)
maximum(norm.(eigvals!(A)))

x00 = copy(x0)
z_j, r_j = jacobiIteration2(A,b,x00,xtrue)

x00 = copy(x0)
z_gs, r_gs = gsIteration2(A,b,x00,xtrue)

w = 0.5
x00 = copy(x0)
z_sor, r_sor = sorIteration2(A,b,x00,xtrue,w)

x00 = copy(x0)
z_ssor, r_ssor = ssorIteration2(A,b,x00,xtrue,w)

println(norm(z_j-xtrue))
println(norm(z_gs-xtrue))
println(norm(z_sor-xtrue))
println(norm(z_ssor-xtrue))

R = hcat(r_j,r_gs,r_sor,r_ssor)

fig, ax = subplots()
ax[:plot](1:100, R[:,1], label="Jacobi")
ax[:plot](1:100, R[:,2],alpha=0.4, label="GS")
ax[:plot](1:100, R[:,3], label="SOR")
ax[:plot](1:100, R[:,4], label="SSOR")
legend()

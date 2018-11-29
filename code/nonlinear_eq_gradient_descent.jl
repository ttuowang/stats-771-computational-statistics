using LinearAlgebra

# Single update of the gradient descent method

function singleGradDescent(A,x,b, alpha)
    """
    b is constant vector.
    A is the coefficient matrix.
    x is current iterate.
    alpha is step size.
    """
    x = x + alpha * A'*(b-A*x)
    return x
end

"""
# Implement four algorithms one for each of the four strategies of alpha discussed in clss
# Strategy 1, alpha = ||A'r^c||_2^2 / ||AA'r^c||_2^2
"""
function gradientDescent1(A,x,b, xreal;alpha_strategy = 1,ϵ = 1e-15,maxIter=1000)
    k = 1
    r_c = A*x - b
    r = norm(x - xreal)
    while norm(A'* r_c) >= ϵ && k <= maxIter
        alpha = norm(A'*r_c)^2 / norm(A*A'*r_c)^2
        x = singleGradDescent(A,x,b,alpha)
        r_c = A*x - b
        k = k+1
        r = vcat(r, norm(x - xreal))
    end
    return x, r, k
end

function generateProblem(n)
    tmp = randn(n,n)
    A = tmp'*tmp  + sqrt(n)*Matrix{Float64}(I,n,n)
    x = randn(n)
    x0 = randn(n)
    b = A*x
    return A, b, x, x0
end

A, b, x, x0 = generateProblem(10)
x_hat, res, iter = gradientDescent1(A,x0,b,x; maxIter = 1000)
println(norm(x_hat-x))

using PyPlot
plot(1:iter, res, "b-")

"""
# Strategy 2,  alpha = ||A'(x_k - x^*)||_2^2 / ||AA'(x_k - x^*)||_2^2
"""
function gradientDescent2(A,x,b, xreal;alpha_strategy = 1,ϵ = 1e-15,maxIter=1000)
    k = 1
    d_c = x - xreal
    r = norm(d_c)
    while norm(d_c) >= ϵ && k <= maxIter
        alpha = norm(A'*d_c)^2 / norm(A*A'*d_c)^2
        x = singleGradDescent(A,x,b,alpha)
        d_c = x - xreal
        k = k+1
        r = vcat(r, norm(d_c))
    end
    return x, r, k
end

A, b, x, x0 = generateProblem(10)
x_hat, res, iter = gradientDescent2(A,x0,b,x; maxIter = 1000)
println(norm(x_hat-x))

using PyPlot
plot(1:iter, res, "b-")

"""
# Strategy 3,  alpha = 2/(sigma(AA^T)_min^2 + sigma(AA^T)_max^2)
"""
function gradientDescent2(A,x,b, xreal;alpha_strategy = 1,ϵ = 1e-15,maxIter=1000)
    k = 1
    d_c = x - xreal
    r = norm(d_c)
    while norm(d_c) >= ϵ && k <= maxIter
        F = svd(A*A');
        alpha = 2 / (maximum(F.S)^2 + minimum(F.S)^2)
        x = singleGradDescent(A,x,b,alpha)
        d_c = x - xreal
        k = k+1
        r = vcat(r, norm(d_c))
    end
    return x, r, k
end

"""
# Strategy 4,  alpha = 2/(sigma(A^TA)_min^2 + sigma(A^TA)_max^2)
"""
function gradientDescent2(A,x,b, xreal;alpha_strategy = 1,ϵ = 1e-15,maxIter=1000)
    k = 1
    d_c = x - xreal
    r = norm(d_c)
    while norm(d_c) >= ϵ && k <= maxIter
        F = svd(A'*A);
        alpha = 2 / (maximum(F.S)^2 + minimum(F.S)^2)
        x = singleGradDescent(A,x,b,alpha)
        d_c = x - xreal
        k = k+1
        r = vcat(r, norm(d_c))
    end
    return x, r, k
end

################################################################################
################################################################################
################################################################################

"""
# Write a function to generate 2 equations with 2 unknowns such that the
# coefficient matrix is dense and symmetric with user specified non-zero eigenvalues
"""
function generateProblem2(eigenval)
    n = length(eigenval)
    B = rand(n,n)
    Q = qr(B).Q
    Σ = diagm(0 => eigenval)
    A = Q*Σ*Q'
    x = randn(n)
    b = A*x
    return A, x, b
end

# check eigenvalues and sigular values.
A, x, b = generateProblem2([1,2])
eigvals(A)
sqrt.(svdvals(A'A))

# Write a function to draw the level sets of an arbitrary residual function
# f(x) = ||Ax - b||_2^2
function contourPlot(A, b;numLevel = 30)
    x1 = collect(-10.0:0.1:10.0)
    x2 = collect(-10.0:0.1:10.0)

    X1 = repeat(reshape(x1, 1, :), length(x2), 1)
    X2 = repeat(x2, 1, length(x1))

    ATA = A'*A
    D = b'*A
    f(x1,x2) = begin
        ATA[1,1]*x1^2 + ATA[2,2]*x2^2 + 2*ATA[1,2]*x1*x2 -
        2*(D[1]*x1+D[2]*x2)+b[1]^2+b[2]^2
    end
    z = map(f, X1, X2)

    levels = (minimum(z):(maximum(z)-minimum(z))/numLevel:maximum(z))
    p = contour(x1,x2, z, levels)
    return p
end

contourPlot(A, b)

# Strategy 2,  alpha = ||A'(x_k - x^*)||_2^2 / ||AA'(x_k - x^*)||_2^2
function gradientDescent2(A,x,b, xreal;ϵ = 1e-15,maxIter=1000)
    k = 1
    d_c = x - xreal
    r = norm(d_c)
    points = x
    while norm(d_c) >= ϵ && k <= maxIter
        alpha = norm(A'*d_c)^2 / norm(A*A'*d_c)^2
        x = singleGradDescent(A,x,b,alpha)
        d_c = x - xreal
        k = k+1
        r = vcat(r, norm(d_c))
        points = hcat(points, x)
    end
    return x, r, k, points
end

A, x, b = generateProblem2([1,800])
x0 = [-5;-5]
x_hat, res, iter, points = gradientDescent2(A,x0,b,x; maxIter = 100)

x1 = points[1,:]
x2 = points[2,:]

contourPlot(A, b)
plot(x1, x2,"b-o")

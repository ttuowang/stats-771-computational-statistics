using LinearAlgebra

# Under determined Least Squares
function underLS(A,b; ϵ = 1e-14)
    """
    Solves an underdetermined linear system given
    the coefficient matrix A and the constant
    vector b. Returns the least norm solution.
    """
    n, m = size(A)
    s = min(n,m)
    F = qr(A, Val(true))

    #Compute rank approximation r
    Rtrm = F.R[1:s,1:s]
    r = rank(F.R)
    l = m - r

    #Generate R and S
    R, S = F.R[1:r,1:r], F.R[1:r,r+1:end]
    d, P = R\(F.Q'*b)[1:r], R\S
    z2 = consistentLS(P'*P + Matrix{Float64}(I,l,l), P'*d)
    z1 = d - P*z2
    return F.P*vcat(z1,z2)
end

function sigmoid(x)
    return 1/(1+exp(-x))
end

"""
Gauss-Newton algorithm into generlized Gauss-Newton Algorithm
other name, iteratively reweighted least squares.
reference:
Pattern Recognition and Machine Learning, P208
The elements of statistical learning, Chapter 4.4
"""
function logisticGaussNewton(f::Function, X, Y, β₀,α₀;ϵ=10e-8,maxIter = 25)

    k = 1
    funcEvals = 0

    #β₀ = [1.4;-0.402560;-0.474052;-0.010970;-0.171357;-0.480331;-0.278603;-0.005615]
    P = sigmoid.(X*β₀)
    ∇L  = -X'*(Y-P)
    W = Diagonal( P.* (1 .- P) )
    ∇²L = X'*W*X
    f₀ = f(β₀)
    funcEvals += 1

    while norm(∇L) > ϵ && k <= maxIter

        # Search Direction
        p = underLS(∇²L, ∇L)

        #Step Length Backtracking
        #α, evals = backtrack(f,f₀,∇L,p,β₀,α₀)
        #funcEvals += evals

        #Step Length Backtrack with quadratic interpolation
        #α, evals = backtrackQuad(f,f₀,g₀,x₀)
        #funcEvals += evals

        #Fix step Length
        α = α₀

        #Update Parameter
        β₀ = β₀ + α * p
        P = sigmoid.(X*β₀)
        ∇L  = -X'*(Y-P)
        W = Diagonal( P.* (1 .- P) )
        ∇²L = X'*W*X
        f₀ = f(β₀)
        funcEvals += 1
        k +=1
    end
    β₀
    return β₀, k
end

function backtrack(f::Function,f₀,g₀,p,x₀,α₀ = 1.0, ρ = 0.5, C = 1e-4)
    funcEvals = 0
    x₊ = x₀ + α₀*p
    f₊ = f(x₊)
    funcEvals += 1
    while f₊ > f₀ + C*α₀*g₀'*p
        α₀ = ρ*α₀
        x₊ = x₀ + α₀*p
        f₊ = f(x₊)
        funcEvals += 1
    end
    return α₀, funcEvals
end

"""
Apply Genralized Gradient Descent on Logistic regression

# use the data from UCBAdmit_Logit.jl
# After running UCBAdmit_Logit.jl
# You will get a response variable vector Y and variable matrix X
"""

function sigmoid(x)
    return 1/(1+exp(-x))
end

function negl(beta)
    """
    return the negative log likelihood for logistic regression
    beta: a vector
    X & Y: global variable, provided outside of function
    """
    n,m = size(X)
    Ŷ = sigmoid.(X*beta)
    loglikelihood = (Y'*log.(Ŷ) + (ones(n)-Y)'*log.(ones(n)-Ŷ))/n
    return -loglikelihood
end

β₀ = [1.4;-0.402560;-0.474052;-0.010970;-0.171357;-0.480331;-0.278603;-0.005615]
β₀ = zeros(8)
α₀ = 1e-6;
β, K = logisticGaussNewton(negl,X, Y, β₀,α₀; maxIter = 1000)

P = sigmoid.(X*β)
∇L  = -X'*(Y-P)
norm(∇L)

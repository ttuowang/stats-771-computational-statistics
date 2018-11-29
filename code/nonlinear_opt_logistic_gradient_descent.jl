using LinearAlgebra
using Pkg

Pkg.add("RDatasets")

using RDatasets

"""
Name: gradientDescent
Description: implements gradient descent method for non-linear optimization problem

INPUTS:
1. f :: Function, object function
2. g :: Function, evaluates the gradient of f
3. x₀ :: Array{Float64,1}, starting point
4. ϵ :: Float64, tolerance. Defaults to 10e-8

KEYWORD INPUTS:
1. maxIter :: Int64, maximum number of iterations. Defaults to 25

OUTPUTS:
1. x :: Array{Float}, solution to the non-linear optimization problem within tolerance
2. funcEvals :: Int64, number of function evaluations
3. gradientEvals :: Int64, number of gradient evaluations
"""
function gradientDescent(f::Function, g::Function, x₀,ϵ=10e-8;maxIter = 25)

    funcEvals = 0
    gradientEvals = 0
    k = 0

    f₀ = f(x₀)
    funcEvals += 1
    g₀ = g(x₀)
    gradientEvals += 1

    func = f₀
    gnorm = norm(g₀)

    while norm(g₀) > ϵ && k <= maxIter
        #Search Direction
        d = -g₀

        #Step Length

        #α, evals = armijoBacktrack(F,F₀,d,x₀)
        #funcEvals += evals

        # Fix step Length
        α = 0.01

        #Update Parameter
        x₀ += α*d
        f₀ = f(x₀)
        funcEvals += 1
        g₀ = g(x₀)
        gradientEvals += 1
        k +=1

        func = vcat(func, f₀)
        gnorm = vcat(gnorm, norm(g₀))
    end
    x = x₀
    return x, func, gnorm
end

# Test on a toy example.
# f(x,y) = x^2 + xy + 2*y^2
# This is a convex function.
x₀ = [0.5;0.5]
f1(x) = x[1]^2 + x[1]*x[2] + 2*x[2]^2
g1(x) = [2*x[1] + x[2]; x[1] + 4*x[2]]
x, funcEvals, gradientEvals = gradientDescent(f1, g1, x₀,10e-14; maxIter = 1000)

# Test gradient descent on Logistic regression
# use the data from UCBAdmit_Logit.jl
# After running UCBAdmit_Logit.jl
# You will get a response variable vector Y and variable matrix X

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

function gl(beta)
    """
    return the gradient of the negative log likelihood function
    at a given point.
    beta: a vector
    X & Y: global variable, provided outside of function
    """
    Ŷ = sigmoid.(X*beta)
    return -X'*(Y - Ŷ)/n
end

# Verify
beta = [1.485427;-0.402560;-0.474052;-0.010970;-0.171357;-0.480331;-0.278603;-0.005615]
negl(beta)
gl(beta)
(negl(beta) - negl(beta+1e-6*Matrix{Float64}(I,8,8)[1,:] ))/1e-6


β₀ = [1.4;-0.402560;-0.474052;-0.010970;-0.171357;-0.480331;-0.278603;-0.005615]
β, fvec, gvec = gradientDescent(negl, gl, β₀,10e-3; maxIter = 100);

using PyPlot
using Plots
plot(0:101, gvec)

β₀ = [1.4;-0.402560;-0.474052;-0.010970;-0.171357;-0.480331;-0.278603;-0.005615]
f₀ = negl(β₀)
g₀ = gl(β₀)

β₁ = β₀ - 0.01 * gl(β₀)
f₁ = negl(β₁)

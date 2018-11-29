using LinearAlgebra

function gradientDescent(f::Function, g::Function,x₀, α₀ = 1.0;ϵ=10e-8,maxIter = 25)

    funcEvals = 0
    gradientEvals = 0
    k = 0

    f₀ = f(x₀)
    funcEvals += 1
    g₀ = g(x₀)
    gradientEvals += 1

    while norm(g₀) > ϵ && k <= maxIter
        #Search Direction
        p = -g₀

        #Step Length Backtracking
        α, evals = backtrack(f,f₀,g₀,p,x₀,α₀)
        funcEvals += evals

        #Step Length Backtrack with quadratic interpolation
        #α, evals = backtrackQuad(f,f₀,g₀,p,x₀)
        #funcEvals += evals

        #Step Length Backtrack with cubic interpolation
        #α, evals = backtrackCubic(f,f₀,g₀,p,x₀)
        #funcEvals += evals

        # Fix step Length
        #α = 0.1

        #Update Parameter
        x₀ += α*p
        f₀ = f(x₀)
        funcEvals += 1
        g₀ = g(x₀)
        gradientEvals += 1
        k +=1
    end
    x = x₀
    return x, funcEvals, gradientEvals
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

function backtrackQuad(f::Function,f₀,g₀,p,x₀,α₀ = 1.0,C = 1e-4)
    funcEvals = 0
    x₊ = x₀ + α₀*p
    f₊ = f(x₊)
    funcEvals += 1
    while f₊ > f₀ + C*α₀*g₀'*p
        #α₀ = ρ*α₀
        c = f₀
        b = g₀'*(-g₀)
        a = (f(x₀ + α₀*p) - b*α₀ - c)/α₀^2
        funcEvals += 1

        α₀ = -b/(2*a)
        x₊ = x₀ + α₀*p
        f₊ = f(x₊)
        funcEvals += 1
    end
    return α₀, funcEvals
end

function backtrackCubic(f::Function,f₀,g₀,p,x₀,α₀ = 1.0,C = 1e-4)
    #f(x) = f1(x)
    #g(x) = g1(x)
    #x₀ = [0.5;0.5]
    #f₀ = f(x₀)
    #g₀ = g(x₀)
    #p = -g₀
    #C = 1e-4
    α₀ = α₀
    funcEvals = 0
    x₊ = x₀ + α₀*p
    f₊ = f(x₊)
    funcEvals += 1
    while f₊ > f₀+C*α₀*g₀'*p
        #α₀ = ρ*α₀
        """
        First generate a,b,c as quadratic interpolation,
        then calculate α₁.
        """
        c = f₀
        b = g₀'*p
        a = (f(x₀ + α₀*p) - b*α₀ - c)/α₀^2
        funcEvals += 1
        α₁ = -b/(2*a)

        if f(x₀ + α₁*p) <= f₀+C*α₁*g₀'*p
            α₀ = α₁
            break
        end

        """
        Second generate a,b,c,d for cubic interpolation,
        then calculate α₂.
        """
        d = f₀
        c = g₀'*p

        print(α₀)
        print("\n")
        print(α₁)
        print("\n")
        A = [α₀^2 -α₁^2;-α₀^3 α₁^3]
        B = [ f(x₀ + α₁*p) - f₀ - g₀'*p*α₁; f(x₀ + α₀*p) - f₀ - g₀'*p*α₀]

        funcEvals += 2
        AB = ( 1/(α₀^2*α₁^2*(α₁-α₀)) )*(A*B)

        a = AB[1]
        b = AB[2]
        α₂ = (-b + sqrt(b^2 - 3*a*g₀'*p))/(3*a)
        """
        Do Line Search
        """
        α₀ = α₂
        print(α₀)
        print("\n")
        x₊ = x₀ + α₀*p
        f₊ = f(x₊)
        funcEvals += 1
    end
    return α₀, funcEvals
end

"""
Test on a toy example.
f(x,y) = x^2 + xy + 2*y^2
This is a convex function.
"""
x₀ = [0.5;0.5]
f1(x) = x[1]^2 + x[1]*x[2] + 2*x[2]^2
g1(x) = [2*x[1] + x[2]; x[1] + 4*x[2]]
x, funcEvals, gradientEvals = gradientDescent(f1, g1, x₀, 1.0; maxIter = 1000)

################################################################################
################################################################################

"""
Apply Gradient Descent on Logistic regression

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

β₀ = zeros(8)
β, fvec, gvec = gradientDescent(negl, gl, β₀,0.0001; maxIter = 10000)

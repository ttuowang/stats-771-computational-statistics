using LinearAlgebra

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

        #Step Length Backtracking
        α, evals = backtrack(f,f₀,g₀,x₀)
        funcEvals += evals

        #Step Length Backtrack with quadratic interpolation
        #α, evals = backtrackQuad(f,f₀,g₀,x₀)
        #funcEvals += evals

        # Fix step Length
        #α = 0.1

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
    return x, funcEvals, gradientEvals
end

function backtrack(f::Function,f₀,g₀,x₀,α₀ = 1.0, ρ = 0.5, C = 1e-4)
    funcEvals = 0
    x₊ = x₀ + α₀*(-g₀)
    f₊ = f(x₊)
    funcEvals += 1
    while f₊ > f₀ + C*α₀*g₀'*(-g₀)
        α₀ = ρ*α₀
        x₊ = x₀ + α₀*(-g₀)
        f₊ = f(x₊)
        funcEvals += 1
    end
    return α₀, funcEvals
end

function backtrackQuad(f::Function,f₀,g₀,x₀,α₀ = 1.0,C = 1e-4)
    funcEvals = 0
    x₊ = x₀ + α₀*(-g₀)
    f₊ = f(x₊)
    funcEvals += 1
    while f₊ > f₀ + C*α₀*g₀'*(-g₀)
        #α₀ = ρ*α₀
        c = f₀
        b = g₀'*(-g₀)
        a = (f(x₀ + α₀*(-g₀)) - b*α₀ - c)/α₀^2
        funcEvals += 1

        α₀ = -b/(2*a)
        x₊ = x₀ + α₀*(-g₀)
        f₊ = f(x₊)
        funcEvals += 1
    end
    return α₀, funcEvals
end

# Test on a toy example.
# f(x,y) = x^2 + xy + 2*y^2
# This is a convex function.
x₀ = [0.5;0.5]
f1(x) = x[1]^2 + x[1]*x[2] + 2*x[2]^2
g1(x) = [2*x[1] + x[2]; x[1] + 4*x[2]]
x, funcEvals, gradientEvals = gradientDescent(f1, g1, x₀,10e-8; maxIter = 1000)

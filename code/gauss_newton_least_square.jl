
"""
Least Squares, need to use this to implement Gauss-Newton Alg
"""

function LS(A,b; ϵ = 1e-14)
    """
    Solves a linear regression problem given
    the coefficient matrix A and the constant
    vector b. Return the x hat and the norm-2 of c2
    """
    n, m = size(A)
    F = qr(A, Val(true))
    c = F.Q' * b
    c1 = c[1:m]
    c2 = c[m+1:n]
    return F.P * inv(F.R) * c1
    #return F.P * backSub(F.R, c1), sqrt(c2' * c2)
end

"""
Implement Gauss-Newton algorithm.
This function works for nonlinear least square.
"""
function gaussNewton(r::Function, J::Function, x₀,ϵ=10e-8;maxIter = 25)

    f(x) = r(x)'*r(x)/2

    funcEvals = 0
    jacobianEvals = 0
    k = 0

    r₀ = r(x₀)
    funcEvals += 1
    f₀ = r(x₀)'*r(x₀)/2
    J₀ = J(x₀)
    jacobianEvals += 1

    g₀ = J₀'*r₀
    while norm(g₀) > ϵ && k <= maxIter
        #Search Direction
        p = LS(J₀, -r₀)

        #Step Length Backtracking
        α, evals = backtrack(f,f₀,g₀,p,x₀)
        funcEvals += evals

        #Step Length Backtrack with quadratic interpolation
        #α, evals = backtrackQuad(f,f₀,g₀,x₀)
        #funcEvals += evals

        #Fix step Length
        #α = 1

        #Update Parameter
        x₀ += α*p
        r₀ = r(x₀)
        funcEvals += 1
        f₀ = r(x₀)'*r(x₀)/2
        J₀ = J(x₀)
        jacobianEvals += 1
        g₀ = J₀'*r₀
        k +=1
    end
    x = x₀
    return x, funcEvals, jacobianEvals
end

"""Toy Example to test gaussNewton function"""
x̃ = [8.3;11.0;14.7;19.7;26.7;35.2;44.4;55.9]
t = [1.0;2.0;3.0;4.0;5.0;6.0;7.0;8.0]
r1(x) = x[1]*exp.(x[2]*t) - x̃
J1(x) = hcat( exp.(t*x[2]), t.*exp.(t*x[2])*x[1] )
θ₀ = [0.2;0.8]
θ, funcEvals, gradientEvals = gaussNewton(r1, J1, θ₀,10e-8; maxIter = 25)

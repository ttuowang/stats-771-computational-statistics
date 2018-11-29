using LinearAlgebra

"""
Name: newtonMethod
Description: implements newton's method for solving a system of equations with
a line search (amijo backtracking)

INPUTS:
1. F :: Function, evaluates the system
2. J :: Function, evaluates the Jacobian of the system
3. x₀ :: Array{Float64,1}, starting point
4. ϵ :: Float64, tolerance. Defaults to 10e-8

KEYWORD INPUTS:
1. maxIter :: Int64, maximum number of iterations. Defaults to 25

OUTPUTS:
1. X :: Array{Float}, solution to the system of equations within tolerance
2. funcEvals :: Int64, number of function evaluations
3. jacobEvals :: Int64, number of jacobian evaluations
"""

function newtonMethod(F::Function, J::Function, x₀,ϵ=10e-8;maxIter = 25)

    funcEvals = 0
    jacobEvals = 0
    k = 0

    F₀ = F(x₀)
    funcEvals += 1

    while norm(F₀) > ϵ && k <= maxIter
        #Search Direction
        J₀ = J(x₀)
        jacobEvals += 1
        d = -(J₀\F₀)

        #Step Length
        α, evals = armijoBacktrack(F,F₀,d,x₀)
        funcEvals += evals
        #α = 1

        #Update Parameter
        x₀ += α*d
        F₀ = F(x₀)
        funcEvals += 1
        k +=1
      end

    return x₀, funcEvals, jacobEvals
end

# Test Function
x₀ = [0;0]
FF(x) = [x[1]^2,x[2]^2]
JJ(x) = [2*x[1] 0; 0 2*x[2]]
x, funcEvals, jacobEvals = newtonMethod(FF, JJ, x₀; maxIter = 25)

function test(f::Function, n)
    return f(n)
end

"""
Name: backtrack \n
Description: implements backtracking line search on 0.5‖S‖² \n
INPUTS: \n
1. F (Function), evaluates the system
2. F₀ (Function), Function evaluated at a point x₀
3. J₀ (Function), Jacobian evaluated at a point x₀
4. d (Array{Float64,1}), search direction
5. x₀ (Array{Float64,1}), starting point
6. α₀ (Float64), starting search length defaults to 1
7. ρ (Float64), decay of search length, usually between [0.1,0.5], defaults to 0.5
8. z (Float64), defaults to 1e-4 \n

OUTPUTS: \n
1. α (Float64), step length
2. funcEvals (Int64), number of function evaluations
"""
function armijoBacktrack(F,F₀,d,x₀,α₀ = 1.0, ρ = 0.5, z = 1e-4)
    funcEvals = 0
    x₊ = x₀ + α₀*d
    F₊ = F(x₊)
    funcEvals += 1
    while norm(F₊) >= (1-z*α₀)*norm(F₀)
        α₀ = ρ*α₀
        x₊ = x₀ + α₀*d
        F₊ = F(x₊)
        funcEvals += 1
        #print(norm(F₊))
        #print('\n')
    end
    return α₀, funcEvals
end


"""
Name: chordNewton
Description: implements chord newton's method for solving a system of equations with
a line search (amijo backtracking)

INPUTS:
v :: Int64, compute the Jacobian every v iterations
"""

function chordNewton(F::Function, J::Function, x₀,ϵ=10e-8,v=3;maxIter = 25)

    funcEvals = 0
    jacobEvals = 0
    k = 0

    F₀ = F(x₀)
    funcEvals += 1

    while norm(F₀) > ϵ && k <= maxIter
        #Search Direction
        #print(k%v)
        if k%v == 0
            global J₀
            J₀ = J(x₀)
            jacobEvals += 1
        end

        #print(J₀)
        d = -(J₀\F₀)
        #Step Length
        α, evals = armijoBacktrack(F,F₀,d,x₀)
        funcEvals += evals
        #α = 1

        #Update Parameter
        x₀ += α*d
        F₀ = F(x₀)
        funcEvals += 1
        k +=1
      end
    return x₀, funcEvals, jacobEvals
end

function f()

    for i = 1:3
        if i == 1 || i==3
            global k
            k = 5
        end
        k = k+ 10
        print(k)
    end
end;

"""
Name: forwardDiff
Description: implements Forward Difference newton's method for solving a system of equations with
a line search (amijo backtracking)

INPUTS:
δ :: Float64, approximation size
"""
function forwardDiff(F::Function, J::Function, x₀,δ=1e-5,ϵ=10e-8;maxIter = 25)

    funcEvals = 0
    jacobEvals = 0
    k = 0

    F₀ = F(x₀)
    funcEvals += 1
    J₀ = J(x₀)
    jacobEvals += 1
    n,m = size(J₀)

    while norm(F₀) > ϵ && k <= maxIter
        #Search Direction
        d = -(J₀\F₀)

        #Step Length
        α, evals = armijoBacktrack(F,F₀,d,x₀)
        funcEvals += evals
        #α = 1

        #Update Parameter
        x₀ += α*d
        F₀ = F(x₀)
        funcEvals += 1
        k +=1

        #Update Jacobian
        J₊ = zeros(n,m)
        for i = 1:n
            g = zeros(1,m)
            Id = Matrix{Float64}(I,m,m)
            for j = 1:m
                J₊[i,j] = (F(x₀ + δ*Id[:,j])[i] - F(x₀)[i])/δ
            end
        end
        J₀ = J₊
    end
    return x₀, funcEvals, jacobEvals
end

"""
Name: quasiNewton
Description: implements quais-newton method(brayden Methods) for solving a s
ystem of equations with a line search (amijo backtracking)

INPUTS:
"""
function quasiNewton(F::Function, J::Function, x₀,ϵ=10e-8;maxIter = 25)

    funcEvals = 0
    jacobEvals = 0
    k = 0

    F₀ = F(x₀)
    funcEvals += 1
    J₀ = J(x₀)
    jacobEvals += 1
    Jinv = inv(J₀)
    while norm(F₀) > ϵ && k <= maxIter
        #Search Direction

        d = -(Jinv*F₀)

        #Step Length
        α, evals = armijoBacktrack(F,F₀,d,x₀)
        funcEvals += evals
        #α = 1

        #Update Parameter
        x₊ = x₀ + α*d
        F₊ = F(x₊)
        funcEvals += 1
        k +=1

        #Update J inverse
        y₊ = F₊ - F₀
        s₊ = x₊ - x₀
        Jinv = Jinv - (Jinv*y₊*s₊'*Jinv - s₊*s₊'*Jinv)/(s₊'*Jinv*y₊)

        #Restore F and x
        F₀ = F₊
        x₀ = x₊
      end

    return x₀, funcEvals, jacobEvals
end

"""
Name: Example functions
Description: Create a example F and J for testing our nonlinear solver
"""
function F(x)
    F₁ = (1.5-x[1]+x[1]*x[2])*(x[2]-1) +
         (2.25-x[1]+x[1]*x[2]^2)*(x[2]^2-1) +
         (2.625-x[1]+x[1]*x[2]^3)*(x[2]^3-1)
    F₂ = (1.5-x[1]+x[1]*x[2])*x[1] +
         (2.25-x[1]+x[1]*x[2]^2)*(2*x[1]*x[2]) +
         (2.625-x[1]+x[1]*x[2]^3)*(3*x[2]^2*x[1])
    return [F₁;F₂]
end

function J(x)
    ∂F₁∂x = (x[2]-1)^2 + (x[2]^2-1)^2 + (x[2]^3-1)^2
    ∂F₁∂y = x[1]*(x[2]-1) + (1.5-x[1]+x[1]*x[2]) + 2*x[1]*x[2]*(x[2]^2-1) +
          2*x[2]*(2.25 - x[1] + x[1]*x[2]^3) + 3*x[1]*x[2]*(x[2]^3-1) +
          3*x[2]^2*(2.625 - x[1] + x[1]*x[2]^3)
    ∂F₂∂x = (-1+x[2])*x[1] + (1.5-x[1]+x[1]*x[2]) + (-1+x[2]^2)*2*x[1]*x[2] +
          (2.25-x[1] + x[1]*x[2]^2)*2*x[2] + (-1+x[2]^3)*(3*x[2]^2*x[1]) +
          (2.625 - x[1] + x[1]*x[2]^3)*(3*x[2])
    ∂F₂∂y = x[1]^2 + (2*x[1]*x[2])^2 + 2*x[1]*(2.25-x[1]-x[1]*x[2]^2) +
          (3*x[1]*x[2]^2)^2 + 6*x[1]*x[2]*(2.625 - x[1] + x[1]*x[2]^3)
    return [∂F₁∂x ∂F₁∂y ; ∂F₂∂x ∂F₂∂y]
end

x₀ = [-1.0;-2.0]
x, funcEvals, jacobEvals = newtonMethod(F, J, x₀; maxIter = 25)
norm(F(x))

x₀ = [-1.0;0.0]
x, funcEvals, jacobEvals = chordNewton(F, J, x₀; maxIter = 25)
norm(F(x))

x₀ = [-1.0;0.0]
x, funcEvals, jacobEvals = forwardDiff(F, J, x₀; maxIter = 25)
norm(F(x))

x₀ = [-1.0;0.0]
x, funcEvals, jacobEvals = quasiNewton(F, J, x₀; maxIter = 25)
norm(F(x))

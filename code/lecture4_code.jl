using Pkg
Pkg.add("IJulia")
using LinearAlgebra

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

# Least Squares
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
    return F.P * inv(F.R) * c1, sqrt(c2' * c2)
    #return F.P * backSub(F.R, c1), sqrt(c2' * c2)
end

n, m = 10, 4
A = rand(10,4)
x = rand(4)
b = A*x

x_hat, _ = LS(A,b)

norm(x_hat - x)

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
    #Rtrm = F.R[1:s,1:s]
    #r = maximum(find(abs.(diag(Rtrm)) .>= ϵ))
    r = rank(F.R)
    l = m - r

    #Generate R and S
    R, S = F.R[1:r,1:r], F.R[1:r,r+1:end]
    d, P = inv(R)*(F.Q'*b)[1:r], inv(R)*S
    #d, P = backSub(R, (F.Q'*b)[1:r])
    z2 = inv(P'*P + Matrix{Float64}(I,l,l)) * P'* d
    #z2 = consistentLS(P'*P + I, P'*d)
    z1 = d - P*z2
    return F.P*vcat(z1,z2)
end


n, m = 10, 4
A = rand(10,4)
x = rand(4)
b = A*x

x_hat = underLS(A,b)

norm(x_hat - x)

## Constrained Least Square

using Random
Random.seed!(0);
using LinearAlgebra

function consistentLS(A,b)
    """
    Solves a consistent linear system given
    the coefficient matrix A and the constant
    vector b. Assumes A is consistent.
    """
    n, m = size(A)
    F = qr(A,Val(true))
    d = F.Q'*b
    c = F.R\d[1:m]
    #c = backSub(F.R, d[1:m])
    return F.P*c
end

# Least Squares
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
    #return F.P * inv(F.R) * c1, sqrt(c2' * c2)
    return F.P * backSub(F.R, c1), sqrt(c2' * c2)
end

function constrainedLS(A,b,C,d)
    p,m = size(C)
    F = qr(C')
    AQ = A*F.Q
    AQ1 = AQ[1:end,1:p]
    AQ2 = AQ[1:end,(p+1):end]
    R = F.R[1:p,1:end]
    y1 = consistentLS(R', d)
    #y1 = F.P'*y1
    y2, residual = LS(AQ2,b-AQ1*y1)
    return F.Q*vcat(y1,y2)
end

n = 10
m = 4
p = 2
A = rand(n,m)
x = rand(m)
b = A*x
C = rand(p,m)
d = C*x
println(x)
x_hat= constrainedLS(A,b,C,d)
println(x_hat)










#################################
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
    r = maximum(findall(abs.(diag(Rtrm)) .>= ϵ))
    l = m - r

    #Generate R and S
    R, S = F.R[1:r,1:r], F.R[1:r,r+1:end]
    d, P = invertUpperTri(R)*(F.Q'*b)[1:r], invertUpperTri(R)*S
    z2 = consistentLS(P'*P + I, P'*d)
    z1 = d - P*z2
    return F.P*vcat(z1,z2)
end

print("EXAMPLE 2: Fat Matrix\n")
n, m = 4, 10
A = rand(n,m)
b = rand(n)
println("A is an $n by $m matrix")
println("Error between underLS and 'truth':
    $(norm(underLS(A,b) - A\b))")

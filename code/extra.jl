function consistentLS(A,b)
    """
    Solves a consistent linear system given
    the coefficient matrix A and the constant
    vector b. Assumes A is consistent.
    """
    n, m = size(A)
    F = qr(A, Val(true))
    d = F.Q'*b
    c = F.R\d[1:m]
    return F.P*c
end

n, m = 10, 4
A = rand(10,4)
x = rand(4)
b = A*x

x_hat = consistentLS(A,b)
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

n, m = 10, 4
A = rand(10,4)
x = rand(4)
b = A*x

x_hat = underLS(A,b)

# Householder
using LinearAlgebra

function householder(a)
    """
    Computes the householder reflection
    given a nonzero vector a.
    """
    nrm_a = norm(a,2)
    nrm_a == 0 && error("Input vector is zero.")

    d = length(a)
    v = copy(a)
    v[1] = v[1] - nrm_a
    H = Matrix{Float64}(I,d,d) - 2*v*v'/dot(v,v)
    return H
end

a = rand(10)
a = [0.8147, 0.9058, 0.1270, 0.9134, 0.6324]
H = householder(a)

function householderQR(A)
    """
    Implement the householder QR decompostion.
    Input a m * n matrix with full rank.
    Output Q, R. (Don't) need to give all the columns of Q
    """
    #A =B'
    m, n = size(A)
    r = rank(A)
    if r != min(m, n)
        println("The input matrix is not full rank")
    else
        H1 = householder(A[:,1])
        A2 = H1 * A
        Q = H1
        Ai = A2
        for i in 2:r
            Hi_hat = householder(Ai[i:end, i])
            b1 = hcat(Matrix{Float64}(I,i-1,i-1), zeros(i-1,m-i+1))
            b2 = hcat(zeros(m-i+1, i-1), Hi_hat)
            Hi = vcat(b1, b2)
            Ai = Hi * Ai
            Q = Q*Hi
        end
        R = Q'*A
    end
    return Q, R
end


m, n = 10, 4
A = rand(m, n)
Q, R = householderQR(A)
norm(A-Q*R, 2)

B = [0.8147 0.9058 0.1270 0.9134 0.6324;
     0.0975 0.2785 0.5469 0.9575 0.9649;
     0.1576 0.9706 0.9572 0.4854 0.8003]

Q, R = householderQR(B')

# Givens Rotation

function givens_rot(a,i,j)
    """
    Computes the Givens Rotation for a
    vector 'a' at indices i and j, where
    the index at j is set to zero.
    """
    d = length(a)
    (i > d || j > d) && error("Index out of range.")
    l = sqrt(a[i]^2 + a[j]^2)
    λ = a[i]/l
    σ = a[j]/l
    G = ones(d)
    G[i] = λ
    G[j] = λ
    G = diagm(0 => G)
    G[i,j] = σ
    G[j,i] = -σ
    return G
end

function givensQR(A)
    """
    Implement Givens Rotation QR decompostion.
    Input a m * n full rank matrix.
    Output Q and R.
    """
    m, n = size(A)
    r = rank(A)
    if r != min(m, n)
        println("The input matrix is not full rank")
    else
        Q = Matrix{Float64}(I,m,m)
        R = A
        for j in 1:n
            for i in m:-1:j+1
                Gij = givens_rot(A[:,j],i-1,i)
                A = Gij * A
                R = Gij * R
                Q = Q * Gij'
            end
        end
    end
    return Q, R
end

m, n = 10, 4
A = rand(m, n)
Q, R = givensQR(A)

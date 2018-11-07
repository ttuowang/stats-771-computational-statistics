using LinearAlgebra

# Gram-Schmidt QR decompostion
# With one loop
function gramschmidtQR(A)
    """
    Implement the gram-schmidt procedure.
    Input a full column rank matrix A.
    Output Q and R.
    """
    m, n = size(A)
    r = rank(A)
    if r != n
        println("The input matrix is not full column rank")
    else
        u1 = A[:,1]
        e1 = u1 ./ norm(u1)
        e = e1
        R = zeros(n,1)
        R[1] = A[:,1]'e1
        for i in 2:n
            ui = A[:, i] - sum(e.* (A[:,i]'*e), dims = 2)
            e = hcat(e, ui ./ norm(ui))
            t = zeros(n,1)
            t[1:i] = A[:,i]'*e
            R = hcat(R, t)
        end
    end
    return e, R
end

A = [1 1 0; 1 0 1; 0 1 1]
Q, R = gramschmidtQR(A)

m, n = 10, 4
A = rand(m, n)
Q, R = gramschmidtQR(A)

# Gram-Schmidt QR decompostion
# With two loops
function gramschmidtQR2(A)
    """
    Implement the gram-schmidt procedure.
    Input a full column rank matrix A.
    Output Q and R.
    """
    m, n = size(A)
    r = rank(A)
    if r != n
        println("The input matrix is not full column rank")
    else
        Q = zeros(m, n)
        R = zeros(n,n)
        R[1,1] = norm(A[:,1])
        Q[:,1] = A[:,1]./R[1,1]
        for k = 2:n
            z = A[:, k]
            for i = 1:k-1
                R[i,k] = A[:,k]'Q[:,i]
                z = z - R[i,k]Q[:,i]
            end
            R[k,k] = norm(z)
            Q[:, k] = z ./ R[k,k]
        end
    end
    return Q, R
end

A = [1 1 0; 1 0 1; 0 1 1]
Q, R = gramschmidtQR2(A)

Q1, R1 = modifiedGSQR(A)
# Example that gm failed
A = [1 1 1; 1e-8 0 0; 0 1e-8 0]
Q, R = gramschmidtQR2(A)

#Modified gram-schmidt QR decompostion
# with two loops
function modifiedGSQR(A)
    """
    Implement the modifies gram-schmidt procedure.
    Input a full column rank matrix A.
    Output Q and R.
    """
    m, n = size(A)
    r = rank(A)
    if r != n
        println("The input matrix is not full column rank")
    else
        Q = float(A)
        R = zeros(n,n)
        for k = 1:n
            R[k,k] = norm(Q[:,k])
            Q[:,k] = Q[:,k] ./ R[k,k]
            for j = k+1:n
                R[k,j] = A[:,j]'Q[:,k]
                Q[:,j] = Q[:,j] - R[k,j]Q[:,k]
            end
        end
    end
    return Q, R
end

A = [1 1 1; 1 1 0 ; 1 0 2;1 0 0]
Q, R = modifiedGSQR(A)

# Another way of writting Modified GramSchimidt
# Idea from Ralph and Erika.

function modifiedGramSchmidt(A)
    n, m = size(A)
    Q = zeros(n,m)
    Q[1:end,1] = A[1:end,1]/norm(A[1:end,1])
    for i = 2:m
        q = A[1:end,i]
        for j = 1:i-1
            q_prev = Q[1:end,j]
            q = q - q'*q_prev*q_prev/norm(q_prev)
        end
        Q[1:end,i] = q/norm(q)
    end
    return Q
end


A = randn(10,5)
Q= modifiedGramSchmidt(A)
Q, R = modifiedGSQR(A)


# Another way of writting Modified GramSchimidt
# Idea from Doug.

function mgs(X)
    n = size(X,1)
    p = size(X,2)
    Q = zeros(n, p)
    R = zeros(p,p)
    for i = 1:p # columns
        Q[:,i] = X[:,i] # copy next vector
        for j = 1:(i-1) # rows, use previous vectors one at a time
            R[j,i] = (Q[:,j])'*Q[:,i] # build next R value
            Q[:,i] = Q[:,i] - R[j,i]*Q[:,j]
#            display(Q)
        end
        R[i,i] = norm(Q[:,i]) # normalizing constant
        Q[:,i] = Q[:,i]./R[i,i] # normalize this vector
    end
    return Q, R
end

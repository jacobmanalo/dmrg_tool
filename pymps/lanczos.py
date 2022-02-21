# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:28:56 2022

Original Code from Dr. Salim Belhaiza
https://www.youtube.com/watch?v=S416IbCFeEA&t=185s

"""
import numpy as np
from scipy.sparse import linalg as la
import scipy as SP

def random_hermitian(n):
    #A=np.random.rand(n,n)
    A = SP.sparse.rand(n, n, density=0.01)
    Adag = A.conjugate().transpose()
    H = 0.5*(A+Adag)
    H = H.toarray()
    return H


def eigensolver(d,e,num_evals):
    evals, evecs = SP.linalg.eigh_tridiagonal(d, e, select='i', select_range=[0,num_evals-1]) 
    return evals, evecs

def lanczos_jake(A,num_iter,num_evals):
    """
    Tridiagonalization of matrix A using Gram-Schmidt Orthogonalization i.e.
    the Lanczos algorithm. To avoid loss of orthogonality due to numerical
    error, I included a QR decomposition of the Krylov vector matrix Q.

    Parameters
    ----------
    A : numpy array
        Matrix
    num_iter : int
        Number of Krylov vectors to be generated
    num_evals : numpy array
        Number of eigenvalues to be generated.

    Returns
    -------
    evals : numpy array
        Array of eigenvalues
    evecs_transformed : numpy array
        Array of eigenvectors in the original basis of A

    """
    m = A.shape[0]
    b = np.random.rand(m)
    Q = np.zeros((m, num_iter))
    q = b / np.linalg.norm(b)
    Q[:,0] = q
  
    for k in range(num_iter):

        v = np.dot(A, Q[:,k])
        #alpha[k] = np.dot(Q[:,k],v)
        v = v - b[k-1]*Q[:,k-1] - np.dot(Q[:,k],v)*Q[:,k]

        normv = np.linalg.norm(v)
        b[k]=normv

        eps = 1e-12
        if normv > eps:
            q = v/normv
            if k < num_iter - 1:
                Q[:,k+1]=q
        else:
            print("norm is zero!",k,normv)
            break

    Q,_ = np.linalg.qr(Q)
    Aprime = np.dot(np.transpose(Q),np.dot(A,Q))


    evals, evecs = la.eigs(Aprime,k=num_evals,which='SR')

    evecs_transformed = np.dot(Q,evecs)

    return evals.real, evecs_transformed

def lanczos(A, r0, num_iter, num_evals):
    """
    Tridiagonalization of matrix A using Gram-Schmidt Orthogonalization i.e.
    the Lanczos algorithm. To avoid loss of orthogonality due to numerical
    error, I included a QR decomposition of the Krylov vector matrix Q.

    Parameters
    ----------
    A : numpy array
        Matrix
    num_iter : int
        Number of Krylov vectors to be generated
    num_evals : numpy array
        Number of eigenvalues to be generated.

    Returns
    -------
    evals : numpy array
        Array of eigenvalues
    evecs_transformed : numpy array
        Array of eigenvectors in the original basis of A

    """
    m = A.shape[0]
    eps = 1e-20
    b = r0#np.random.rand(m)
    norm1 = np.linalg.norm(b)
    if norm1 < eps:
        b = np.random.rand(m)
    Q = np.zeros((m, num_iter))
    q = b / np.linalg.norm(b)
    Q[:,0] = q
    b = q 

    for k in range(num_iter):
        
        #v = np.matmul(A, Q[:,k])
        v = A.dot(Q[:,k])
        #alpha[k] = np.dot(Q[:,k],v)
        
        v = v - b[k-1]*Q[:,k-1] - np.dot(Q[:,k],v)*Q[:,k]

        normv = np.linalg.norm(v)
        b[k]=normv

        
        if normv > eps:
            q = v/normv
            if k < num_iter - 1:
                Q[:,k+1]=q
        else:
            print("norm is zero!",k,normv)
            break

    Q,_ = np.linalg.qr(Q)
    Aprime = np.dot(np.transpose(Q),A.dot(Q))

    
    evals, evecs = la.eigs(Aprime,k=num_evals,which='SR')

    evecs_transformed = np.dot(Q,evecs)

    return evals.real, evecs_transformed.real

def lanczos2(A, r0, num_iter, num_evals):
    a = []
    b = []
    m = r0.shape[0]
    v = np.zeros((m, num_iter)) #change for a dynamic version
    norm = np.linalg.norm(r0)
    eps = 1e-20
    if norm < eps:
        r = np.random.rand(m)
        norm = np.linalg.norm(r)
        b.append(norm)
    else: 
       r = r0
       b.append(norm)
    id = 0
    for i in range(num_iter):
        id = i
        if b[i] < eps:
            print("norm is zero!",i,b[i]) # to improve 
            break
        v[:,i] = (r/b[i])
        r = A.dot(v[:,i])
        if i > 0:
            r = r - b[i]*v[:,i-1]
        a.append(np.dot(v[:,i],r))
        r = r - a[i]*v[:,i]
        b.append(np.linalg.norm(r))
    
    evals, evecs = SP.linalg.eigh_tridiagonal(a,b[1:id+1],select='i', select_range=[0,num_evals-1])
    res = v.dot(evecs)
    return evals, res/np.linalg.norm(res) 
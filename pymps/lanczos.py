# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:28:56 2022

Original Code from Dr. Salim Belhaiza
https://www.youtube.com/watch?v=S416IbCFeEA&t=185s

"""
import numpy as np
from scipy.sparse import linalg as la
import scipy as SP
import sys

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



def lanczos2(A, r0, num_iter):
    """
    Obtain the lowest eigevalue and eigenvector of a Linear Operator A
    Implementation of Lanczos algorithm. 
    Following section 4.4 of the book " Templates for the solution of algebraic 
    eigenvalue problems: a practical guide"
    Parameters
    ----------
    A : Linear Operator 
        A linaer operador with the matrix a matrix-vector multiplication (matvec)  as
        a member function.
    r0: ket state compatible with the linear operator
        Initial guest. Compatible mean that the operation A.matvec(r0) is defined
    num_iter : int
        Number max of iteration
    Returns
    -------
    evals : lowest eigenvalue- float 64 
        
    evecs_transformed : ket corresponding to the lowest eigenvalue
    """
    a = []
    b = []
    shape = r0.shape
#    for i in range(len(r0.shape)):
#      shape.append(r0.shape[i])
#    #shape = r0.shape
    v = []    
    eps = sys.float_info.min
    tol = 1e-10
    eval_ref = 1
    ntest = 5 
    num_iter_max = num_iter + ntest - num_iter%ntest
    r = r0
    norm = np.linalg.norm(r0)
    b.append(norm)
    id = 0
    for i in range(num_iter_max):
        id = i        
        if b[i] < abs(eps*eval_ref):
            r = np.random.rand(*shape)
            r *= 1./np.linalg.norm(r)
            Orthogonalize(r,v)
            b[i] = np.linalg.norm(r)
            
        v.append(r/b[i])
        r = A.matvec(v[i])
        if i > 0:
            r -= b[i]*v[i-1]
        a.append(np.sum(v[i]*r))
        r -= a[i]*v[i]
        Orthogonalize(r,v)
        b.append(np.linalg.norm(r))
        if (i+1)%ntest == 0:
          evals, evecs = SP.linalg.eigh_tridiagonal(a,b[1:id+1],select='i', select_range=[0,0])
          error = abs(b[i+1]*evecs[i])
          eval_ref = evals
          if error < tol:
             break
    if error > tol:
        print("Lanczos failed, residual norm = {}".format(error))
      
    #res = np.transpose(v).dot(evecs)
    
    evecs_transformed = v[0]*evecs[0]    
    for i in range(1,id):
        evecs_transformed += v[i]*evecs[i] 
    
    return evals, evecs_transformed/np.linalg.norm(evecs_transformed) 



def Orthogonalize(r,v):
    """
    Ortogonalize vector r with the vectors 
    contained in v
    """
    fac = 0.7
    tol = 1e-14
    n0 = np.linalg.norm(r)
    for i in range(len(v)):
        prod = np.sum(r*v[i])
        if abs(prod) > tol:
            r -= prod*v[i]
    if np.linalg.norm(r)/n0 < fac:
        return
    for i in range(len(v)):
        prod = np.sum(r*v[i])
        if abs(prod) > tol:
            r -= prod*v[i]
            
#def lanczos2t(A, r0, num_iter):
#    """
#    Obtain the lowest eigevalue and eigenvector of a Linear Operator A
#    Implementation of Lanczos algorithm. 
#    Following section 4.4 of the book " Templates for the solution of algebraic 
#    eigenvalue problems: a practical guide"
#    Parameters
#    ----------
#    A : Linear Operator
#    
#    r0: vector
#        Initial guest 
#    num_iter : int
#        Number max of iteration
#    Returns
#    -------
#    evals : numpy array
#        Array of eigenvalues
#    evecs_transformed : numpy array
#        Array of eigenvectors in the original basis of A
#    """
#    a = []
#    b = []
#    m = r0.shape[0]
#    v = []    
#    eps = sys.float_info.min
#    tol = 1e-13
#    eval_ref = 1
#    num_iter_max = num_iter + 5 - num_iter%5
#    r = r0
#    norm = np.linalg.norm(r0)
#    b.append(norm)
#    id = 0
#    for i in range(num_iter_max):
#        id = i
#        
#        if b[i] < abs(eps*eval_ref):
#            r = np.random.rand(m)
#            r *= 1./np.linalg.norm(r)
#            Orthogonalize(r,v)
#            b[i] = np.linalg.norm(r)
#            
#        v.append(r/b[i])
#        r = A.dot(v[i])
#        if i > 0:
#            r -= b[i]*v[i-1]
#        a.append(np.dot(v[i],r))
#        r -= a[i]*v[i]
#        Orthogonalize(r,v)
#        b.append(np.linalg.norm(r))
#        if (i+1)%5 == 0:
#          evals, evecs = SP.linalg.eigh_tridiagonal(a,b[1:id+1],select='i', select_range=[0,0])
#          error = abs(b[i+1]*evecs[i])
#          eval_ref = evals
#          if error < tol:
#             break
#    if error > tol:
#        print("Lanczos failed, residual norm = {}".format(error))
#      
#    res = np.transpose(v).dot(evecs)
#    return evals, res/np.linalg.norm(res) 
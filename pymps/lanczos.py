# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:28:56 2022

from Wikipedia: https://en.wikipedia.org/wiki/Lanczos_algorithm
"""
import numpy as np
from scipy.sparse import linalg as la
import scipy as SP
from numba import jit

#@jit
def eigensolver(d,e,num_evals):
    evals, evecs = SP.linalg.eigh_tridiagonal(d, e, select='i', select_range=[0,num_evals-1]) 
    return evals, evecs

def lanczos(A,num_iter,num_evals):
    m = A.shape[0]
    b = np.random.rand(m)
    #T = np.zeros((num_iter+1,num_iter+1))
    d = np.zeros((num_iter+1))
    e = np.zeros((num_iter))
    v = b/np.linalg.norm(b)

    
    for k in range(num_iter+1):
        if k== 0:
            beta = 0.
            v0=0.
        wp = np.dot(A,v)
        a = np.dot(v.conj(),wp)
        w = wp - a*v-b*v0
        
        b=np.linalg.norm(w)
        if b==0:
            break
        v0=v
        v=w/b
        d[k]=a
        #T[k,k]=a
        #input()
        if k <= num_iter-1:
            e[k]=b
            #T[k+1,k]=b
           # T[k,k+1]=b
    #print('d shape',d.shape)
    #print('e shape',e.shape)
        evals, evecs = eigensolver(d,e,num_evals)
    #return d,e
    
    return evals, evecs

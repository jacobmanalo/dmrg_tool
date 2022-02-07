#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy import linalg as la
import scipy as SP



one = np.array([[1.0,0.0],[0,1.0]])
sg = np.array([[1.0,0],[0,-1.0]])
cdag = np.array([[0,1.0],[0,0]])
c = np.array([[0,0],[1.0,0]])
occ = np.array([1,0])
emp = np.array([0,1])


def SumOpTerms(op1, op2):
    if(len(op1.shape)< 3):
     dim = 2;
     res = np.array([[[[0.]*2]*2]*dim]*dim)
     res[0,0] = op1;
     res[1,1] = op2;
    if(len(op1.shape)>= 3):
     dim = op1.shape[0] + 1
     res = np.array([[[[0.]*2]*2]*dim]*dim)
     res[0:dim-1,0:dim-1] = op1;
     res[dim-1,dim-1] = op2;
    return res

class LocalOperator:
    def __init__(self, O, L):
        self.O = O #np.array([[[0]*2]*2]*L)
        self.L = L
    def __mul__(self, op2):
        if(type(op2)==float):
            coeff = abs(op2)**(1./self.L)
            sign = 0
            if abs(op2) > 1e-10:
             sign = op2/abs(op2)
            return LocalOperator(self.O*coeff*sign,self.L)
        res = self.O
        for j in range(self.L):
          res[j] = np.matmul(self.O[j],op2.O[j])
        return LocalOperator(res, self.L)
    def __add__(self, op2):
        Ovec = []
        for j in range(self.L):
            Ovec.append( [self.O[j],op2.O[j]])
        return Ovec
    
  
   
def FermiOp(i, L, dagged):
    if dagged == 1:
       f = cdag
    else:
       f = c;
    O = np.array([[[0.]*2]*2]*L)
    for j in range(L):
        if j < i :
          O[j] = sg
        elif j > i:
          O[j] = one
        elif j==i:
          O[j] = f
    return LocalOperator(O,L)

    
class Hamiltonian:
    """A simple Hamiltonian class"""  
    
    def __init__(self, L):
        self.Ovec = []
        self.L = L          

    def add(self, op2, coeff = 1):
         if len(self.Ovec) > 0:
           if(len(self.Ovec.shape)< 4):
             dim = 2;
             res = np.array([[[[0.]*2]*2]*dim]*self.L)
             for j in range(self.L):
               res[j,0] = self.Ovec[j];
               res[j,1] = op2.O[j];
             self.Ovec = coeff*res
           else :
               dim = self.Ovec.shape[1]+1
               res = np.array([[[[0.]*2]*2]*dim]*self.L)
               for j in range(self.L):
                   for i in range(dim-1):
                     res[j,i] = self.Ovec[j,i];
                   res[j,dim-1] = op2.O[j];
               self.Ovec = coeff*res
        
         else:
             self.Ovec = coeff*op2.O
         return self.Ovec
     
    def GetMPO(self):
         nterms = self.Ovec.shape[1]
         H = np.array([[[[0.]*nterms]*2]*2]*self.L)
         states = np.array([[0,1],[1,0]])
         for j in range(self.L):
             for n in range(2):
                 for n1 in range(2):
                     for i in range(nterms):
                        H[j,n,n1,i] = np.dot(states[n],np.matmul(self.Ovec[j,i],states[n1]))
                        
         return H
 
    def GetMPOTensors(self):
         H = self.GetMPO()
         L = self.L
         O0 = H[0]
         hMPO = [tn.Node(O0,axis_names=["n_0p","n_0","i_0"] )]
         nterms = self.Ovec.shape[1]
         Oj = np.array([[[[0.]*nterms]*nterms]*2]*2)
         for j in range(1,self.L-1):
             for n1 in range(2):
                 for n2 in range(2):
                  Oj[n1,n2] = np.diag(H[j,n1,n2])
             hMPO += [tn.Node(Oj,axis_names=["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])]
         
         OL = H[self.L-1]
         hMPO += [tn.Node(OL,axis_names=["n_{}p".format(L-1),"n_{}".format(L-1),"i_{}".format(L-2)] )]
         
         
             
         #simplify tensors using SVD        
         u, vh, trun_err = tn.split_node(hMPO[0], left_edges=[hMPO[0]["n_0p"],hMPO[0]["n_0"]],right_edges=[hMPO[0]["i_0"]],max_truncation_err=0.01,edge_name="g")
         hMPO[0] = tn.Node(u.tensor,axis_names=["n_0p","n_0","i_0"] )
         
         for j in range(1,L-1):
            hMPO[j]= tn.Node(tn.ncon([vh.tensor,hMPO[j].tensor],[(-3,1),(-1,-2,1,-4)]),axis_names=["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])
            u, vh, trun_err = tn.split_node(hMPO[j], left_edges=[hMPO[j]["n_{}p".format(j)],hMPO[j]["n_{}".format(j)],hMPO[j]["i_{}".format(j-1)]],right_edges=[hMPO[j]["i_{}".format(j)]],max_truncation_err=0.01,edge_name="g")
            print(j)
            print(tn.shape(vh))
            print(tn.shape(hMPO[j+1]))
            hMPO[j] = tn.Node(u.tensor,axis_names=["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])
         hMPO[L-1]= tn.Node(tn.ncon([vh.tensor,hMPO[L-1].tensor],[(-3,1),(-1,-2,1)]),axis_names=["n_{}p".format(L-1),"n_{}".format(L-1),"i_{}".format(L-2)])
         
         connected_edges2=[]
         for j in range(1,L):
            conn2 = hMPO[j-1]["i_{}".format(j-1)]^hMPO[j]["i_{}".format(j-1)]
            connected_edges2.append(conn2)
        
         return hMPO,connected_edges2
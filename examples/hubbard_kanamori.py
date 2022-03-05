#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy import linalg as la
import scipy as SP

import time

import sys
sys.path.append('../') 
import pymps as mp
<<<<<<< HEAD
n_dots = 2         
=======
n_dots = 3
>>>>>>> 7458ede07542113e860c8cab7efc992a253c1bc5
L = 4*n_dots


#PARAMETERS

def v_x(x):
    A=51.2541
    a=5.62548
    B=4.77128
    f_x=A/(x+a)+B
    return f_x

def t_x(x):
    A=29.11
    alpha=0.25
    f_x=A*np.exp(-alpha*x)
    return f_x

def ix(dot,orb,spin):
    """
    Converts indices to a single index

    Parameters
    ----------
    dot : int
        Dot number
    orb : int
        Orbital number (0 for p- and 1 for p+)
    spin : int
        0 for spin down, 1 for spin up.

    Returns
    -------
    int

    """
    return 4*dot+2*orb+spin
    

d_int=10.
#d_int = 15

#x.1 PARAMS
U=15.9712
J=2.49952*2.
V = v_x(d_int)
W= 0.
t = t_x(d_int) #2.389494309941694 meV
delta = 0.84439

def C_dag(i):
    return mp.FermiOp(i,L,1)
def C(i):
    return mp.FermiOp(i,L,0) 

def N(i):
    return C_dag(i)*C(i)

def Sp(i,alpha):
    return C_dag(ix(i,alpha,1))*C(ix(i,alpha,0))

def Sm(i,alpha):
    return C_dag(ix(i,alpha,0))*C(ix(i,alpha,1))

t1=time.time()
H = mp.QuantumOperator(L)
"""
The following code is for the single dot Hamiltonian
"""
for i in range(n_dots):
    for alpha in range(2):
        H.add(N(ix(i,alpha,1))*N(ix(i,alpha,0))*U)


delta_2 = delta/2.
U_J_4=U-J/4.
for i in range(n_dots):
    for spin in range(2):
        H.add(C_dag(ix(i,0,spin))*C(ix(i,1,spin))*(delta_2))
        H.add(C_dag(ix(i,1,spin))*C(ix(i,0,spin))*(delta_2))
        
for i in range(n_dots):
    for spin1 in range(2):
        for spin2 in range(2):
            H.add(N(ix(i,0,spin1))*N(ix(i,1,spin2))*U_J_4)

J1=-0.5*J
J2 = -0.25*J
for i in range(n_dots):
    H.add(Sm(i,0)*Sp(i,1)*J1)
    H.add(Sm(i,1)*Sp(i,0)*J1)
    H.add(N(ix(i,0,1))*N(ix(i,1,1))*J2)
    H.add(N(ix(i,0,0))*N(ix(i,1,0))*J2)
    H.add(N(ix(i,0,1))*N(ix(i,1,0))*(-J2))
    H.add(N(ix(i,0,0))*N(ix(i,1,1))*(-J2))





"""
The following code is for the interaction Hamiltonian
"""
tunn=2.389494309941694
for i in range(n_dots-1):
    for alpha in range(2):
        for spin in range(2):
            H.add(C_dag(ix(i,alpha,spin))*C(ix(i+1,alpha,spin))*(tunn))
            H.add(C_dag(ix(i+1,alpha,spin))*C(ix(i,alpha,spin))*(tunn))
            
            

for i in range(n_dots-1):
    for alpha in range(2):
        for beta in range(2):
            for spin1 in range(2):
                for spin2 in range(2):
                    H.add(N(ix(i,alpha,spin1))*N(ix(i+1,beta,spin2))*V)


#POSITIVE BACKGROUND
V2 = -2.*V
for i in range(n_dots-1):
    for alpha in range(2):
        for spin in range(2):
            H.add(N(ix(i,alpha,spin))*(V2))
            H.add(N(ix(i+1,alpha,spin))*(V2))


"""
Chemical potential term
"""
chem_pot = 20.
for i in range(L):
   H.add(N(i)*(-chem_pot))
   
   


bonddim = 100
MPO, MPO_edges = H.GetMPOTensors()
t2=time.time()
print("Finished building MPO=",t2-t1)

<<<<<<< HEAD
=======
bonddim = 100
MPO, MPO_edges = H.GetMPOTensors()
>>>>>>> 7458ede07542113e860c8cab7efc992a253c1bc5


#MAKE PSI
# =============================================================================
# psi = np.random.random_sample(tuple([2]*L))
# psi = psi/np.linalg.norm(psi)
# =============================================================================

print(":)")
#input()
# =============================================================================
# norm= 0.
# single_tuple = list([0]*L)
# for i in range(4):
#     single_tuple[i] = 1
# for tup in set(itt.permutations(single_tuple,L)):
#     psi[tup] = np.random.uniform(-1,1)
#     #norm += np.abs(psi[tup])**2 
# #norm = np.sqrt(norm)
# psi = psi/np.linalg.norm(psi)
# =============================================================================
#psi = np.random.random_sample(tuple([2]*L))



#INITIALIZE WAVEFUNCTION
# =============================================================================
# arg=[0]*L
# for i in range(n_dots):
#     for alpha in range(2):
#         if i % 2 == 0:
#             arg[ix(i,alpha,0)]=1
#         else:
#             arg[ix(i,alpha,1)]=1
# 
# arg_not = [1-arg[k] for k in range(len(arg))]
# 
# 
# idx = []
# idx_not = []
# 
# idx.append(arg[0])
# idx_not.append(arg_not[0])
# for i in range(1,L-1):
#     idx.append(arg[i]+idx[i-1])
#     idx_not.append(arg_not[i]+idx_not[i-1])
# 
# 
# M = {}
# #bond_dim=500
# 
# M[0]=np.zeros((2,bonddim))
# M[0][arg[0],idx[0]]=1.
# M[0][arg_not[0],idx_not[0]]=np.random.rand()
# for i in range(1,L-1):
#     M[i]=np.zeros((2,bonddim,bonddim))
#     M[i][arg[i],idx[i-1],idx[i]]=np.random.rand()
#     M[i][arg_not[i],idx_not[i-1],idx_not[i]]=np.random.rand()
# M[L-1]=np.zeros((2,bonddim))
# M[L-1][arg[L-1],idx[L-2]]=np.random.rand()
# M[L-1][arg_not[L-1],idx_not[L-2]]=np.random.rand()
# 
# mps = mp.init_wavefunction(M,bonddim)
# =============================================================================


mps = mp.init_wavefunction(L, bonddim)
#input()
#bond_dim=4
#mps = mp.create_MPS(L,bond_dim)

t1 = time.time()
energy, MPS = mp.DMRG(L, MPO, 4, mps)
t2 = time.time()

print("Time of diag=",t2-t1)


#mpscopy = tn.replicate_nodes(mps)
#for i in range(L):
#    mps[i].tensor = MPS[i].tensor

# =============================================================================
# mps_con = []
# for i in range(1,L):
#     MPS[i-1]["i_{}".format(i-1)] ^ MPS[i]["i_{}".format(i-1)]
#     #mps_con.append(con)
# =============================================================================

# =============================================================================
# test=MPS[0]
# for i in range(1,L):
#     test@=MPS[i]
# =============================================================================

#test=mps[0]
#for i in range(1,L):
#    test@=mps[i]

#test1 = tn.contract(mps_con)
#test=MPS[0]@MPS[1]@MPS[2]


"""
Create S^2 operator
"""
S2 = mp.QuantumOperator(L)

for i in range(n_dots):
    for alpha in range(2):
        for j in range(n_dots):
            for beta in range(2):
                S2.add(Sp(i,alpha)*Sm(j,beta)*0.5) #Sx^2+Sy^2
                S2.add(Sm(i,alpha)*Sp(j,beta)*0.5) #Sx^2+Sy^2
                S2.add(N(ix(i,alpha,1))*N(ix(j,beta,1))*(0.25)) #Sz^2
                S2.add(N(ix(i,alpha,0))*N(ix(j,beta,0))*(0.25)) #Sz^2
                S2.add(N(ix(i,alpha,1))*N(ix(j,beta,0))*(-0.25)) #Sz^2
                S2.add(N(ix(i,alpha,0))*N(ix(j,beta,1))*(-0.25)) #Sz^2
Stot2 = S2.ExpectatioValue(MPS)


"""
Defining number operator for expectation value
"""
n = mp.QuantumOperator(L)
for i in range(L):
   n.add(N(i))
number_e = n.ExpectatioValue(MPS)

#np.transpose(np.where(np.abs(test.tensor)>=1e-10))[0]
#number_e=np.count_nonzero(np.transpose(np.where(np.abs(test.tensor)>=1e-10))[0])

print('Energy = {}'.format(energy+number_e*chem_pot),'S^2=',Stot2)
print("Number of particles = {}".format(round(number_e)))

# =============================================================================
# f = open("GS_energy_{}dot.dat".format(n_dots),"w")
# f.write("Energy = {}, Number of electrons = {}, Bond dimension = {}".format(energy+number_e*chem_pot, number_e, bonddim))
# f.close()
# 
# for i in range(L):
#     w = open("M_{}_for_{}dots.mps".format(i,n_dots),"w")
#     if i == 0 or i == L-1:
#         for l in range(MPS[i].shape[0]):
#             for m in range(MPS[i].shape[1]):
#                 w.write("{} {} {}\n".format(l,m,MPS[i].tensor[l,m]))
#     else:
#         for l in range(MPS[i].shape[0]):
#             for m in range(MPS[i].shape[1]):
#                 for n in range(MPS[i].shape[2]):
#                     w.write("{} {} {} {}\n".format(l,m,n,MPS[i].tensor[l,m,n]))
#     w.close()
#                     
# =============================================================================

#m = 200
#M = 20
#d = 2
#L = np.random.random_sample((m,M,m))
#R = np.random.random_sample((m,M,m))
#W = np.random.random_sample((d,d,M,M))
#y = np.random.random_sample((d,m,m))
#t1 = time.time()
#yp = tn.ncon([L, W, R, y], [(-2,'d','a'),(-1,'c','d','i'),(-3,'i','b'),('c','a','b')])
#t2 = time.time()
#
#print("Contract time=",t2-t1)

# =============================================================================
# for i in range(L):
#    H.add(N(i)*(-chem_pot))
# for i in range(L-1):
#    H.add(C_dag(i)*C(i+1)*(t))
#    H.add(C_dag(i+1)*C(i)*(t))
#    H.add(N(i)*N(i+1)*(v))
# =============================================================================



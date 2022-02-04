#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy import linalg as la
import scipy as SP

import sys
sys.path.append('../') 
import pymps as mp
L = 3
H = mp.Hamiltonian(L)
e = 1.0
t = -1.0
chem_pot = 50.0

def C_dag(i):
    return mp.FermiOp(i,L,1)
def C(i):
    return mp.FermiOp(i,L,0) 

for i in range(L):
   H.add(C_dag(i)*C(i)*(-chem_pot))
#for i in range(L-1):
#   H.add(C_dag(i)*C(i+1)*t)
#   H.add(C(i)*C_dag(i+1)*t)
   
   
MPO, MPO_edges = H.GetMPOTensors()



mps = mp.init_wavefunction(L, conserve_n = True, num_e = 2)
energy, energies, MPS = mp.DMRG(L, MPO, 20, mps)

#MPS[0]["i_0"]^MPS[1]["i_0"]
#MPS[1]["i_1"]^MPS[2]["i_1"]
#MPS[2]["i_2"]^MPS[3]["i_2"]
mps_con = []
for i in range(1,L):
    con1 = MPS[i-1]["i_{}".format(i-1)] ^ MPS[i]["i_{}".format(i-1)]
    test = tn.contract(con1)
    
#test1 = tn.contract_between(mps_con)
#test=MPS[0]@MPS[1]@MPS[2]@MPS[3]

np.transpose(np.where(np.abs(test.tensor)>=1e-10))[0]
number_e=np.count_nonzero(np.transpose(np.where(np.abs(test.tensor)>=1e-10))[0])

print('Corrected Energy = {}'.format(energy + number_e*chem_pot))
print("Number of particles = {}".format(number_e))

for i in range(L-1):
   base = tn.contract(MPO_edges[i])

base
# Hubbard kanamori

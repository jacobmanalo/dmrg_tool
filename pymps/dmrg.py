import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy.sparse import linalg as la
import scipy as SP
import pymps as mp

import time
#from numba import jit

# =============================================================================
# @jit
# def eigensolver(T,num_evals):
#     #evals, evecs = SP.linalg.eigh_tridiagonal(d, e, select='i', select_range=[0,num_evals-1]) 
#     evals, evecs = la.eigs(T,k=num_evals,which='SR') 
#     return evals, evecs
# =============================================================================
#import lanczos

class SweepOpt:
    """
    A class for the sweeping optimization routine in the DMRG algorithm. It takes
    the Hamiltonian in MPO format and the MPS as input.
    
    Class methods:
        fit
        create_L
        create_R
        _forward_sweep
        _backward_sweep
    
    """
    
    
    def __init__(self, ham, MPS):
        self.ham = tn.replicate_nodes(ham, conjugate=False)
        self.MPS = tn.replicate_nodes(MPS, conjugate=False)
        self.MPS_star = tn.replicate_nodes(MPS,conjugate=True)
        self.n_sites=len(self.MPS)
        self.mpscopy = tn.replicate_nodes(self.MPS)
        for i in range(len(self.MPS)):
            self.ham[i]["n_{}".format(i)]^self.MPS[i]["n_{}".format(i)]
            self.ham[i]["n_{}p".format(i)]^self.MPS_star[i]["n_{}".format(i)]
        
        self.L = {}
        self.R = {}
        for p in range(self.n_sites-1,0,-1):
            if p == self.n_sites-1:
                self.R[p-1] = tn.ncon([self.MPS_star[p].tensor,self.ham[p].tensor,self.MPS[p].tensor],\
                            [('n{}p'.format(p),-1),('n{}p'.format(p),'n{}'.format(p),-2),('n{}'.format(p),-3)])
            else:
                self.R[p-1] = tn.ncon([self.MPS_star[p].tensor,self.ham[p].tensor,self.MPS[p].tensor,self.R[p]],\
                            [('n{}p'.format(p),-1,1),('n{}p'.format(p),'n{}'.format(p),-2,2),\
                             ('n{}'.format(p),-3,3),(1,2,3)])
        
        
    
    def fit(self, num_sweeps):
        """
        This is the sweeping part of the DMRG algorithm.
        
        Parameters
        ----------
        num_sweeps : int
            Number of sweeps. 1 sweep consists of going from one end to the
            other of the chain regardless of direction.

        Returns
        -------
        energy : float
            Lowest eigenvalue.
        energies : numpy array
            List of eigenvalues.
        mps : Tensor Network
            Resultant Matrix Product State.

        """
        sweep_forward = True
        for sweep in range(num_sweeps):
            print("Sweep Number: {} \n".format(sweep))
            if sweep_forward:
                energy = self._forward_sweep()
                sweep_forward = False
            else:
                energy = self._backward_sweep()
                sweep_forward = True
        return energy, self.MPS
    
# =============================================================================
#     def init_R_block(self):
#         #self.R = {}
#         for p in range(self.n_sites-1,0,-1):
#             if p == self.n_sites-1:
#                 self.R[p-1] = tn.ncon([self.MPS_star[p].tensor,self.ham[p].tensor,self.MPS[p].tensor],\
#                             [('n{}p'.format(p),-1),('n{}p'.format(p),'n{}'.format(p),-2),('n{}'.format(p),-3)])
#             else:
#                 self.R[p-1] = tn.ncon([self.MPS_star[p].tensor,self.ham[p].tensor,self.MPS[p].tensor,self.R[p]],\
#                             [('n{}p'.format(p),-1,1),('n{}p'.format(p),'n{}'.format(p),-2,2),\
#                              ('n{}'.format(p),-3,3),(1,2,3)]) 
#         return self.R
# =============================================================================
                
# =============================================================================
#     def init_R_block(self):
#         for p in range(self.n_sites-1,0,-1):
#             if p == self.n_sites-1:
#                 self.R[p-1] = tn.ncon([self.MPS_star[p].tensor,self.ham[p].tensor,self.MPS[p].tensor],\
#                             [('n{}p'.format(p),-1),('n{}p'.format(p),'n{}'.format(p),-2),('n{}'.format(p),-3)])
#             else:
#                 self.R[p-1] = tn.ncon([self.MPS_star[p].tensor,self.ham[p].tensor,self.MPS[p].tensor,self.R[p]],\
#                             [('n{}p'.format(p),-1,1),('n{}p'.format(p),'n{}'.format(p),-2,2),\
#                              ('n{}'.format(p),-3,3),(1,2,3)]) 
#         return self.R    
# =============================================================================
        
    def _forward_sweep(self):
        """
        This is the forward sweep. One full forward sweep goes from site
        0 to site L-1.

        Returns
        -------
        energy : float
            lowest eigenvalue.
        energies : numpy array
            list of eigenvalues.
        MPS : Tensor Network
            Matrix Product State
        """
        nsites = len(self.MPS)
        for i in range(nsites-1):
            if i==0:
                #rblock = self.create_R(i)
                hsuper = tn.ncon([self.ham[i].tensor,self.R[i]],[(-1,-3,'i{}'.format(i)),(-2,'i{}'.format(i),-4)])
                
            else:
                #lblock = self.create_L(i)
               # rblock = self.create_R(i)
                hsuper = tn.ncon([self.L[i],self.ham[i].tensor,self.R[i]],[(-2,'d',-5),(-1,-4,'d','i'),(-3,'i',-6)])
                
            num_edges_to_con = len(np.shape(self.ham[i]))-1
            hsuper_node = tn.Node(hsuper)
            hsuper_node_copy = tn.replicate_nodes([hsuper_node])
            
            dim_hsuper = 1
            hsuper_shape = np.shape(hsuper_node.tensor)
            
            for k in range(len(hsuper_shape)//2):
                dim_hsuper*=hsuper_shape[k]
            
            hsuper_matrix = np.reshape(hsuper_node.tensor, (dim_hsuper,dim_hsuper))

            #energies, evecs = la.eigs(hsuper_matrix,k=1,which='SR',maxiter=len(hsuper_matrix))
            energies, evecs = la.eigsh(hsuper_matrix,k=1,which='SA')

#THE FOLLOWING CODE IS TO ENABLE LANCZOS DIAGONALIZATION
# =============================================================================
#             num_iter = len(hsuper_matrix)-1
# # =============================================================================
# #             if len(hsuper_matrix) > 30:
# #                 num_iter=30
# # =============================================================================
#             energies, evecs = mp.lanczos(hsuper_matrix,num_iter,1)
# =============================================================================

            energy = min(energies)
            min_idx=np.argmin(energies)
            
            new_m = np.reshape(evecs[:,min_idx],np.shape(self.MPS[i].tensor))
            
          #  print("SHAPE BEFORE DIAG",self.MPS[i].tensor.shape)
          #  print(self.MPS[i].tensor,i,"\n","before")
            #input()
            
            self.MPS[i].tensor = new_m
            
            #DO SVD LEFT NORMALIZATION ON NEWFOUND M
            if i == 0:
                ledges = [self.MPS[i]["n_{}".format(i)]]
            else:
                ledges = [self.MPS[i]["n_{}".format(i)],self.MPS[i]["i_{}".format(i-1)]]
            
            redges = [self.MPS[i]["i_{}".format(i)]]
            q,r = tn.split_node_qr(self.MPS[i], left_edges=ledges, right_edges=redges, edge_name="ip_{}".format(i))
            
            if i == 0:
                q.reorder_edges([q["n_{}".format(i)],q["ip_{}".format(i)]])
            else:
                q.reorder_edges([q["n_{}".format(i)],q["i_{}".format(i-1)],q["ip_{}".format(i)]])
                
            
            if i==nsites-2:
                self.MPS[i+1].tensor = tn.ncon([r.tensor,self.MPS[i+1].tensor],[(-2,'k'),(-1,'k')])
            else:
                self.MPS[i+1].tensor = tn.ncon([r.tensor,self.MPS[i+1].tensor],[(-2,'k'),(-1,'k',-3)])
                
            
            self.MPS_star[i+1] = tn.replicate_nodes([self.MPS[i+1]],conjugate=True)[0]
            
            self.MPS[i].tensor = q.tensor
            self.MPS_star[i] = tn.replicate_nodes([self.MPS[i]],conjugate=True)[0]
            
            
            #print(self.MPS[i].tensor,i,"\n","after")
            #input()
            
            #self.MPS_star=tn.replicate_nodes(self.MPS,conjugate=True)
            print('site {}:   Energy={}\n'.format(i,energies.real))
            #print('time for eig',t2-t1)
           # print(i,self.R[i].shape,"SHAPE OF R")
           # print("SHAPE AFTER DIAG",self.MPS[i].tensor.shape)
            #UPDATE L AND R
            if i < nsites-2:
                self.R[i] = tn.ncon([self.MPS_star[i+1].tensor,self.ham[i+1].tensor,self.MPS[i+1].tensor,self.R[i+1]],\
                            [('n{}p'.format(i),-1,1),('n{}p'.format(i),'n{}'.format(i),-2,2),\
                             ('n{}'.format(i),-3,3),(1,2,3)])
            else:
                self.R[i] = tn.ncon([self.MPS_star[i+1].tensor,self.ham[i+1].tensor,self.MPS[i+1].tensor],\
                            [('n{}p'.format(i),-1),('n{}p'.format(i),'n{}'.format(i),-2),\
                             ('n{}'.format(i),-3)])
                
            
            if i == 0:
                self.L[i+1] = tn.ncon([self.MPS_star[i].tensor,self.ham[i].tensor,self.MPS[i].tensor],\
                                [('n{}p'.format(i),-1),('n{}p'.format(i),'n{}'.format(i),-2),('n{}'.format(i),-3)])
            else:
                self.L[i+1] = tn.ncon([self.L[i],self.MPS_star[i].tensor,self.ham[i].tensor,self.MPS[i].tensor],\
                                [(1,2,3),('n{}p'.format(i),1,-1),('n{}p'.format(i),'n{}'.format(i),2,-2),\
                                 ('n{}'.format(i),3,-3)]) 
        return energy.real

    
    def _backward_sweep(self):
        """
        This is the backward sweep. One full backward sweep goes from site
        L-2 to site 0.

        Returns
        -------
        energy : float
            lowest eigenvalue.
        energies : numpy array
            list of eigenvalues.
        MPS : Tensor Network
            Matrix Product State
        """
        nsites = len(self.MPS)
        for i in range(nsites-1,0,-1):
            if i == nsites-1:
               # lblock = self.create_L(i)
                hsuper = tn.ncon([self.L[i],self.ham[i].tensor],[(-2,'d',-4),(-1,-3,'d')])
            else:
               # lblock = self.create_L(i)
               # rblock = self.create_R(i)
                hsuper = tn.ncon([self.L[i],self.ham[i].tensor,self.R[i]],[(-2,'d',-5),(-1,-4,'d','i'),(-3,'i',-6)])
            
            num_edges_to_con = len(np.shape(self.ham[i]))-1
            hsuper_node = tn.Node(hsuper)
            
            dim_hsuper = 1
            hsuper_shape = np.shape(hsuper_node.tensor)
            
            for k in range(len(hsuper_shape)//2):
                dim_hsuper*=hsuper_shape[k]
            
            hsuper_matrix = np.reshape(hsuper_node.tensor, (dim_hsuper,dim_hsuper))

#THE FOLLOWING CODE IS TO ENABLE LANCZOS DIAGONALIZATION           
# =============================================================================
#             num_iter = len(hsuper_matrix)-1
# # =============================================================================
# #             if len(hsuper_matrix) > 100:
# #                 num_iter=100
# # =============================================================================
#             energies, evecs = mp.lanczos(hsuper_matrix,num_iter,1)
# =============================================================================
            #energies, evecs = eigensolver(hsuper_matrix,1)
            #energies, evecs = la.eigs(hsuper_matrix,k=1,which='SR',maxiter=len(hsuper_matrix))
            energies, evecs = la.eigsh(hsuper_matrix,k=1,which='SA')
            
                        
            energy = min(energies)
            min_idx=np.argmin(energies)
            
            new_m = np.reshape(evecs[:,min_idx],np.shape(self.MPS[i].tensor))
            
            self.MPS[i].tensor = new_m
            
            #DO SVD RIGHT NORMALIZATION ON NEWFOUND M
            if i == nsites-1:
                redges = [self.MPS[i]["n_{}".format(i)]]
            else:
                redges = [self.MPS[i]["i_{}".format(i)],self.MPS[i]["n_{}".format(i)]]
                
            ledges = [self.MPS[i]["i_{}".format(i-1)]]
            r,q = tn.split_node_rq(self.MPS[i], left_edges=ledges, right_edges=redges, edge_name="ip_{}".format(i-1))
            
            if i == nsites-1:
                q.reorder_edges([q["n_{}".format(i)],q["ip_{}".format(i-1)]])
                
            else:
                q.reorder_edges([q["n_{}".format(i)],q["ip_{}".format(i-1)],q["i_{}".format(i)]])
            
            if i == 1:
                self.MPS[i-1].tensor = tn.ncon([self.MPS[i-1].tensor, r.tensor],[(-1,'k'),('k',-2)])
            else:
                self.MPS[i-1].tensor = tn.ncon([self.MPS[i-1].tensor, r.tensor],[(-1,-2,'k'),('k',-3)])
            self.MPS_star[i-1] = tn.replicate_nodes([self.MPS[i-1]],conjugate=True)[0]
            
            self.MPS[i].tensor = q.tensor
            self.MPS_star[i] = tn.replicate_nodes([self.MPS[i]],conjugate=True)[0]

            if i > 1:
                self.L[i] = tn.ncon([self.L[i-1],self.MPS_star[i-1].tensor,self.ham[i-1].tensor,self.MPS[i-1].tensor],\
                                [(1,2,3),('n{}p'.format(i),1,-1),('n{}p'.format(i),'n{}'.format(i),2,-2),\
                                 ('n{}'.format(i),3,-3)])   
            else:
                self.L[i] = tn.ncon([self.MPS_star[i-1].tensor,self.ham[i-1].tensor,self.MPS[i-1].tensor],\
                                [('n{}p'.format(i),-1),('n{}p'.format(i),'n{}'.format(i),-2),\
                                 ('n{}'.format(i),-3)])   
                    
            
            
            
            
            if i == nsites-1:
                self.R[i-1] = tn.ncon([self.MPS_star[i].tensor,self.ham[i].tensor,self.MPS[i].tensor],\
                            [('n{}p'.format(i),-1),('n{}p'.format(i),'n{}'.format(i),-2),('n{}'.format(i),-3)])
            else:
                self.R[i-1] = tn.ncon([self.MPS_star[i].tensor,self.ham[i].tensor,self.MPS[i].tensor,self.R[i]],\
                            [('n{}p'.format(i),-1,1),('n{}p'.format(i),'n{}'.format(i),-2,2),\
                             ('n{}'.format(i),-3,3),(1,2,3)])
            ######
            
# =============================================================================
#             if i == 0:
#                 self.L[i+1] = tn.ncon([self.MPS_star[i].tensor,self.ham[i].tensor,self.MPS[i].tensor],\
#                                 [('n{}p'.format(i),-1),('n{}p'.format(i),'n{}'.format(i),-2),('n{}'.format(i),-3)])
#             else:
#                 self.L[i+1] = tn.ncon([self.L[i],self.MPS_star[i].tensor,self.ham[i].tensor,self.MPS[i].tensor],\
#                                 [(1,2,3),('n{}p'.format(i),1,-1),('n{}p'.format(i),'n{}'.format(i),2,-2),\
#                                  ('n{}'.format(i),3,-3)])
# =============================================================================
            
            
            #self.MPS_star=tn.replicate_nodes(self.MPS,conjugate=True)
            print('site {}:   Energy={}\n'.format(i,energies.real))
        return energy.real

# =============================================================================
#     def test_fsweep(self):
#         self.init_R_block()
#         self._forward_sweep()
# =============================================================================

def DMRG(n_sites,ham,n_sweeps,psi):
    #pass
    """
    The DMRG algorithm takes an initial guess MPS and a Hamiltonian in MPO
    form and uses the variational principle to obtain the ground state.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    ham : Tensor Network
        The Hamiltonian in MPO format.
    n_sweeps : int
        Number of sweeps.
    psi : Tensor Network
        Initial guess for the Matrix Product State.

    Returns
    -------
    energy : float
        Lowest eigenstate.
    energies : array
        list of eigenvalues.
    mps : Tensor Network
        Matrix Product State.

    """
    sweep_opt = SweepOpt(ham,psi)
    energy, mps = sweep_opt.fit(n_sweeps)
    return energy, mps 
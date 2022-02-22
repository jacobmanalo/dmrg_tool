import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy.sparse import linalg as la
import scipy as SP
import pymps as mp
from .lanczos import lanczos2

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

def SuperBlockLO(L, W, R):
    """
    A  function to define an auxiliary superblock Linear Operator. It the tensor L, W 
    and R,  and  defines the rules for aplication on vectors   
    """
    if L.shape[0] == 0:
        dim = W.shape[0]*R.shape[0]
        lmps_shape = (W.shape[0],R.shape[0])
        #new_ten =  tn.ncon([ W, R], [(-1,-5,'i'),(-3,'i',-6)])
        def matvec(y):
           mps_y = tn.reshape(y , lmps_shape )
           #new_v =  tn.ncon([new_ten, mps_y], [(-1,-2,'c','b'),('c','b')])
           new_v =  tn.ncon([W, R, mps_y], [(-1,'c','i'),(-3,'i','b'),('c','b')])
           return tn.reshape(new_v, dim)
    elif R.shape[0] == 0:
        dim = W.shape[0]*L.shape[0]
        lmps_shape = (W.shape[0],L.shape[0])
        #new_ten =  tn.ncon([L, W], [(-2,'d',-4),(-1,-5,'d')])
        def matvec(y):
          mps_y = tn.reshape(y , lmps_shape )
          #new_v =  tn.ncon([new_ten, mps_y], [(-1,-2,'a','c'),('c','a')])
          new_v =  tn.ncon([L, W, mps_y], [(-2,'d','a'),(-1,'c','d'),('c','a')])
          return tn.reshape(new_v, dim)
    else:
        dim = W.shape[0]*L.shape[0]*R.shape[0]
        lmps_shape = (W.shape[0],L.shape[0],R.shape[0])
        
        def matvec(y):          
          mps_y = tn.reshape(y , lmps_shape )          
          new_v =  tn.ncon([L, W, R, mps_y], [(-2,'d','a'),(-1,'c','d','i'),(-3,'i','b'),('c','a','b')])
          return tn.reshape(new_v, dim)
      
    dtype = 'float64'
    shape = (dim, dim)
           
    def matmul(M):        
        m = M.shape[1]
        A = []
        for i in range(m):
            A.append(matvec(M[:,i]))
        return A
    
    return la.LinearOperator(shape=shape, matvec=matvec, rmatvec=matvec, dtype=dtype) 
    
class SuperBlockLO1:
    """
    A class function to define an auxiliary superblock Linear Operator. It the tensor L, W 
    and R,  and  defines the rules for aplication on vectors   
    """
    def __init__(self, L, W, R):
        self.L = L
        self.R = R
        self.W = W
#       if L.shape[0] == 0:
#          self.shape = (W.shape[0],R.shape[0])
#          def matvec1(self,y):
#            new_v =  tn.ncon([W, R, y], [(-1,'c','i'),(-3,'i','b'),('c','b')])
#            return new_v
#       elif R.shape[0] == 0:
#          self.shape = (W.shape[0],L.shape[0])
#          def matvec1(self,y):
#            new_v =  
#            return new_v
#       else:
#           self.shape = (W.shape[0],L.shape[0],R.shape[0])        
#           def matvec(self,y):          
#            new_v =  tn.ncon([L, W, R, y], [(-2,'d','a'),(-1,'c','d','i'),(-3,'i','b'),('c','a','b')])
#            return new_v
    def matvec(self,y):
        if self.L.shape[0] == 0:
          return tn.ncon([self.W, self.R, y], [(-1,'c','i'),(-3,'i','b'),('c','b')])
        if self.R.shape[0] == 0:
           return tn.ncon([self.L, self.W, y], [(-2,'d','a'),(-1,'c','d'),('c','a')])
       
        return tn.ncon([self.L, self.W, self.R, y], [(-2,'d','a'),(-1,'c','d','i'),(-3,'i','b'),('c','a','b')])
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
 #      self.mpscopy = tn.replicate_nodes(self.MPS)
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
                #hsuper = tn.ncon([self.ham[i].tensor,self.R[i]],[(-1,-3,'i{}'.format(i)),(-2,'i{}'.format(i),-4)])
                #hsuper = SuperBlockLO(np.zeros(0), self.ham[i].tensor, self.R[i])
                hsuper = SuperBlockLO1(np.zeros(0), self.ham[i].tensor, self.R[i])
            else:
                #hsuper = tn.ncon([self.L[i],self.ham[i].tensor,self.R[i]],[(-2,'d',-5),(-1,-4,'d','i'),(-3,'i',-6)])
                #hsuper = SuperBlockLO(self.L[i], self.ham[i].tensor, self.R[i])
                hsuper = SuperBlockLO1(self.L[i], self.ham[i].tensor, self.R[i])
#            num_edges_to_con = len(np.shape(self.ham[i]))-1
                
            num_iter = min(np.prod(self.MPS[i].tensor.shape),30)
            #v0 = np.reshape(self.MPS[i].tensor,hsuper.shape[0])
            #energies, evecs = la.eigsh(hsuper1,k=1,which='SA', v0=v0)
            #energies, evecs = la.eigsh(hsuper,k=1,which='SA')
#            energies, evecs = mp.lanczos(hsuper,v0,num_iter,1)
#            energies, evecs = lanczos2(hsuper,v0,num_iter)
            energies, evecs = lanczos2(hsuper,self.MPS[i].tensor,num_iter)
#            print(energies-energies1)

           
           
            
            energy = min(energies)
            
            
#            new_m = np.reshape(evecs[:,min_idx],np.shape(self.MPS[i].tensor))
            
          #  print("SHAPE BEFORE DIAG",self.MPS[i].tensor.shape)
          #  print(self.MPS[i].tensor,i,"\n","before")
            #input()
            
            #self.MPS[i].tensor = np.reshape(evecs,np.shape(self.MPS[i].tensor))
            self.MPS[i].tensor = evecs
            
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
#            if i < nsites-2:
#                self.R[i] = tn.ncon([self.MPS_star[i+1].tensor,self.ham[i+1].tensor,self.MPS[i+1].tensor,self.R[i+1]],\
#                            [('n{}p'.format(i),-1,1),('n{}p'.format(i),'n{}'.format(i),-2,2),\
#                             ('n{}'.format(i),-3,3),(1,2,3)])
#            else:
#                self.R[i] = tn.ncon([self.MPS_star[i+1].tensor,self.ham[i+1].tensor,self.MPS[i+1].tensor],\
#                            [('n{}p'.format(i),-1),('n{}p'.format(i),'n{}'.format(i),-2),\
#                             ('n{}'.format(i),-3)])
                
            
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
                #hsuper = tn.ncon([self.L[i],self.ham[i].tensor],[(-2,'d',-4),(-1,-3,'d')])
                #hsuper = SuperBlockLO(self.L[i], self.ham[i].tensor, np.zeros(0))
                hsuper = SuperBlockLO1(self.L[i], self.ham[i].tensor, np.zeros(0))
            else:
                #hsuper = tn.ncon([self.L[i],self.ham[i].tensor,self.R[i]],[(-2,'d',-5),(-1,-4,'d','i'),(-3,'i',-6)])
                #hsuper = SuperBlockLO(self.L[i],self.ham[i].tensor, self.R[i])
                hsuper = SuperBlockLO1(self.L[i],self.ham[i].tensor, self.R[i])
                
            num_iter = min(np.prod(self.MPS[i].tensor.shape)-1,100)
            #v0 = np.reshape(self.MPS[i].tensor,hsuper.shape[0])
            #energies, evecs = la.eigsh(hsuper1,k=1,which='SA')
            #energies, evecs = la.eigsh(hsuper,k=1,which='SA', v0=v0)
            #energies, evecs = mp.lanczos(hsuper,v0,num_iter,1)
            #energies, evecs = lanczos2(hsuper,v0,num_iter)
            energies, evecs = lanczos2(hsuper,self.MPS[i].tensor,num_iter)
            self.MPS[i].tensor = evecs        
                        
            energy = min(energies)
            #min_idx=np.argmin(energies)
            
           
            #self.MPS[i].tensor = np.reshape(evecs,np.shape(self.MPS[i].tensor))
            
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

#            if i > 1:
#                self.L[i] = tn.ncon([self.L[i-1],self.MPS_star[i-1].tensor,self.ham[i-1].tensor,self.MPS[i-1].tensor],\
#                                [(1,2,3),('n{}p'.format(i),1,-1),('n{}p'.format(i),'n{}'.format(i),2,-2),\
#                                 ('n{}'.format(i),3,-3)])   
#            else:
#                self.L[i] = tn.ncon([self.MPS_star[i-1].tensor,self.ham[i-1].tensor,self.MPS[i-1].tensor],\
#                                [('n{}p'.format(i),-1),('n{}p'.format(i),'n{}'.format(i),-2),\
#                                 ('n{}'.format(i),-3)])   
                    
            
            
            
            
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

#class SuperBlock:
#    def __init__(self, L, W, R):
#        self.L = L
#        self.W = W
#        self.R = R
#        self.dtype = 'float64'
#        self.dim = self.W.shape[0]*self.L.shape[0]*self.R.shape[0]
#        self.shape = (self.dim,self.dim)
##        if L == False:
##            self.dim = self.W.shape[0]*self.R.shape[0]
##        elif R == False:
##            self.dim = self.W.shape[0]*self.L.shape[0]
##        else:
#        self.dim = self.W.shape[0]*self.L.shape[0]*self.R.shape[0]
#        #tn.ncon([self.ham[i].tensor,self.R[i]],[(-1,-3,'i{}'.format(i)),(-2,'i{}'.format(i),-4)])
#    def __mul__(self, y):
#         if len(y.shape) == 1:
#            mps_y = np.reshape(y ,(self.W.shape[0],self.L.shape[0],self.R.shape[0]))
#            new_v =  tn.ncon([self.L, self.W, self.R, mps_y], [(-2,'d','a'),(-1,'c','d','i'),(-3,'i','b'),('c','a','b')])
#            return np.reshape(new_v,self.dim)
#         else:
#             m = y.shape[1]
#             A = np.zeros(y.shape)
#             for i in range(m):
#                 mps_y = np.reshape(y[:,i] ,(self.W.shape[0],self.L.shape[0],self.R.shape[0]))
#                 new_v =  tn.ncon([self.L, self.W, self.R, mps_y], [(-2,'d','a'),(-1,'c','d','i'),(-3,'i','b'),('c','a','b')])
#                 A[:,i] = np.reshape(new_v,self.dim)
#             return A
import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy import linalg as la
import scipy as SP
from tensornetwork.tensor import Tensor


class LocalOperator:
    """A simple class to define auxiliary operators that support multiplication and addition,
        based on the rules for MPO defined in  PHYSICAL REVIEW B 95, 035129 (2017)
     
     
    """  
    def __init__(self, O, L):
        self.O = O 
        self.L = L
        
    def __mul__(self, op2):
        """Multiplication of two local operators or operator by a number(float)
        Parameters
        ----------
        op2 : float or LocalOperator
        operator to be multiplied     
        
        Returns
        -------
        new operator : LocalOperator
        """
        # operator by float/int multiplication
        if type(op2) == float or type(op2) == int: 
            coeff = abs(op2)**(1./self.L)
            sign = 0
            if abs(op2) > 1e-10:
             sign = op2/abs(op2)
             O = self.O
             O[0] *= sign
            for j in range(self.L):
                O[j]*= coeff
            return LocalOperator(O,self.L)
        
        # operator by operator multiplication
        res = []#self.O
        for j in range(self.L):
            if j==0 or j==self.L-1:
              res.append(np.matmul(self.O[j][:,:,0],op2.O[j][:,:,0]).reshape(op2.O[j].shape))
            else: 
              res.append(np.matmul(self.O[j][:,:,0,0],op2.O[j][:,:,0,0]).reshape(op2.O[j].shape))
              
        
        return LocalOperator(res, self.L)
    
    def __add__(self, op2):
        """ Add two operator and append in diagonal, secc. IV in PHYSICAL REVIEW B 95, 035129 (2017)
        """
        Ovec = []
        L = self.L
        for i in range(L):
            s1 = self.O[i].shape
            s2 = op2.O[i].shape
            if i == 0 or  i == L-1:
               m = s1[2] + s2[2]
               i2 = s1[2]
               M = np.zeros((2,2,m))
               for j in range(i2):
                   M[:,:,j] = self.O[i][:,:,j]
               for j in range(i2,m):
                   M[:,:,j] = op2.O[i][:,:,j-i2]
            else:
               n = s1[2] + s2[2]
               m = s1[3] + s2[3]
               i1 = s1[2] 
               i2 = s1[3]
               M = np.zeros((2,2,n,m))
               for j1 in range(i1):
                   for j2 in range(i2):
                     M[:,:,j1,j2] = self.O[i][:,:,j1,j2]
               for j1 in range(i1,n):
                   for j2 in range(i2,m):
                     M[:,:,j1,j2] = op2.O[i][:,:,j1-i1,j2-i2]           
                        
            Ovec.append(M)
        
        ReduceTensorSVD(Ovec,L)
        
        return LocalOperator(Ovec,self.L)
    
  
   

class QuantumOperator:
    """A simple Hamiltonian class"""  
    
    def __init__(self, L):
        self.LocOpVec = []
        self.L = L          

    def add(self, op2):
        self.LocOpVec.append(op2)

     
    def GetMPOSum(self):
        """
        Sum al term in the hamiltonia, we group similar size terms 
        """
        n = len(self.LocOpVec)
        step = 1
        LocOp =  self.LocOpVec
        while n > step:
            i = 0
            while i < n:
                if i+step < n:
                    LocOp[i] = LocOp[i] + LocOp[i+step]
                i += 2*step;
            step += step
            
#        LocOp = self.LocOpVec[0]
#        for i in range(1,n):
#            LocOp = LocOp + self.LocOpVec[i]
        return LocOp[0].O
 
    def GetMPOTensors(self, truncation_error = 1e-5):
         H = self.GetMPOSum()
         L = self.L
         hMPO = [tn.Node(H[0], axis_names = ["n_0p","n_0","i_0"] )]         
         
         for j in range(1, L-1):
             hMPO += [tn.Node(H[j], axis_names = ["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])]
         
         hMPO += [tn.Node(H[L-1], axis_names = ["n_{}p".format(L-1),"n_{}".format(L-1),"i_{}".format(L-2)] )]
         
         
         
         connected_edges2 = []
         for j in range(1,L):
            conn2 = hMPO[j-1]["i_{}".format(j-1)]^hMPO[j]["i_{}".format(j-1)]
            connected_edges2.append(conn2)
            #print(hMPO[j-1].tensor.shape)
        
         return hMPO,connected_edges2
     
    def ExpectatioValue(self, MPS):
        if len(MPS) != self.L:
           print("Operator and states of differents sizes")
           return
        ham, _ = self.GetMPOTensors()
        MPS_star = tn.replicate_nodes(MPS,conjugate=True)
        e = tn.ncon([MPS_star[0].tensor,ham[0].tensor,MPS[0].tensor],[('np',-1),('np','n',-2),('n',-3)])
        for i in range(1,self.L-1):
          e = tn.ncon([e,MPS_star[i].tensor, ham[i].tensor, MPS[i].tensor], [(1,2,3),('np',1,-1),('np','n',2,-2),('n',3,-3)])         
                    
        e = tn.ncon([e,MPS_star[self.L-1].tensor, ham[self.L-1].tensor, MPS[self.L-1].tensor], \
                                [(1,2,3),('np',1),('np','n',2),('n',3)])       
            
        return abs(e.max())
#def SVDarray(array, piva, truncation_error = 1e-5):
#  
#  
#  out = backend.svd(array, pivot_axis = piva,
#                    max_truncation_error= truncation_error)
#  tensors = [tn.Tensor(t, backend=backend) for t in out]
#  return tuple(tensors)
#def ReduceTensorsSDVnp(H,L):
#    truncation_error = 1e-5  
#    for j in range(1, L-1):
#        if j == 0:
#          piva = 2
#        else:
#          piva = 3
#        U,S,V,_ = tn.svd(tn.Tensor(H[j]), pivot_axis = piva, max_truncation_error = truncation_error)
#        H[j] = U
#        
##        for i in range(S.shape[0]):
##            V[i:]*= S[i]
#        V = S*V
#        if j==L-2:
#            H[j+1] = tn.ncon([V,H[j+1]],[(-3,'k'),(-1,-2,'k')])
#        else:
#            H[j+1] = tn.ncon([V,H[j+1]],[(-3,'k'),(-1,-2,'k',-4)])
#    for j in range(L-1,1,-1):
#        if j == L-1:
#           piva = 1
#        else:
#           piva = 1
#        Ht = tn.transpose(tn.Tensor(H[j]))
#        U,S,V,_ = tn.svd(Ht, pivot_axis = piva, max_truncation_error = truncation_error)
#        H[j] = U
#        #for i in range(S.shape[0]):
#        V = S*V
#        print(j)
#        print(H[j-1].shape,U.shape)
#        if j == 1:
#          H[j-1] = tn.ncon([H[j-1], U],[(-1,-2,'k'),('k',-3)])
#        else:
#          H[j-1] = tn.ncon([H[j-1], U],[(-1,-2,-3,'k'),('k',-4)])
#        
        
        
        
    
    

def ReduceTensorSVD(H,L):
        truncation_error = 1e-5
        hMPO = [tn.Node(H[0], axis_names = ["n_0p","n_0","i_0"] )]         
         
        for j in range(1, L-1):
               hMPO += [tn.Node(H[j], axis_names = ["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])]
         
        hMPO += [tn.Node(H[L-1], axis_names = ["n_{}p".format(L-1),"n_{}".format(L-1),"i_{}".format(L-2)] )]
         
         
             
         #simplify tensors using SVD        
        u, vh, trun_err = tn.split_node(hMPO[0], left_edges = [hMPO[0]["n_0p"],hMPO[0]["n_0"]],right_edges=[hMPO[0]["i_0"]], max_truncation_err = truncation_error, edge_name = "g")
        hMPO[0] = tn.Node(u.tensor, axis_names = ["n_0p","n_0","i_0"] )         
        for j in range(1,L-1):
            hMPO[j]= tn.Node(tn.ncon([vh.tensor,hMPO[j].tensor],[(-3,1),(-1,-2,1,-4)]),axis_names=["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])
            u, vh, trun_err = tn.split_node(hMPO[j], left_edges = [hMPO[j]["n_{}p".format(j)],hMPO[j]["n_{}".format(j)],hMPO[j]["i_{}".format(j-1)]], right_edges = [hMPO[j]["i_{}".format(j)]], max_truncation_err = truncation_error, edge_name  = "g")
            hMPO[j] = tn.Node(u.tensor,axis_names=["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])
        hMPO[L-1]= tn.Node(tn.ncon([vh.tensor,hMPO[L-1].tensor],[(-3,1),(-1,-2,1)]),axis_names=["n_{}p".format(L-1),"n_{}".format(L-1),"i_{}".format(L-2)])
         
         
        for i in range(L-1):
            if i == 0:
                ledges = [hMPO[i]["n_{}p".format(i)],hMPO[i]["n_{}".format(i)]]
            else:
                ledges = [hMPO[i]["n_{}p".format(i)],hMPO[i]["n_{}".format(i)],hMPO[i]["i_{}".format(i-1)]]
            
            redges = [hMPO[i]["i_{}".format(i)]]
            q,r,_ = tn.split_node(hMPO[i], left_edges=ledges, right_edges=redges, edge_name="ip_{}".format(i), max_truncation_err = truncation_error)
               
            hMPO[i].tensor = q.tensor
            if i==L-2:
                hMPO[i+1].tensor = tn.ncon([r.tensor,hMPO[i+1].tensor],[(-3,'k'),(-1,-2,'k')])
            else:
                hMPO[i+1].tensor = tn.ncon([r.tensor,hMPO[i+1].tensor],[(-3,'k'),(-1,-2,'k',-4)])
         
        for i in range(L-1,1,-1):
            if i == L-1:
                redges = [hMPO[i]["n_{}p".format(i)],hMPO[i]["n_{}".format(i)]]
            else:
                redges = [hMPO[i]["n_{}p".format(i)],hMPO[i]["n_{}".format(i)],hMPO[i]["i_{}".format(i)]]
                
            ledges = [hMPO[i]["i_{}".format(i-1)]]
            q,r,_ = tn.split_node(hMPO[i], left_edges=ledges, right_edges=redges, edge_name="ip_{}".format(i-1), max_truncation_err = truncation_error)
            
            if i == L-1:
                r.reorder_edges([r["n_{}p".format(i)],r["n_{}".format(i)],r["ip_{}".format(i-1)]])
                
            else:
                r.reorder_edges([r["n_{}p".format(i)],r["n_{}".format(i)],r["ip_{}".format(i-1)],r["i_{}".format(i)]])
            
           
            if i == 1:
                hMPO[i-1].tensor = tn.ncon([hMPO[i-1].tensor, q.tensor],[(-1,-2,'k'),('k',-3)])
                #print(hMPO[i-1].tensor.shape)
            else:
                hMPO[i-1].tensor = tn.ncon([hMPO[i-1].tensor, q.tensor],[(-1,-2,-3,'k'),('k',-4)])
            
            
            hMPO[i].tensor = r.tensor
#        Ovec = H
        for i in range(L):
          H[i] = hMPO[i].tensor
          #print(H[i].shape) 
        return H
        
def FermiOp(i, L, dagged):
    """
    A function that creates a local fermion operator in a way that
    anticommutation relations are fulfilled. 

    Parameters
    ----------
    i : int
      site index
    L : int
      system length
    dagged : int
      equal 1 for creation operator and egual 0 for annihilation    
    
    Returns
    -------
    mps : LocalOperator
        Local fermion operator

    """
    #Define local matrices
    one = np.array([1.,0,0,1.]).reshape(2,2,1,1)
    sg = np.array([1.,0,0,-1.]).reshape(2,2,1,1)
    cdag = np.array([0,1.,0,0]).reshape(2,2,1,1)
    c = np.array([0,0,1.,0]).reshape(2,2,1,1)
    
    
    #construct operators array to fulfill  anticommutation relations
    if dagged == 1:
       f = cdag
    else:
       f = c;
    O = []
    for j in range(L):
        if j < i :
          O.append(sg)
        elif j > i:
          O.append(one)
        elif j == i:
          O.append(f)
    
    O[0] = O[0].reshape(2,2,1)
    O[L-1] = O[L-1].reshape(2,2,1)
    return LocalOperator(O,L)
    
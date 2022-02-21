import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy import linalg as la
import scipy as SP


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
        # operator by float multiplication
        if(type(op2) == float): 
            coeff = abs(op2)**(1./self.L)
            sign = 0
            if abs(op2) > 1e-10:
             sign = op2/abs(op2)
             O = self.O
             O[0] *= sign
            return LocalOperator(O*coeff,self.L)
        
        # operator by operator multiplication
        res = self.O
        for j in range(self.L):
          res[j] = np.matmul(self.O[j],op2.O[j])
        return LocalOperator(res, self.L)
    
#    def __add__(self, op2):
#        """ Add two operator and append in diagonal, secc. IV in PHYSICAL REVIEW B 95, 035129 (2017)
#        """
#        Ovec = []
#        for j in range(self.L):
#            Ovec.append( [self.O[j],op2.O[j]])
#        return Ovec
    
  
   
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
    one = np.array([[1.0,0.0],[0,1.0]])
    sg = np.array([[1.0,0],[0,-1.0]])
    cdag = np.array([[0,1.0],[0,0]])
    c = np.array([[0,0],[1.0,0]])
    
    #construct operators array to fulfill  anticommutation relations
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
        elif j == i:
          O[j] = f
    return LocalOperator(O,L)

    
class QuantumOperator:
    """A simple Hamiltonian class"""  
    
    def __init__(self, L):
        self.Ovec = []
        self.L = L          

    def add(self, op2, coeff = 1):
         """ Add two operator and append in diagonal, secc. IV in PHYSICAL REVIEW B 95, 035129 (2017)
        """
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
         for j in range(self.L): #site index
             for n in range(2):
                 for n1 in range(2):
                     for i in range(nterms):
                        H[j,n,n1,i] = np.dot(states[n],np.matmul(self.Ovec[j,i],states[n1]))
                        #print("{},".format(n),"{},".format(n1),"{},".format(j),"{},".format(i),H[j,n,n1,i])
                        
         return H
 
    def GetMPOTensors(self, truncation_error = 1e-5):
         H = self.GetMPO()
         L = self.L
         nterms = self.Ovec.shape[1]
         
         hMPO = [tn.Node(H[0], axis_names = ["n_0p","n_0","i_0"] )]         
         
         for j in range(1, L-1):
             Oj = np.array([[[[0.]*nterms]*nterms]*2]*2)
             for n1 in range(2):
                 for n2 in range(2):
                  Oj[n1,n2] = np.diag(H[j,n1,n2])
             hMPO += [tn.Node(Oj, axis_names = ["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])]
         
         hMPO += [tn.Node(H[L-1], axis_names = ["n_{}p".format(L-1),"n_{}".format(L-1),"i_{}".format(L-2)] )]
         
         
             
         #simplify tensors using SVD        
         u, vh, trun_err = tn.split_node(hMPO[0], left_edges = [hMPO[0]["n_0p"],hMPO[0]["n_0"]],right_edges=[hMPO[0]["i_0"]], max_truncation_err = truncation_error, edge_name = "g")
         hMPO[0] = tn.Node(u.tensor, axis_names = ["n_0p","n_0","i_0"] )         
         for j in range(1,L-1):
            hMPO[j]= tn.Node(tn.ncon([vh.tensor,hMPO[j].tensor],[(-3,1),(-1,-2,1,-4)]),axis_names=["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])
            u, vh, trun_err = tn.split_node(hMPO[j], left_edges = [hMPO[j]["n_{}p".format(j)],hMPO[j]["n_{}".format(j)],hMPO[j]["i_{}".format(j-1)]], right_edges = [hMPO[j]["i_{}".format(j)]], max_truncation_err = truncation_error, edge_name  = "g")
            hMPO[j] = tn.Node(u.tensor,axis_names=["n_{}p".format(j),"n_{}".format(j),"i_{}".format(j-1),"i_{}".format(j)])
         hMPO[L-1]= tn.Node(tn.ncon([vh.tensor,hMPO[L-1].tensor],[(-3,1),(-1,-2,1)]),axis_names=["n_{}p".format(L-1),"n_{}".format(L-1),"i_{}".format(L-2)])
         
         connected_edges2 = []
         for j in range(1,L):
            conn2 = hMPO[j-1]["i_{}".format(j-1)]^hMPO[j]["i_{}".format(j-1)]
            connected_edges2.append(conn2)
        
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
            
         
        

    
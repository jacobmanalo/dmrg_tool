import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy.sparse import linalg as la

def init_wavefunction(n_sites,**kwargs):
    """
    A function that initializes the coefficients of a wavefunction for L sites (from 0 to L-1) and arranges
    them in a tensor of dimension n_0 x n_1 x ... x n_L for L sites. SVD
    is applied to this tensor iteratively to obtain the matrix product state.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    kwargs
    ----------
    conserve_n : boolean
        True for conservation of number of particles.
    
    num_e : int
        Number of electrons

    Returns
    -------
    mps : tensornetwork
        Matrix Product State.

    """
    conserve_n=kwargs.get('conserve_n',False)
    
    psi = np.zeros(tuple([2]*n_sites))
    
    norm= 0.
    if conserve_n == True:
        num_e = kwargs.get('num_e')
        single_tuple = list([0]*n_sites)
        for i in range(num_e):
            single_tuple[i] = 1
        for tup in set(itt.permutations(single_tuple,n_sites)):
            psi[tup] = np.random.uniform(-1,1)
            norm += np.abs(psi[tup])**2 
        norm = np.sqrt(norm)
    else:
        psi = np.random.random_sample(tuple([2]*n_sites))
        norm = np.linalg.norm(psi)
    psi = tn.Node(psi, axis_names=["n_{}".format(i) for i in range(n_sites)])
    
    #THIS PART RIGHT NORMALIZES THE MPS
    u = {}
    s = {}
    v = {}
    
    u[n_sites] = psi
    
    for i in range(n_sites-1,0,-1):
        l_edges=[u[i+1]["n_{}".format(k)] for k in range(i)]
        r_edges=[u[i+1]["n_{}".format(i)]]
        if i < n_sites-1:
            r_edges+=[u[i+1]["i_{}".format(i)]]
        #print('hello',i)
        u[i],s[i],v[i],_ = tn.split_node_full_svd(u[i+1],left_edges=l_edges, \
                                                  right_edges=r_edges,left_edge_name="d_{}".format(i-1),right_edge_name="i_{}".format(i-1))
        
        if i == n_sites-1:
            reord_edges=[v[i]["n_{}".format(i)],v[i]["i_{}".format(i-1)]]
        else:
            reord_edges=[v[i]["n_{}".format(i)],v[i]["i_{}".format(i-1)],v[i]["i_{}".format(i)]]
        v[i].reorder_edges(reord_edges)
        
        cont_edges = ["n_{}".format(k) for k in range(i)]+["i_{}".format(i-1)]
        u[i]=tn.contract(u[i]["d_{}".format(i-1)],axis_names=cont_edges)
    
    mps = [u[1]]
    for i in range(1,n_sites):
        mps+= [v[i]]
    

    
    return mps

import numpy as np
import tensornetwork as tn
import itertools as itt
from scipy.sparse import linalg as la
import time

def block(*dimensions):
    '''Construct a new matrix for the MPS with random numbers from 0 to 1'''
    size = tuple([x for x in dimensions])
    return np.random.random_sample(size)

def init_wavefunction(n_sites,bond_dim,**kwargs):
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
# =============================================================================
#     t1 = time.time()
#     mps = [ \
#         tn.Node( block(2, bond_dim),axis_names=["n_0","i_0"] )] + \
#         [tn.Node( block(2, bond_dim, bond_dim),axis_names=["n_{}".format(l),"i_{}".format(l-1),"i_{}".format(l)]) for l in range(1,n_sites-1)] + \
#         [tn.Node( block(2, bond_dim),axis_names=["n_{}".format(n_sites-1),"i_{}".format(n_sites-2)] ) \
#         ]
# 
#     
#     #Right Canonicalize
#     for i in range(n_sites-1,0,-1):
#         if i == n_sites-1:
#             redges = [mps[i]["n_{}".format(i)]]
#         else:
#             redges = [mps[i]["i_{}".format(i)],mps[i]["n_{}".format(i)]]
#             
#         ledges = [mps[i]["i_{}".format(i-1)]]
#         u,s,v,_ = tn.split_node_full_svd(mps[i], left_edges=ledges, right_edges=redges,\
#                                          left_edge_name="d_{}".format(i-1), right_edge_name="i_{}".format(i-1),\
#                                              max_truncation_err=1e-5)
#         
#         if i == n_sites-1:
#             reord_edges=[v["n_{}".format(i)],v["i_{}".format(i-1)]]
#         else:
#             reord_edges=[v["n_{}".format(i)],v["i_{}".format(i-1)],v["i_{}".format(i)]]
#         v.reorder_edges(reord_edges)
#         
#         if i == 1:
#             mps[i-1].tensor = tn.ncon([mps[i-1].tensor, u.tensor, s.tensor],[(-1,'k'),('k','l'),('l',-2)])
#         else:
#             mps[i-1].tensor = tn.ncon([mps[i-1].tensor, u.tensor, s.tensor],[(-1,-2,'k'),('k','l'),('l',-3)])
#         mps[i].tensor = v.tensor
#     #connect edges to build mps
#     connected_edges=[]
#     conn=mps[0]["i_0"]^mps[1]["i_0"]
#     connected_edges.append(conn)
#     for k in range(1,n_sites-1):
#         conn=mps[k]["i_{}".format(k)]^mps[k+1]["i_{}".format(k)]
#         connected_edges.append(conn)
# 
#     mod = np.linalg.norm(mps[0].tensor)
#     mps[0].tensor /= mod
#     t2 = time.time()
#     print("MPS CONSTRUCTION TIME=",t2-t1)
#     return mps
# =============================================================================
        
    #NOW FOR SVD
    
    t1 = time.time()
    mps = [ \
        tn.Node( block(2, bond_dim),axis_names=["n_0","i_0"] )] + \
        [tn.Node( block(2, bond_dim, bond_dim),axis_names=["n_{}".format(l),"i_{}".format(l-1),"i_{}".format(l)]) for l in range(1,n_sites-1)] + \
        [tn.Node( block(2, bond_dim),axis_names=["n_{}".format(n_sites-1),"i_{}".format(n_sites-2)] ) \
        ]
    for i in range(n_sites-1,0,-1):
        #DO RQ RIGHT NORMALIZATION ON NEWFOUND M
        if i == n_sites-1:
            redges = [mps[i]["n_{}".format(i)]]
        else:
            redges = [mps[i]["i_{}".format(i)],mps[i]["n_{}".format(i)]]
            
        ledges = [mps[i]["i_{}".format(i-1)]]
        r,q = tn.split_node_rq(mps[i], left_edges=ledges, right_edges=redges, edge_name="ip_{}".format(i-1))
        
        if i == n_sites-1:
            q.reorder_edges([q["n_{}".format(i)],q["ip_{}".format(i-1)]])
            
        else:
            q.reorder_edges([q["n_{}".format(i)],q["ip_{}".format(i-1)],q["i_{}".format(i)]])
        
        if i == 1:
            mps[i-1].tensor = tn.ncon([mps[i-1].tensor, r.tensor],[(-1,'k'),('k',-2)])
        else:
            mps[i-1].tensor = tn.ncon([mps[i-1].tensor, r.tensor],[(-1,-2,'k'),('k',-3)])
        mps[i].tensor = q.tensor
    t2 = time.time()
    print("MPS CONSTRUCTION TIME=",t2-t1)
    return mps
# =============================================================================
#     conserve_n=kwargs.get('conserve_n',False)
#     
#     psi = np.zeros(tuple([2]*n_sites))
#     
#     norm= 0.
#     t1=time.time()
#     if conserve_n == True:
#         num_e = kwargs.get('num_e')
#         single_tuple = list([0]*n_sites)
#         for i in range(num_e):
#             single_tuple[i] = 1
#         for tup in set(itt.permutations(single_tuple,n_sites)):
#             psi[tup] = np.random.uniform(-1,1)
#             norm += np.abs(psi[tup])**2 
#         norm = np.sqrt(norm)
#     else:
#         psi = np.random.random_sample(tuple([2]*n_sites))
#         norm = np.linalg.norm(psi)
#     t2=time.time()
#     print("Time=",t2-t1)
#     psi = tn.Node(psi, axis_names=["n_{}".format(i) for i in range(n_sites)])
#     
#     #THIS PART RIGHT NORMALIZES THE MPS
#     u = {}
#     s = {}
#     v = {}
#     
#     u[n_sites] = psi
#     
#     for i in range(n_sites-1,0,-1):
#         l_edges=[u[i+1]["n_{}".format(k)] for k in range(i)]
#         r_edges=[u[i+1]["n_{}".format(i)]]
#         if i < n_sites-1:
#             r_edges+=[u[i+1]["i_{}".format(i)]]
#         #print('hello',i)
#         u[i],s[i],v[i],_ = tn.split_node_full_svd(u[i+1],left_edges=l_edges, \
#                                                   right_edges=r_edges,left_edge_name="d_{}".format(i-1),\
#                                                       right_edge_name="i_{}".format(i-1),\
#                                                           max_singular_values=bond_dim)
#         
#         if i == n_sites-1:
#             reord_edges=[v[i]["n_{}".format(i)],v[i]["i_{}".format(i-1)]]
#         else:
#             reord_edges=[v[i]["n_{}".format(i)],v[i]["i_{}".format(i-1)],v[i]["i_{}".format(i)]]
#         v[i].reorder_edges(reord_edges)
#         
#         cont_edges = ["n_{}".format(k) for k in range(i)]+["i_{}".format(i-1)]
#         u[i]=tn.contract(u[i]["d_{}".format(i-1)],axis_names=cont_edges)
#     
#     mps = [u[1]]
#     for i in range(1,n_sites):
#         mps+= [v[i]]
#     
# 
#     
#     return mps
# =============================================================================

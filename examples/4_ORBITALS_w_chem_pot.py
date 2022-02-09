

import numpy as np
import tensornetwork as tn
import itertools as itt
#from scipy.sparse import linalg as la
#import matrixproductstates as mp
import scipy as SP
import pymps as mp

def kdelta(i,j):
    """
    Parameters
    ----------
    i : int
        State index i.
    j : int
        State index j.

    Returns
    -------
    int
        Kronecker_Delta(i,j).
    """
    return int(i==j)

# Construct MPS for 4 sites

bond_dim=2

n_sites = 4
mps = mp.init_wavefunction(n_sites, bond_dim=2)

#

# Creating the things we need for the Hamiltonian
#============================
from scipy import linalg as LA
from scipy import special as sp
#FUNDAMENTAL CONSTANTS
hbar = 6.582119569e-16 #eV*s
m_e = 0.51099895000e6 #eV/c^2
m_eff = 0.067
c_light = 299792458 #m/s
bohr =  5.7883818060e-2 #meV/T
lande = 0.52

rydberg=5.93 #meV
#=======
def vmatrix(n1l,m1l,n2l,m2l,n2r,m2r,n1r,m1r):
    """
    Computes Coulomb matrix elements for a parabolic quantum dot. Analytic formula
    derived by Marek Korkusinski can be found here: https://mysite.science.uottawa.ca/phawrylak/member_pages/korkusinski/thesis/thesis.html.
    
    Computes <n1l m1l, n2l m2l|V|n2r m2r, n1r m1r>

    Parameters
    ----------
    n1l : int
        index of Landau level of electron 1 for initial state.
    m1l : int
        index of quasidegenerate orbital of electron 1 for initial state.
    n2l : int
        index of Landau level of electron 2 for initial state.
    m2l : int
        index of quasidegenerate orbital of electron 1 for initial state.
    n2r : int
        index of Landau level of electron 2 for final state.
    m2r : int
        index of quasidegenerate orbital of electron 2 for final state.
    n1r : int
        index of Landau level of electron 1 for final state.
    m1r : int
        index of quasidegenerate orbital of electron 1 for final state.

    Returns
    -------
    vmatel : float
        <n1l m1l, n2l m2l|V|n2r m2r, n1r m1r>

    """
    delta_rl_rr = kdelta((m1l+m2l)-(n1l+n2l),(m1r+m2r)-(n1r+n2r))
    fac_denom = np.sqrt(sp.factorial(n1l)*sp.factorial(m1l)*\
                        sp.factorial(n1r)*sp.factorial(m1r)*\
                            sp.factorial(n2l)*sp.factorial(m2l)*\
                                sp.factorial(n2r)*sp.factorial(m2r))
    phase = (-1)**(n2l+m2l+n2r+m2r)
    
    
    total = 0.
    for p1 in range(min(n1l,n1r)+1):
        for p2 in range(min(m1l,m1r)+1):
            for p3 in range(min(n2l,n2r)+1):
                for p4 in range(min(m2l,m2r)+1):
                    power = n1l + n2l + m1r + m2r - (p1+p2+p3+p4)
                    
                    p1fac=sp.factorial(p1)*sp.binom(n1l,p1)*sp.binom(n1r,p1)
                    p2fac=sp.factorial(p2)*sp.binom(m1l,p2)*sp.binom(m1r,p2)
                    p3fac=sp.factorial(p3)*sp.binom(n2l,p3)*sp.binom(n2r,p3)
                    p4fac=sp.factorial(p4)*sp.binom(m2l,p4)*sp.binom(m2r,p4)
                    gammafac=(-0.5)**power*sp.gamma(power+0.5)
                    total+=p1fac*p2fac*p3fac*p4fac*gammafac
    
    vmatel = delta_rl_rr*phase*total/(fac_denom*np.sqrt(np.pi))
    return vmatel

def sp_energies(n,m,B,spin,hbar,m_e,m_eff,c_light,bohr,lande,rydberg,omega_0,omega_c):
    """
    

    Parameters
    ----------
    n : int
        Landau level.
    m : int
        Sub orbital in Landau level n.
    B : float
        Magnetic field in T.
    spin : float
        Spin of electron.
    hbar : float
        Planck's constant.
    m_e : flaot
        Mass of electron.
    m_eff : float
        Effective mass of electron.
    c_light : float
        Speed of light.
    bohr : float
        Bohr radius.
    lande : float
        g-factor.
    rydberg : float
        Rydberg energy.
    omega_0 : float
        Characteristic frequency of harmonic oscillator.
    omega_c : float
        Cyclotron frequency.

    Returns
    -------
    energy : float
        Single particle energy.

    """
    omega_p=np.sqrt(omega_0**2+0.25*omega_c**2)+0.5*omega_c
    omega_m=np.sqrt(omega_0**2+0.25*omega_c**2)-0.5*omega_c
    energy = omega_p*(n+0.5)+omega_m*(m+0.5)-lande*bohr*B*spin
    return energy



B=10

omega_0 = 3.31 #meV
omega_c = 1e3*hbar*B*c_light**2/(m_e*m_eff)

OMEGA_H = np.sqrt(omega_0**2+0.25*omega_c**2)/rydberg
E_0=np.sqrt(np.pi*OMEGA_H)*rydberg

epsilon = []
for m in range(4):
    epsilon.append(sp_energies(1,m,B,0.5,hbar,m_e,m_eff,c_light,bohr,lande,rydberg,omega_0,omega_c))

    
v12=(vmatrix(1, 0, 1, 1, 1, 1, 1, 0)-vmatrix(1, 0, 1, 1, 1, 0, 1, 1))*E_0
v13=(vmatrix(1, 0, 1, 2, 1, 2, 1, 0)-vmatrix(1, 0, 1, 2, 1, 0, 1, 2))*E_0
v14=(vmatrix(1, 0, 1, 3, 1, 3, 1, 0)-vmatrix(1, 0, 1, 3, 1, 0, 1, 3))*E_0

v23=(vmatrix(1, 1, 1, 2, 1, 2, 1, 1)-vmatrix(1, 1, 1, 2, 1, 1, 1, 2))*E_0
v24=(vmatrix(1, 1, 1, 3, 1, 3, 1, 1)-vmatrix(1, 1, 1, 3, 1, 1, 1, 3))*E_0
v34=(vmatrix(1, 2, 1, 3, 1, 3, 1, 2)-vmatrix(1, 2, 1, 3, 1, 2, 1, 3))*E_0

w=(vmatrix(1, 0, 1, 3, 1, 1, 1, 2)-vmatrix(1, 0, 1, 3, 1, 2, 1, 1))*E_0
for m in range(4):
    epsilon.append(sp_energies(1,m,B,0.5,hbar,m_e,m_eff,c_light,bohr,lande,rydberg,omega_0,omega_c))


# Create H MPO

G0 = np.array([[[0.]*4]*2]*2)
G1 = np.array([[[[0.]*6]*4]*2]*2)
G2 = np.array([[[[0.]*4]*6]*2]*2)
G3 = np.array([[[0.]*4]*2]*2)

for n0p in range(2):
    for n0 in range(2):
        G0[n0p,n0]=np.array([n0*kdelta(n0p,n0),kdelta(n0p,n0),kdelta(n0p-1,n0),kdelta(n0p,n0-1)])
        
for n1p in range(2):
    for n1 in range(2):
        G1[n1p,n1]=np.array([[v14*kdelta(n1p,n1),0,epsilon[0]*kdelta(n1p,n1)+v12*n1*kdelta(n1p,n1),v13*kdelta(n1p,n1),0,0]\
                             ,[epsilon[3]*kdelta(n1p,n1)+v24*n1*kdelta(n1p,n1),v34*kdelta(n1p,n1),epsilon[1]*n1*kdelta(n1p,n1),epsilon[2]*\
                               kdelta(n1p,n1)+v23*n1*kdelta(n1p,n1),0,0],\
                                 [0,0,0,0,-w*kdelta(n1p,n1-1),0],[0,0,0,0,0,-w*kdelta(n1p-1,n1)]])

for n2p in range(2):
    for n2 in range(2):
        G2[n2p,n2]=np.array([[kdelta(n2p,n2),0,0,0],[n2*kdelta(n2p,n2),0,0,0],[0,kdelta(n2p,n2),0,0],[0,n2*kdelta(n2p,n2),0,0],\
                             [0,0,kdelta(n2p,n2-1),0],[0,0,0,kdelta(n2p-1,n2)]])

for n3p in range(2):
    for n3 in range(2):
        G3[n3p,n3]=np.array([n3*kdelta(n3p,n3),kdelta(n3p,n3),kdelta(n3p-1,n3),kdelta(n3p,n3-1)])

#Create the chemical potential MPO

W0 = np.array([[[0.]*2]*2]*2)
W1 = np.array([[[[0.]*2]*2]*2]*2)
W2 = np.array([[[[0.]*2]*2]*2]*2)
W3 = np.array([[[0.]*2]*2]*2)

chem_pot=-35
for n0p in range(2):
    for n0 in range(2):
        W0[n0p,n0]=np.array([n0*kdelta(n0p,n0),kdelta(n0p,n0)])*chem_pot
        
for n1p in range(2):
    for n1 in range(2):
        W1[n1p,n1]=np.array([[kdelta(n1p,n1),0.],[n1*kdelta(n1p,n1),kdelta(n1p,n1)]])

for n2p in range(2):
    for n2 in range(2):
        W2[n2p,n2]=np.array([[kdelta(n2p,n2),0.],[n2*kdelta(n2p,n2),kdelta(n2p,n2)]])

for n3p in range(2):
    for n3 in range(2):
        W3[n3p,n3]=np.array([kdelta(n3p,n3),n3*kdelta(n3p,n3)])



O0 = np.array([[[0.]*6]*2]*2)
O1 = np.array([[[[0.]*8]*6]*2]*2)
O2 = np.array([[[[0.]*6]*8]*2]*2)
O3 = np.array([[[0.]*6]*2]*2)
for n0p in range(2):
    for n0 in range(2):
        O0[n0p,n0]=np.hstack((G0[n0p,n0],W0[n0p,n0]))
for n1p in range(2):
    for n1 in range(2):
        O1[n1p,n1]=SP.linalg.block_diag(G1[n1p,n1],W1[n1p,n1])
        
for n2p in range(2):
    for n2 in range(2):
        O2[n2p,n2]=SP.linalg.block_diag(G2[n2p,n2],W2[n2p,n2])
for n3p in range(2):
    for n3 in range(2):
        O3[n3p,n3]=np.hstack((G3[n3p,n3],W3[n3p,n3]))

#Creating MPO as a tensornetwork

hmpo = [ \
        tn.Node(O0,axis_names=["n_0p","n_0","i_0"] )] + \
            [tn.Node(O1,axis_names=["n_1p","n_1","i_0","i_1"])] + \
                [tn.Node(O2,axis_names=["n_2p","n_2","i_1","i_2"])] + \
                 [tn.Node(O3,axis_names=["n_3p","n_3","i_2"])]

# Connect edges to build MPO
connected_edges2=[]
conn2=hmpo[0]["i_0"]^hmpo[1]["i_0"]
connected_edges2.append(conn2)
conn2=hmpo[1]["i_1"]^hmpo[2]["i_1"]
connected_edges2.append(conn2)
conn2=hmpo[2]["i_2"]^hmpo[3]["i_2"]
connected_edges2.append(conn2)


#Run DMRG algorithm


energy,energies,MPS=mp.DMRG(4,hmpo,10,mps)



MPS[0]["i_0"]^MPS[1]["i_0"]
MPS[1]["i_1"]^MPS[2]["i_1"]
MPS[2]["i_2"]^MPS[3]["i_2"]
test=MPS[0]@MPS[1]@MPS[2]@MPS[3]

np.transpose(np.where(np.abs(test.tensor)>=1e-10))[0]
number_e=np.count_nonzero(np.transpose(np.where(np.abs(test.tensor)>=1e-10))[0])

print('Corrected Energy = {}'.format(energy-number_e*chem_pot))



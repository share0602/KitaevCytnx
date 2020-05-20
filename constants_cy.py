import numpy as np
from scipy import linalg

from setting import *
import cytnx
from cytnx import cytnx_extension as cyx

EPS = 1.E-32

def physical_dimension_to_spin(d):
    return str((d - 1) // 2) if (d - 1) % 2 == 0 else str(d - 1) + '/2'

def spin_to_physical_dimension(spin):
    s = int(spin.split('/')[0]) / int(spin.split('/')[1]) if ('/' in spin) else int(spin)
    return int(2 * s + 1)

def Get_spin_operators(spin):
    #Returns tuple of 3 spin operators and a unit matrix for given value of spin.
    s = int(spin.split('/')[0]) / int(spin.split('/')[1]) if ('/' in spin) else int(spin)
    d = int(2 * s + 1)
   
    return cytnx.physics.spin(s,'x')  , cytnx.physics.spin(s,'y'), cytnx.physics.spin(s,'z'), cytnx.eye(d).astype(cytnx.Type.ComplexDouble)

def Create_loop_gas_operator(spin):
    """Returns loop gas (LG) operator Q_LG for spin=1/2 or spin=1 Kitaev model."""

    tau_tensor = cytnx.zeros((2, 2, 2), dtype=cytnx.Type.ComplexDouble)  # tau_tensor_{i j k}

    if '/' in spin:
        tau_tensor[0,0,0] = - 1j
    else:
        tau_tensor[0,0,0] = 1

    tau_tensor[0,1,1] = tau_tensor[1,0,1] = tau_tensor[1,1,0] = 1

    sx, sy, sz, one = Get_spin_operators(spin)
    d = one.shape()[0]

    Q_LG = cytnx.zeros((d, d, 2, 2, 2), dtype=cytnx.Type.ComplexDouble)  # Q_LG_{s s' i j k}

    u_gamma = None
    
    if '/' in spin:
        u_gamma = list(map(lambda x: -1j*cytnx.linalg.ExpM(1j * np.pi * x), (sx, sy, sz)))
    else:
        u_gamma = list(map(lambda x: cytnx.linalg.ExpM(1j * np.pi * x), (sx, sy, sz)))

    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = cytnx.eye(d)
                if i == 0:
                    temp = cytnx.linalg.Matmul(temp, u_gamma[0])
                if j == 0:
                    temp = cytnx.linalg.Matmul(temp, u_gamma[1])
                if k == 0:
                    temp = cytnx.linalg.Matmul(temp, u_gamma[2])

                for s in range(d):
                    for sp in range(d):
                        Q_LG[s,sp,i,j,k] = tau_tensor[i,j,k] * temp[s,sp]

    return Q_LG


def Construct_kitaev_hamiltonian(spin, h=0.,k=1.):
    """Returns list of two-site Hamiltonian in [x, y, z]-direction for Kitaev model"""
    
    sx, sy, sz, one = Get_spin_operators(spin)
    hamiltonian = [-k*cytnx.linalg.Kron(sx,sx), -k*cytnx.linalg.Kron(sy,sy), -k*cytnx.linalg.Kron(sz,sz)]
    
    return hamiltonian

def Construct_heisenberg_hamiltonian(spin, h=0., k=1.):
    """Returns list of two-site Hamiltonian in [x, y, z]-direction for Heisenberg model"""
    
    sx, sy, sz, one = Get_spin_operators(spin)
    hamiltonian = [k*cytnx.linalg.Kron(sx,sx)+k*cytnx.linalg.Kron(sy,sy)+k*cytnx.linalg.Kron(sz,sz)]*3
    return hamiltonian

def Construct_ising_hamiltonian(spin, h=3.1, k=1.):
    """Returns list of two-site Hamiltonian in [x, y, z]-direction for Ising model"""
    
    sx, sy, sz, one = Get_spin_operators(spin)
    hamiltonian = - k *cytnx.linalg.Kron(sx, sx) - h * (cytnx.linalg.Kron(sy, one) + cytnx.linalg.Kron(one, sy)) / 2
    return [hamiltonian]*3

def become_LGstate(ten_a, Q_LG):
    ten_a.set_labels([-1,2,4,6])
    Q_LG.set_labels([0,-1,1,3,5]) 
    ten_a = cyx.Contract(ten_a, Q_LG)
    ten_a.permute_(np.arange(7),by_label=True)
    #ten_a.print_diagram()
    ten_a.reshape_(-1,2,2,2)
    return ten_a

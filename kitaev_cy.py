import time
import copy
import numpy as np
from scipy import linalg
import constants_cy
from tqdm import tqdm
import ctmrg_cy # Used for create weight, impurity, cns, tms, and calculate energy
import ite_cy  # Used for ITE (simple update)
from args import args

##### Import cytnx module
from setting import *
import cytnx
from cytnx import cytnx_extension as cyx

start = time.time()

spin = args.spin
D = args.D; m = args.chi
model = args.model
h = args.hz
tau = 0.01
refresh = 100; ITEsteps = 5
d = constants_cy.spin_to_physical_dimension(spin)
sx, sy, sz, _ = constants_cy.Get_spin_operators(spin)
if model == 'Kitaev':
    Construct_hamiltonian = constants_cy.Construct_kitaev_hamiltonian
elif model == 'Heisenberg':
    Construct_hamiltonian = constants_cy.Construct_heisenberg_hamiltonian
elif model == 'TFIM':
    Construct_hamiltonian = constants_cy.Construct_ising_hamiltonian

##### Prepare initial magnetized state
ten_a = cytnx.zeros((d, 1, 1, 1))
ten_a = ten_a.astype(cytnx.Type.ComplexDouble)
ten_b = cytnx.zeros((d, 1, 1, 1))
ten_b = ten_b.astype(cytnx.Type.ComplexDouble)
w, v = cytnx.linalg.Eigh(-1*(sx + sy + sz))
state = v[:, 0] # eigenvector with the smallest eigenvalues
ten_a[:,0,0,0] = state;
ten_b[:,0,0,0] = state;

####### Prepare initial LG state    
Q_LG = constants_cy.Create_loop_gas_operator(spin)
Q_LG = cyx.CyTensor(Q_LG, 0); 
## ten_a, ten_b
ten_a = cyx.CyTensor(ten_a, 0); 
ten_b = cyx.CyTensor(ten_b, 0);
ten_a = constants_cy.become_LGstate(ten_a, Q_LG)
ten_b = constants_cy.become_LGstate(ten_b, Q_LG)
# ten_a = np.random.randn(d,2,2,2)
# ten_a = cytnx.from_numpy(ten_a)
# ten_a = ten_a.astype(cytnx.Type.ComplexDouble)
# ten_b = ten_a.clone()
# ten_a = cyx.CyTensor(ten_a, 0);
# ten_b = cyx.CyTensor(ten_b, 0);
# ten_b.print_diagram()

## lambda x,y,z
lx = cyx.CyTensor([cyx.Bond(2),cyx.Bond(2)],rowrank = 0, is_diag=True)
lx.put_block(cytnx.ones(2));
l_three_dir = [lx, lx.clone(), lx.clone()]
# print(l_three_dir)

## construct ITE gate
H = Construct_hamiltonian(spin,h)
u_gates = [cytnx.linalg.ExpH(-tau*hamiltonian).reshape(d, d, d, d) for hamiltonian in H]
u_gates = [cyx.CyTensor(u,0) for u in u_gates ]

print('spin:%s d:%d D:%d m:%d' %(spin, d, D, m))

## Calculate LG energy
weight, weight_imp, cns, tms = ctmrg_cy.create_w_imp_cns_tms(H,ten_a, ten_b, l_three_dir)
weight.reshape_(2**2, 2**2, 2**2, 2**2)
weight_imp.reshape_(2**2, 2**2, 2**2, 2**2)
energy = ctmrg_cy.ctmrg_coarse_graining(24, weight, weight_imp, cns, tms)
#exit(1)
## Do ITE and calculate energy
for i in range(ITEsteps):
    for _ in tqdm(range(refresh)):
        ten_a, ten_b, l_three_dir = ite_cy.simple_update(ten_a, ten_b, l_three_dir, D, u_gates)
    print(l_three_dir[0].get_block().numpy())
    print(l_three_dir[1].get_block().numpy())
    print(l_three_dir[2].get_block().numpy())
    weight, weight_imp, cns, tms = ctmrg_cy.create_w_imp_cns_tms(H,ten_a, ten_b, l_three_dir)
    weight.reshape_(D**2, D**2, D**2, D**2)
    weight_imp.reshape_(D**2, D**2, D**2, D**2)
    energy = ctmrg_cy.ctmrg_coarse_graining(m, weight, weight_imp, cns, tms)

end = time.time()
print(end-start)
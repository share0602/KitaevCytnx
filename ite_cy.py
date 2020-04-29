import numpy as np
from scipy import linalg

import sys
sys.path.append("/usr/local/")
import cytnx
from cytnx import cytnx_extension as cyx

import constants_cy
def sort_label(A):
    label = A.labels()
    perm = [label.index(i) for i in range(len(label))]
    A.permute_(perm)
    if A.labels() == list(np.sort(label)):
        return A
    else:
        raise ValueError('A.labels() is not equal list(np.sort(label))')

def Tensor_rotate(ten):
    """Returns tensor with virtual indices rotated anti-clock wise."""
    ten1 = ten.clone()
    ten1.permute_([0, 2, 3, 1])
    return ten1

def Lambdas_rotate(lam):
    """Returns lambdas rotated anti-clock wise."""

    return lam[1:] + [lam[0]]

def simple_update(ten_a, ten_b, l_three_dir, D, u_gates):
    for i in range(3):
        u_gates[i].set_labels([-1,-5,0,3])
        ## first set_labels, which will be used for contraction later
        ten_a.set_labels([-1,-2,-3,-4]);
        ten_b.set_labels([-5,-6,-7,-8]);
        lx = l_three_dir[0].clone()
        lx.set_labels([-2,-6])
        
        ## those will contract with ten_a later
        ly_a = l_three_dir[1].clone()
        ly_a.set_labels([1,-3])
        lz_a = l_three_dir[2].clone()
        lz_a.set_labels([2,-4])
        
        ## those will contract with ten_b later
        ly_b = l_three_dir[1].clone()
        ly_b.set_labels([4,-7])
        lz_b = l_three_dir[2].clone()
        lz_b.set_labels([5,-8])

        # pair contraction + apply gate
        ten_axyz = cyx.Contract(cyx.Contract(cyx.Contract(ten_a, lx), ly_a),lz_a)
        ten_byz = cyx.Contract(cyx.Contract(ten_b, ly_b), lz_b)
        pair_ten = cyx.Contract(ten_axyz, ten_byz)
        apply_ten = cyx.Contract(pair_ten, u_gates[i])
        #apply_ten.permute_([4,0,1,5,2,3]) # Not trivial, please use print_diagram()
        apply_ten = sort_label(apply_ten)
        apply_ten.set_Rowrank(3)
        # apply_ten.print_diagram()

        ## SVD truncate
        d = ten_a.shape()[0]; #print(d)
        dim_new = min(2*2*d, D)
        lx,ten_a,ten_b = cyx.xlinalg.Svd_truncate(apply_ten, dim_new)
        ten_a.set_labels([0,-1,-2,1])
        ten_b.set_labels([1,0,-1,-2])
        ly_a_inv = 1./ly_a
        lz_a_inv = 1./lz_a
        ly_a_inv.set_labels([2,-1])
        lz_a_inv.set_labels([3,-2])


        ten_a = cyx.Contract(cyx.Contract(ten_a, ly_a_inv), lz_a_inv)
        ten_a.set_Rowrank(0)
        ly_b_inv = 1./ly_b
        lz_b_inv = 1./lz_b
        ly_b_inv.set_labels([2,-1])
        lz_b_inv.set_labels([3,-2])
        ten_b = cyx.Contract(cyx.Contract(ten_b, ly_b_inv), lz_b_inv)
        #ten_b.permute_([1,0,2,3]) ## not so trivial, please use print_diagram()
        ten_b = sort_label(ten_b)
        ten_b.set_Rowrank(0)

        Norm = sum(lx.get_block().numpy())
        l_three_dir[0] = lx/Norm
        l_three_dir = Lambdas_rotate(l_three_dir)
        ten_a = Tensor_rotate(ten_a)
        ten_b = Tensor_rotate(ten_b)

    return ten_a, ten_b, l_three_dir

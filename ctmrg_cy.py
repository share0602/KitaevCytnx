import copy
import numpy as np
from scipy import linalg
import constants_cy
import ite_cy
#from ncon import ncon
##### Import cytnx module
from setting import *
import cytnx
from cytnx import cytnx_extension as cyx

def sort_label(A):
    label = A.labels()
    perm = [label.index(i) for i in range(len(label))]
    A.permute_(perm)
    if A.labels() == list(np.sort(label)):
        return A
    else:
        raise ValueError('A.labels() is not equal list(np.sort(label))')

def create_w_imp_cns_tms(ham,ten_a, ten_b, l_three_dir):
    'Create weight, impurity, cns, and tms with the imput ten_a, ten_b, l_three_dir'
    D = l_three_dir[0].shape()[0]
    for i in ('weight', 'impurity'):
        ## Clone and prepare the tesnors needed for contraction
        ten_a1 = ten_a.clone(); ten_a2 = ten_a.clone()
        ten_b1 = ten_b.clone(); ten_b2 = ten_b.clone()
        lx1 =  l_three_dir[0].clone(); lx2 = l_three_dir[0].clone()
        ly = l_three_dir[1].clone(); lz = l_three_dir[2].clone()

        ly_tmp = ly.get_block().numpy(); ly_tmp = cytnx.from_numpy(np.sqrt(ly_tmp))
        ly_sqrt_a1 = cyx.CyTensor([cyx.Bond(D),cyx.Bond(D)],rowrank = 0, is_diag=True)
        ly_sqrt_a1.put_block(ly_tmp); ly_sqrt_a2 = ly_sqrt_a1.clone()
        ly_sqrt_b1 = ly_sqrt_a1.clone(); ly_sqrt_b2 = ly_sqrt_a1.clone()

        lz_tmp = lz.get_block().numpy(); lz_tmp = cytnx.from_numpy(np.sqrt(lz_tmp))
        lz_sqrt_a1 = cyx.CyTensor([cyx.Bond(D),cyx.Bond(D)],rowrank = 0, is_diag=True)
        lz_sqrt_a1.put_block(lz_tmp); lz_sqrt_a2 = lz_sqrt_a1.clone()
        lz_sqrt_b1 = lz_sqrt_a1.clone(); lz_sqrt_b2 = lz_sqrt_a1.clone()

        ## Set labels
        lx1.set_labels([-3,-6]); lx2.set_labels([-9,-12])
        ly_sqrt_a1.set_labels([-4,4]); ly_sqrt_b1.set_labels([-7,0])
        lz_sqrt_a1.set_labels([-5,6]); lz_sqrt_b1.set_labels([-8,2])
        ly_sqrt_a2.set_labels([-10,5]); ly_sqrt_b2.set_labels([-13,1])
        lz_sqrt_a2.set_labels([-11,7]); lz_sqrt_b2.set_labels([-14,3])
        
        if i == 'weight':
            ## Calculate weights
            ten_a1.set_labels([-1,-3,-4,-5]); ten_b1.set_labels([-2,-6,-7,-8])
            ten_a2.set_labels([-1,-9,-10,-11]); ten_b2.set_labels([-2,-12,-13,-14])
            ## Contract
            # a1_xyz = cyx.Contract(cyx.Contract(cyx.Contract(ten_a1, lx1), ly_sqrt_a1), lz_sqrt_a1)
            # b1_yz = cyx.Contract(cyx.Contract(ten_b1, ly_sqrt_b1), lz_sqrt_b1)
            # upper_half = cyx.Contract(a1_xyz, b1_yz)
            #
            # a2_xyz = cyx.Contract(cyx.Contract(cyx.Contract(ten_a2, lx2), ly_sqrt_a2), lz_sqrt_a2)
            # b2_yz = cyx.Contract(cyx.Contract(ten_b2, ly_sqrt_b2), lz_sqrt_b2)
            #
            # lower_half = cyx.Contract(a2_xyz, b2_yz)
            # weight = cyx.Contract(upper_half, lower_half.Conj())
            # weight = sort_label(weight)
            # weight1 = weight.clone().get_block().numpy()
            anet = cyx.Network("weight.net")
            # lz_sqrt_a1.print_diagram()
            anet.PutCyTensor("a1", ten_a1); anet.PutCyTensor("b1", ten_b1);
            anet.PutCyTensor("a2", ten_a2.Conj()); anet.PutCyTensor("b2", ten_b2.Conj());
            anet.PutCyTensor("lx1", lx1); anet.PutCyTensor("lx2", lx2);
            anet.PutCyTensor("ly_sqrt_a1", ly_sqrt_a1); anet.PutCyTensor("ly_sqrt_b1", ly_sqrt_b1);
            anet.PutCyTensor("ly_sqrt_a2", ly_sqrt_a2.Conj()); anet.PutCyTensor("ly_sqrt_b2", ly_sqrt_b2.Conj());
            anet.PutCyTensor("lz_sqrt_a1", lz_sqrt_a1); anet.PutCyTensor("lz_sqrt_b1", lz_sqrt_b1);
            anet.PutCyTensor("lz_sqrt_a2", lz_sqrt_a2.Conj()); anet.PutCyTensor("lz_sqrt_b2", lz_sqrt_b2.Conj());
            weight = anet.Launch()
            # weight2 = weight.clone().get_block().numpy()
            # print(linalg.norm(weight1-weight2))
        elif i == 'impurity':
            ## Calculate impurities
            d = ten_a.shape()[0]
            spin = constants_cy.physical_dimension_to_spin(d)
            sx,sy,sz,one = constants_cy.Get_spin_operators(spin)
            H = ham[0] 
            #H = - k *cytnx.linalg.Kron(sx, sx) - h * (cytnx.linalg.Kron(sz, one) + cytnx.linalg.Kron(one, sz)) / 2
            # H = cytnx.linalg.Kron(one,sx)
            H = H.reshape(d,d,d,d).permute(0,2,1,3)
            H = cyx.CyTensor(H,0)
            H.set_labels([-1,-15,-2,-16])
            #op1 = cyx.CyTensor(1j*sx.clone(), 0)
            #op2 = op1.clone()
            #op1.set_labels([-1,-15])
            #op2.set_labels([-2,-16])
            # ten_a1.set_labels([-1,-3,-4,-5]); ten_b1.set_labels([-2,-6,-7,-8])
            # ten_a2.set_labels([-15,-9,-10,-11]); ten_b2.set_labels([-16,-12,-13,-14])
            # # ## Contract
            # a1_xyz = cyx.Contract(cyx.Contract(cyx.Contract(ten_a1, lx1), ly_sqrt_a1), lz_sqrt_a1)
            # b1_yz = cyx.Contract(cyx.Contract(ten_b1, ly_sqrt_b1), lz_sqrt_b1)
            # upper_half = cyx.Contract(cyx.Contract(a1_xyz, b1_yz), H)
            #
            # a2_xyz = cyx.Contract(cyx.Contract(cyx.Contract(ten_a2, lx2), ly_sqrt_a2), lz_sqrt_a2)
            # b2_yz = cyx.Contract(cyx.Contract(ten_b2, ly_sqrt_b2), lz_sqrt_b2)
            # lower_half = cyx.Contract(a2_xyz, b2_yz)
            #
            # weight_imp = cyx.Contract(upper_half, lower_half.Conj())
            # weight_imp = sort_label(weight_imp)
            anet = cyx.Network("impurity.net")
            # lz_sqrt_a1.print_diagram()
            anet.PutCyTensor("a1", ten_a1);
            anet.PutCyTensor("b1", ten_b1);
            anet.PutCyTensor("a2", ten_a2.Conj());
            anet.PutCyTensor("b2", ten_b2.Conj());
            anet.PutCyTensor("lx1", lx1);
            anet.PutCyTensor("lx2", lx2);
            anet.PutCyTensor("ly_sqrt_a1", ly_sqrt_a1);
            anet.PutCyTensor("ly_sqrt_b1", ly_sqrt_b1);
            anet.PutCyTensor("ly_sqrt_a2", ly_sqrt_a2.Conj());
            anet.PutCyTensor("ly_sqrt_b2", ly_sqrt_b2.Conj());
            anet.PutCyTensor("lz_sqrt_a1", lz_sqrt_a1);
            anet.PutCyTensor("lz_sqrt_b1", lz_sqrt_b1);
            anet.PutCyTensor("lz_sqrt_a2", lz_sqrt_a2.Conj());
            anet.PutCyTensor("lz_sqrt_b2", lz_sqrt_b2.Conj());
            anet.PutCyTensor('H', H)
            weight_imp = anet.Launch()
            #weight_imp.reshape_(D**2, D**2, D**2, D**2)
        w = weight.get_block().numpy()
        # print(w.shape)
        # Here we use np.einsum() to calculate cns and tms, for cytnx doesn't support contraction 
        # itself. An alternative is using cytnx.linalg.Trace(); however, it is still not that 
        # convenient
        dy = dz = w.shape[0]
        c1 = w.reshape((dy, dy, dz * dz, dy * dy, dz, dz))
        c1 = np.einsum('i i j k l l->j k', c1)
        c2 = w.reshape((dy, dy, dz, dz, dy * dy, dz * dz))
        c2 = np.einsum('i i j j k l->k l', c2)
        c3 = w.reshape((dy * dy, dz, dz, dy, dy, dz * dz))
        c3 = np.einsum('i j j k k l->l i', c3)
        c4 = w.reshape((dy * dy, dz * dz, dy, dy, dz, dz))
        c4 = np.einsum('i j k k l l->i j', c4)

        t1 = np.einsum('i i j k l->j k l', w.reshape((dy, dy, dz * dz, dy * dy, dz * dz)))
        t2 = np.einsum('i j j k l->k l i', w.reshape((dy * dy, dz, dz, dy * dy, dz * dz)))
        t3 = np.einsum('i j k k l->l i j', w.reshape((dy * dy, dz * dz, dy, dy, dz * dz)))
        t4 = np.einsum('i j k l l->i j k', w.reshape((dy * dy, dz * dz, dy * dy, dz, dz)))
        def normalize(x):
            return x / np.max(np.abs(x))
        corners = tuple(map(normalize, (c1, c2, c3, c4)))
        corners = tuple(cyx.CyTensor(cytnx.from_numpy(c),0) for c in corners)
        transfer_matrices = tuple(map(normalize, (t1, t2, t3, t4)))
        transfer_matrices = tuple(cyx.CyTensor(cytnx.from_numpy(t),0) for t in transfer_matrices)


    return weight, weight_imp, corners, transfer_matrices

def ctmrg_coarse_graining(dim_cut, weight, weight_imp, cns, tms, num_of_steps = 15):
    'Return energy, which is obtained by CTMRG coarse graining scheme (Orus and Vidal\s method)'
    def tuple_rotation(c1, c2, c3, c4):
        """Returns new tuple shifted to left by one place."""
        return c2, c3, c4, c1
    def weight_rotate(weight):
        """Returns weight rotated anti-clockwise."""
        weight.permute_([1,2,3,0])
        return weight

    def extend_tensors(c1, t1, t4, weight, c4, t3):
        ## c1t1
        c1_tmp = c1.clone(); t1_tmp = t1.clone();
        c1_tmp.set_labels([-1, 1]); t1_tmp.set_labels([0, 2, -1]);
        c1t1 = sort_label(cyx.Contract(c1_tmp, t1_tmp));
        chi0 = t1.shape()[0]; chi2 = t1.shape()[1]; chi1 = c1.shape()[1]
        c1t1.reshape_(chi0, chi1 * chi2);
        ## c4t3
        c4_tmp = c4.clone(); t3_tmp = t3.clone();
        c4_tmp.set_labels([0, -1]); t3_tmp.set_labels([-1, 1, 2]);
        c4t3 = sort_label(cyx.Contract(c4_tmp, t3_tmp));
        chi0 = c4.shape()[0]; chi1 = t3.shape()[1];
        chi2 = t3.shape()[2];
        c4t3.reshape_(chi0 * chi1, chi2)
        ## t4w
        t4_tmp = t4.clone(); w_tmp = weight.clone();
        t4_tmp.set_labels([0, -1, 3]); w_tmp.set_labels([1, 2, 4, -1]);
        t4w = sort_label(cyx.Contract(t4_tmp, w_tmp))
        chi0 = t4w.shape()[0] * t4w.shape()[1];
        chi1 = t4w.shape()[2];
        chi2 = t4w.shape()[3] * t4w.shape()[4]
        t4w.reshape_(chi0, chi1, chi2)
        return c1t1, t4w, c4t3
    ############## Orus's coarse graining
    def create_projectors_Orus(c1t1, c4t3, dim_cut):
        c1t1_tmp1 = c1t1.clone(); c1t1_tmp2 = c1t1.clone();
        c4t3_tmp1 = c4t3.clone(); c4t3_tmp2 = c4t3.clone();
        c1t1_tmp1.set_labels([-1, 1]); c1t1_tmp2.set_labels([-1, 0]);
        c4t3_tmp1.set_labels([1, -1]); c4t3_tmp2.set_labels([0, -1])
        m1 = sort_label(cyx.Contract(c1t1_tmp1, c1t1_tmp2.Conj())).get_block()
        m2 = sort_label(cyx.Contract(c4t3_tmp1, c4t3_tmp2.Conj())).get_block()
        w, u = cytnx.linalg.Eigh(m1 + m2)
        u = u[:, ::-1].Conj()
        u = cyx.CyTensor(u[:, :dim_cut], 0)
        u_up = u; u_down = u.Conj()
        return u_up, u_down
    ############## Corboz's coarse graining
    def corner_extend_corboz(c1, t1, t4, w):
        anet = cyx.Network("extend_corner_corboz.net")
        anet.PutCyTensor("c1", c1)
        anet.PutCyTensor("t1", t1)
        anet.PutCyTensor("t4", t4)
        anet.PutCyTensor("w", w)
        c1_new = anet.Launch()
        dim1 = c1_new.shape()[0] * c1_new.shape()[1]
        dim2 = c1_new.shape()[2] * c1_new.shape()[3]
        c1_new = c1_new.reshape(dim1, dim2)
        return c1_new

    def create_projectors_corboz(c1, c2, c3, c4, dim_cut):
        c1_tmp = c1.clone();
        c2_tmp = c2.clone();
        c3_tmp = c3.clone();
        c4_tmp = c4.clone()
        c1_tmp.set_labels([-1, 1]);
        c2_tmp.set_labels([0, -1])
        upper_half = sort_label(cyx.Contract(c1_tmp, c2_tmp)).get_block().contiguous()
        c4_tmp.set_labels([1, -1]);
        c3_tmp.set_labels([-1, 0])
        lower_half = sort_label(cyx.Contract(c4_tmp, c3_tmp)).get_block().contiguous()
        _, r_up = cytnx.linalg.QR(upper_half)
        _, r_down = cytnx.linalg.QR(lower_half)
        rr = cytnx.linalg.Matmul(r_up.contiguous(), r_down.permute(1, 0).contiguous())
        s, u, vt = cytnx.linalg.Svd(rr)
        vt = vt.Conj();
        u = u.Conj()
        dim_new = min(s.shape()[0], dim_cut)
        s = s[:dim_new]
        s_inv2 = 1 / s ** 0.5
        s_inv2 = cytnx.linalg.Diag(s_inv2);
        u_tmp = cytnx.linalg.Matmul(u[:, :dim_new], s_inv2);
        v_tmp = cytnx.linalg.Matmul(s_inv2, vt[:dim_new, :]).permute(1, 0).contiguous();
        p_up = cytnx.linalg.Matmul(r_down.permute(1, 0).contiguous(), v_tmp)
        p_down = cytnx.linalg.Matmul(r_up.permute(1, 0).contiguous(), u_tmp)
        p_up = cyx.CyTensor(p_up, 0);
        p_down = cyx.CyTensor(p_down, 0)
        return p_up, p_down
    ########## Coarse grain
    def coarse_grain(c1t1, t4w, c4t3, u_up, u_down):
        u1 = u_up.clone();
        c1t1.set_labels([0, -1]); u1.set_labels([-1, 1]);
        c1 = sort_label(cyx.Contract(c1t1, u1))
        u2 = u_down.clone()
        c4t3.set_labels([-1, 1]); u2.set_labels([-1, 0]);
        c4 = sort_label(cyx.Contract(c4t3, u2))
        t4w.set_labels([-1, 1, -2]);
        u3 = u_down.clone(); u4 = u_up.clone();
        u3.set_labels([-1, 0]); u4.set_labels([-2, 2]);
        t4 = sort_label(cyx.Contract(cyx.Contract(u3, t4w), u4))
        return c1, t4, c4

    ############ Mian program
    c1, c2, c3, c4 = cns
    t1, t2, t3, t4 = tms
    energy = 0
    energy_mem = -1
    steps = 0
    while abs(energy - energy_mem) > 1.E-6 and steps < num_of_steps:
        for i in range(4): # four direction
            ######################## Extend tensors
            c1t1, t4w, c4t3 = extend_tensors(c1,t1,t4,weight,c4,t3)
            ######################## Create projector
            ## Orus scheme
            # u_up, u_down = create_projectors_Orus(c1t1, c4t3, dim_cut);
            ## Corboz scheme
            corner_extended = []
            for _ in range(4):
                corner_extended.append(corner_extend_corboz(c1, t1, t4, weight))
                c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
                t1, t2, t3, t4 = tuple_rotation(t1, t2, t3, t4)
                weight = weight_rotate(weight)
            u_up, u_down = create_projectors_corboz(*corner_extended, dim_cut)
            ###################### Coarse grain tensors
            c1, t4, c4 = coarse_grain(c1t1, t4w, c4t3,u_up, u_down)

            ###################### Rotate tensors
            c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
            t1, t2, t3, t4 = tuple_rotation(t1, t2, t3, t4)
            weight = weight_rotate(weight)
        cns = [c1, c2, c3, c4]; tms = [t1, t2, t3, t4]
        for j in range(4):
            norm = np.max(cytnx.linalg.Abs(cns[j].get_block_()).numpy())
            cns[j] = cns[j]/norm
            norm = np.max(cytnx.linalg.Abs(tms[j].get_block_()).numpy())
            tms[j] = tms[j]/norm 
        c1, c2, c3, c4 = cns
        t1, t2, t3, t4 = tms


        ####################### Measurement
        ## build network (see plotnet.py) 
        anet = cyx.Network("all.net")
        anet.PutCyTensor("c1",c1)
        anet.PutCyTensor("c2",c2)
        anet.PutCyTensor("c3",c3)
        anet.PutCyTensor("c4",c4)
        anet.PutCyTensor("t1",t1)
        anet.PutCyTensor("t2",t2)
        anet.PutCyTensor("t3",t3)
        anet.PutCyTensor("t4",t4)
        anet.PutCyTensor("w",weight)
        norm = anet.Launch().item()

        ## expect (reuse network)
        anet.PutCyTensor("w",weight_imp);
        
        expect = anet.Launch().item()
        ### Other method using ncon.py
        ### ncon.py can contract multiple tensors in a single operation, and it can also 
        ### find the optimized contraction steps.
        #all_mat = [a.clone() for a in all_mat]
        #all_mat  = [mat.get_block().numpy() for mat in all_mat]

        #weight_np = weight.get_block().numpy()
        #index_array = all_mat.copy()
        #index_array.append(weight_np)
        #weight_imp_np = weight_imp.get_block().numpy()
        #norm = ncon(index_array, 
        #          [c1_label, c2_label, c3_label, c4_label, t1_label, t2_label, t3_label, t4_label, [3,6,8,5]])
        #index_array = all_mat.copy()
        #index_array.append(weight_imp_np)
        #expect = ncon(index_array, 
        #           [c1_label, c2_label, c3_label, c4_label, t1_label, t2_label, t3_label, t4_label, [3,6,8,5]])
        energy_mem = energy
        #print(norm)
        energy = 3/2*expect/norm.real
    
        print('Coarse-graining(CTMRG) steps:%d'%steps,'energy = ',energy )
        steps+=1
    if steps < num_of_steps-1: print('Converge!')
    return energy

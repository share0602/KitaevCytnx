##### Import cytnx module
from setting import *
import cytnx
from cytnx import cytnx_extension as cyx
import numpy as np
from numpy import linalg as LA
ten_a = np.random.randn(2,3,3)
ten_a = cytnx.from_numpy(ten_a)
ten_a = ten_a.astype(cytnx.Type.ComplexDouble)
s = 0.5
sy = cytnx.physics.spin(s, 'y')
sx = cytnx.physics.spin(s, 'x')
ten_a = cyx.CyTensor(ten_a, 0);
# print(ten_a)
# print(ten_a.Trace(1,2))
Test = cytnx.random.normal([2,3],0.,1.)
print(Test)
test2 = cytnx.Tensor([2,3])
print(test2)
cytnx.random.Make_normal(test2, 0.,1.)
test3 = cytnx.random.Make_normal(0.,1.)
print(test2)
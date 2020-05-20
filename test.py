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
print(ten_a)
print(ten_a.Trace(1,2))
from setting import *
import cytnx as cy
from cytnx import cytnx_extension as cyx


anet = cyx.Network("extend_corner_corboz.net")
# anet = cyx.Network("weight.net")
# anet = cyx.Network("impurity.net")

anet.Diagram(figsize=[6,5])


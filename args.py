import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-spin",default='1/2', help="spin")
parser.add_argument("-D", type=int, default=4, help="D")
parser.add_argument("-chi", type=int, default=16, help="chi")
parser.add_argument("-model", default='Kitaev', choices=['TFIM', 'Heisenberg','Kitaev'], help="model name")
parser.add_argument("-hz", type=float, default=3.1, help="magnetic field")

args = parser.parse_args()


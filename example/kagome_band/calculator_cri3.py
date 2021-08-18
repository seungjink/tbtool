import os
import time
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('D:\\Project\\tbtool'))

import tbtool.io as io
from tbtool.calculator import base
from tbtool.kpoints import monkhorst_pack
import tbtool.kpoints as kp
from tbtool.calculator import berry, algo, dos, band

import ham_custom

ham = ham_custom.Cst()
ham.t1 = 1

kpt = kp.Kpath([[0,0,0],[1/3,1/3,0],[0.5,0,0], [0,0,0]])

bandcalc = band.Band(hamiltonian=ham, kpath=kpt)
res = bandcalc.calculate(51)

print(res)
np.save("kagome", res)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(np.arange(90), res)
plt.show()


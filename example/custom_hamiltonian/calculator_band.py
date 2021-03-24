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
ham.t1_z2 = -0.046
ham.t1_x2 = -0.003
ham.t3 = 0.07   # 250 -> 176
ham.t2 = -0.01
ham.t5_a = 0.034
ham.t5_b = -0.015

ham.t1_soc = 0.005
ham.t3_soc = 0.015

ham.aniso = -0.003
ham.t2_soc_z = 0.002
ham.t2_soc_xy = 0.001



#ham.t2 = 0.1

kpt = kp.Kpath([[0,0,0],[0.5,0,0],[1/3,1/3,0], [0,0,0]])

#print(a.get(3)) 
#ham = io.read_openmx_hamiltonian("./symGra.scfout", 3.8)
##print(kpt.get(30).shape)
bandcalc = band.Band(hamiltonian=ham, kpath=kpt)
res = bandcalc.calculate(30)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(np.arange(90), res)
plt.show()



kmesh = kp.Kmesh([31,31,1])
berrycalc = berry.ChernNumber(ham, kmesh=kmesh)
print(berrycalc.calculate())
#print(res)
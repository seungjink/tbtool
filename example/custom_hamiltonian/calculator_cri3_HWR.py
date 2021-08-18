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

ham = io.read_hwr("./CrI3.HWR")

#ham.t2 = 0.1

kpt = kp.Kpath([[0,0,0],[1/3,1/3,0],[0.5,0,0], [0,0,0]])

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



kmesh = kp.Kmesh([70,70,1])
berrycalc = berry.ChernNumber(ham, kmesh=kmesh)
print(berrycalc.calculate())
#print(res)
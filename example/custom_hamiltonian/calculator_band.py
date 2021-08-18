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

import ham_custom_modified_for_cri3 as ham_custom
#import ham_custom #_modified_for_cri3 as ham_custom

ham = ham_custom.Cst()
ham.t1_x2 = -0.00
ham.t1_z2 = -0.043
#ham.t3 = -0.081   # 250 -> 176
ham.t3_a = 0.07
ham.t3_b = 0.07 
ham.t2   = -0.02   
ham.t5_a = 0.034   
ham.t5_b = -0.015

ham.t1_soc =  -0.01
ham.t3_soc =  -0.02
###
ham.aniso = 0.003
ham.t2_soc_z = -0.005

ham.magmom = [1,1,0.01]
#ham.t2_soc_xy = -0.005

##### Band
kpt = kp.Kpath([[0,0,0],[0.5,0,0],[1/3,1/3,0], [0,0,0]])
bandcalc = band.Band(hamiltonian=ham, kpath=kpt)
res = bandcalc.calculate(100)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(np.arange(90), res)
plt.show()
##


kmesh = kp.Kmesh([41,41,1])
berrycalc = berry.ChernNumber(ham, kmesh=kmesh)
print(berrycalc.calculate())
#
#xarr = np.arange(-0.02, 0.0120001, 0.001)
#yarr = np.arange(0.06, 0.10001, 0.001)
#
#paramset = []
#paramval = []
#
#for x in xarr:
#    for y in yarr:
#        ham.aniso = x
#        ham.t3_a = y
#        ham.t3_b = y
#        berrycalc = berry.ChernNumber(ham, kmesh=kmesh)
#        paramset.append([x,y])
#        paramval.append(berrycalc.calculate())
#
#paramset = np.array(paramset)
#paramval = np.rint(np.array(paramval)).astype(int)
#
#np.save("paramset", paramset)
#np.save("paramval", paramval)
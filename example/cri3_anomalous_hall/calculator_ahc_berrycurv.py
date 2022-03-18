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

ham = io.read_hwr("./crsite3_2684.HWR")
#ham = io.read_hwr("./theta_90_n_2.0.HWR")
kpt = kp.Kmesh([20,20, 1])

#calc = berry.AnomalousHallConductivity(ham, kmesh=kpt)

kpt = kp.Kpath([[0,0,0],[0.5,0,0],[1/3,1/3,0], [0,0,0]])
bandcalc = band.Band(hamiltonian=ham, kpath=kpt)
res = bandcalc.calculate(100)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(np.arange(90), res)
plt.show()

#e, f = calc.calculate()

#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#plt.plot(e,f[:,0], color='r', label='band 1')
#plt.plot(e,f[:,0] + f[:,1], color='g', label='band 1 + 2')
#plt.plot(e,f[:,0] + f[:,1] + f[:,2], color='b', label='band 1+2+3')
#plt.ylabel("AHC")
#plt.xlabel("E - E_fermi")
#plt.axvline(0, linestyle='--', color='grey')
#plt.legend()
##plt.plot(e,f[:,0])
##plt.plot(e,f[:,0])
#plt.show()
#
#init_calculator = base.Mesh(ham, kpt)
#init_calculator.save('test', hamiltonian=False)


#doscalc = dos.Pdos(method='2d')
#doscalc.load('test')


### Plot
### dos[4] = 1pz
### dos[7] = 2pz
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#plt.plot(erange, dos[4] + dos[7])
#plt.show()

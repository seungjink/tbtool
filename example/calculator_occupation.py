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

ham = io.read_openmx_hamiltonian("./symGra.scfout", 3.8)
kpt = kp.Kpath([[0,0,0],[0.5,0,0],[1/3,1/3,0], [0,0,0]])
##print(kpt.get(30).shape)
#kmesh = kp.Kmesh([5,5,1])
doscalc = dos.Occupation(hamiltonian=ham, kmesh=kpt)
a = doscalc.calculate()
print(a.shape)
#print(a[2, :, 4])
# a [ scatrer_size, N_kpt, N_band]

bandcalc = band.Band(hamiltonian=ham, kpath=kpt)
res = bandcalc.calculate()

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# [x, Nband]
plt.plot(np.arange(90), res, c='k')


# [90 , band, projector]

# x = 0, 1, 2, 3.. 
for i, bd in enumerate(res):
    # bd = band energy
    #plt.scatter(x=i*np.ones(26), y=bd, s=np.abs((a[2,i,:] + a[3, i,:]))*100, c='r')
    plt.scatter(x=i*np.ones(26), y=bd, s=np.abs((a[4, i,:]))*100, c='r')

#for i, sc in enumerate(a):
#    plt.scatter(x=np.arange(90), y=res[:,i], s=sc*50)
plt.show()
#print(res)
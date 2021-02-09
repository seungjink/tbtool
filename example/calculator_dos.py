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
kpt = kp.Kmesh([20,20, 1])

#init_calculator = base.Mesh(ham, kpt)
#init_calculator.save('test', hamiltonian=False)


#doscalc = dos.Pdos(method='2d')
#doscalc.load('test')

doscalc = dos.Pdos(hamiltonian=ham, kmesh=kpt, method='2d')
erange = np.arange(-40, 40, 0.1)
dos = doscalc.calculate(erange)
print(dos.shape)

### Plot
### dos[4] = 1pz
### dos[7] = 2pz
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#plt.plot(erange, dos[4] + dos[7])
#plt.show()

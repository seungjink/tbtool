import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('D:\\Project\\tbtool'))

import tbtool.io as io
import tbtool.kpoints as kp
import tbtool.calculator.dos as dos

"""Read Hamiltonian from OpenMX scfout"""
ham = io.read_openmx_hamiltonian("./symGra.scfout", 3.8)

"""Create kpoint mesh"""
kpt = kp.Kmesh([20, 20, 1])

"""Cumulative Dos calculator"""
doscalc = dos.Cdos(hamiltonian=ham, kmesh=kpt, method='2d')
erange = np.arange(-40, 40, 0.1)
cdos = doscalc.calculate(erange)

"""Projected Dos calculator"""
#doscalc = dos.Pdos(hamiltonian=ham, kmesh=kpt, method='2d')
#erange = np.arange(-40, 40, 0.1)
#dos = doscalc.calculate(erange)

"""Fermi level"""
#doscalc = dos.Fermi(hamiltonian=ham, kmesh=kpt, method='2d')
#fermi = doscalc.calculate(9, emax=30)
#print(fermi)


### Plot
### dos[4] = 1pz
### dos[7] = 2pz
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#plt.plot(erange, cdos)
#plt.show()
#
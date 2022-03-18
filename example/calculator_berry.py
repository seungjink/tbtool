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

ham = io.read_hwr("./crsite3.HWR")

kpt = kp.Kmesh([20,20, 1])
kpt_dos = kp.Kmesh([30,30, 1])

############# Band calc #############
#kpt_band = kp.Kpath([[0,0,0],[0.5,0,0],[1/3,1/3,0], [0,0,0]])
#calc = band.Band(hamiltonian=ham, kpath=kpt_band)
#n = 30   # number of k-points between two k points.
#res = calc.calculate(n)
#
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#plt.plot(np.arange(90), res)
#plt.show()
############### END ###############



############# Chemical potential calc #############
#kpt_chempo = kp.Kmesh([40,40, 1])   # Check mesh convergence
#calc= dos.Fermi(hamiltonian=ham, kmesh=kpt_chempo)
#number_of_electrons = 2
#chemical_potential = calc.calculate(number_of_electrons)
#print(f"Calculated chemical potential : {chemical_potential}")
############### END ###############
kpts = kp.Kmesh([2,2,1], expansion=[2,1,1])
print(kpts.get())

############# Chern number calc #############
#kpt_chern = kp.Kmesh([40,40, 1])
#calc = berry.ChernNumber(ham, kmesh=kpt_chern)
#chern_number = calc.calculate()
#print(f"Chern number : {chern_number}")
############### END ###############



############# AHC calc #############
## For AHC calc, set either number of electrons or chempo.
## n =2         # n      : number of elec
## chempo = 0.0 # chempo : chemical potential  
## chempo_kmesh = kp.Kmesh([40, 40, 1])
## chempo_kmesh : Kmesh used for chemical potential calculation.
##                You can set the kmesh differently
##                when calculating the chemical potential.
##                If not set, same kmesh as AHC calculation is used.
##ahc0 = calc.calculate_only_filled_bands(n=2)
##ahc1 = calc.calculate_only_filled_bands(n=2,      chempo_kmesh=kpt_dos)
##ahc2 = calc.calculate_only_filled_bands(chempo=0, chempo_kmesh=kpt_dos)

#kpt_ahc= kp.Kmesh([5, 5, 1], expansion=[2,1,1])
#calc = berry.AnomalousHallConductivity(ham, kmesh=kpt_ahc)
#ahc = calc.calculate_only_filled_bands(n=2) # ahc : (n_kx, n_ky)-shaped array. AHC value on each kpoint
#ahc_total = np.sum(ahc)
#print(ahc_total)
############### END ###############



######## Plotting AHC by matplotlib ########
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#kpt_ahc.unitcell = np.array([
#    [1, 0, 0],
#    [-0.5, np.sqrt(3)/2, 0],
#    [0, 0, 1]
#])
#
#k_cart = kpt_ahc.get_cartesian().reshape(kpt_ahc.mesh[0],kpt_ahc.mesh[1],3)
#kx = k_cart[:,:,0].reshape(kpt_ahc.mesh[0], kpt_ahc.mesh[1])
#ky = k_cart[:,:,1].reshape(kpt_ahc.mesh[0], kpt_ahc.mesh[1])
#
#fig, ax = plt.subplots()
#c = ax.pcolormesh(kx, ky, ahc, cmap='RdBu') #, vmin=-1, vmax=1)
#fig.colorbar(c, ax=ax)
#plt.show()
############### END ###############



############# AHC calc - Band by Band contribution #############
# This calculates total AHC band by band by increasing chemical potential by ediff.
#kpt_ahc= kp.Kmesh([30,30,1], expansion=[2,2,1], gamma_center=False)
#calc = berry.AnomalousHallConductivity(ham, kmesh=kpt_ahc)
#energy, ahc = calc.calculate(emin=-1, emax=1, ediff=0.1)
#print(ahc)
############### END ###############

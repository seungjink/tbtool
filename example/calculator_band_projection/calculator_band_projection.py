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

scfout_path = './Fe3C12O12.scfout'
spin = 'up'

projector = [
    [2, '1dyz', '1dxz'],
    [2, '1dz2'],
    [2, '1dx2'],
    [2, '1dxy'],
    [17, '1px', '1py'],
    [17, '1pz']
]
projector_label = ['dyz+dxz', 'dz2', 'dxy', 'dx2', 'px+py', 'pz']
projector_factor = [1,1,1,1,4,4]

def get_index(projectors):
    idx = []
    for proj in projectors:
        idx_tmp = []
        for i in range(len(proj)-1):
            idx_tmp.append(mxbasis.basis.getindex(proj[0], proj[i+1]))
        idx.append(idx_tmp)
    return idx

def draw_multiple(idx, filename, fig, band, occ_total):
    ax = fig.add_subplot(111)
    for j in idx:
        occ = occ_total[j]
        for i, bd in enumerate(band):
            ax.scatter(x=kx[i]*np.ones(count_band), y=bd, s=occ[i] * projector_factor[j] * 100, c=f'C{j}')

    plotfont = {'fontname':'serif', 'size':16}
    plottickfont = {'fontname':'serif', 'size':16}

    ax.plot(kx, band, c='darkgray', linewidth=0.5)
    ax.set_ylim(-8,5)

    legend_elements = []
    for j in idx:
        legend_elements.append(Line2D([0], [0], marker='o', color=f'C{j}', label=f'{projector_label[j]}'))
    ax.legend(handles=legend_elements, loc='upper right',prop={'size':14, 'family':'serif'}, fancybox=True, framealpha=0.0)

    ax.set_xticks(klabel)
    ax.set_xticklabels('GMKG', **plottickfont)
    ax.set_yticklabels(ax.get_yticks(), **plottickfont)
    ax.vlines(x=klabel, ymin=-8, ymax=5, linestyle='--', color='lightgray')

    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.savefig(f'{filename}.pdf')
    plt.clf()

ham = io.read_openmx_hamiltonian(scfout_path, 3.9, spin=spin)

kpt = kp.Kpath([[0,0,0],[0.5,0,0],[1/3,1/3,0], [0,0,0]], n=30)
kpt.unitcell = ham.scfout.tv*0.529177249

bandcalc = band.Band(hamiltonian=ham, kpath=kpt)
occcalc = dos.Occupation(hamiltonian=ham, kmesh=kpt)

occ = occcalc.calculate()
band = bandcalc.calculate()

#######END of base calculation.

# band = [kpt_index, energy_index]
# occ = [basis_index, kpt_index, energy_index]

# Set projector and get occupation of each orbital
mxbasis = io.read_openmx_basis(scfout_path, 3.9)

occ_total = np.zeros((len(projector), band.shape[0], band.shape[1]))
index_projectors = get_index(projector)
for i, proj in enumerate(index_projectors):
    occ_total[i] = np.abs(np.sum(occ[proj], axis=0))

# Now plotting part.

count_band = band.shape[1]
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
kx, klabel = kpt.get_linear()
fig = plt.figure()

# draw single
for j, occ in enumerate(occ_total):
    ax = fig.add_subplot(111)
    for i, bd in enumerate(band):
        ax.scatter(x=kx[i]*np.ones(count_band), y=bd, s=occ[i] * projector_factor[j] * 100, c=f'C{j}')

    plotfont = {'fontname':'serif', 'size':16}
    plottickfont = {'fontname':'serif', 'size':16}

    ax.plot(kx, band, c='darkgray', linewidth=0.5)
    ax.set_ylim(-8,5)

    legend_elements = [Line2D([0], [0], marker='o', color=f'C{j}', label=f'{projector_label[j]}')]
    ax.legend(handles=legend_elements, loc='upper right',prop={'size':14, 'family':'serif'}, fancybox=True, framealpha=0.0)

    ax.set_xticks(klabel)
    ax.set_xticklabels('GMKG', **plottickfont)
    ax.set_yticklabels(ax.get_yticks(), **plottickfont)
    ax.vlines(x=klabel, ymin=-8, ymax=5, linestyle='--', color='lightgray')

    plt.tight_layout()
    plt.savefig(f'{projector_label[j]}_{spin}.png')
    plt.savefig(f'{projector_label[j]}_{spin}.pdf')
    plt.clf()


# draw multiple 
# projector_label = ['dyz+dxz', 'dz2', 'dxy', 'dx2', 'px+py', 'pz']
draw_multiple(np.arange(len(projector)), f'total_{spin}', fig, band, occ_total)
draw_multiple([0, 5], f'dyz_dzx_pz_{spin}', fig, band, occ_total)
draw_multiple([2, 3,4], f'dxy_dx2_px_py_{spin}', fig, band, occ_total)
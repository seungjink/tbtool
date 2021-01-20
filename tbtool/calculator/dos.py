import sys
import functools
import logging
import pickle
from pathlib import Path
import numpy as np
import tbtool.kpoints as kp
import tbtool.unit as unit
import tbtool.calculator.algo as algo
import tbtool.calculator.base as base

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s - %(name)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Pdos:
    def __init__(self, hamiltonian=None, kmesh=None, method='3d'):
        self.hamiltonian = hamiltonian
#        assert isinstance(kmesh, kp.Kmesh), 'Argument {} must be kpoints.Kmesh'.format(kmesh)
        self.kmesh = kmesh
        self.method = method

        self.wavefunctions = None
        self.energies = None
        self.overlaps = None

        self.previous_mesh = np.zeros([1,3])
        self.previous_unitcell = np.zeros([3,3])
        self.overlap_gamma = None

    def _calculate_mesh(self):
        logger.info(f'Diagonalizing wavefunction on given meshes.')
        chk = self.is_mesh_changed(self.kmesh.mesh, self.kmesh.unitcell)
        if chk :
            kpts = self.kmesh.get()
            evs = []
            ens = []
            olps = []
            for kpt in kpts:
                en, ev = self.hamiltonian.diagonalize(kpt, eigvals_only=False)
                ens.append(en)
                evs.append(ev)
                olps.append(self.hamiltonian.get(kpt)[1])

            self.wavefunctions = np.array(evs)
            self.energies = np.array(ens)
            self.overlaps = np.array(olps) 
        else:
            pass
    
    def calculate(self, projector, erange, occupation='dual'):
        logger.info('Calculate projections')
        self._calculate_mesh()

        ek, vk, olpk = self.energies, self.wavefunctions, self.overlaps 

#        ek_gamma, vk_gamma, = self.hamiltonian.diagonalize([0,0,0], eigvals_only=False)
        if self.overlap_gamma is None:
            h, olpk_gamma = self.hamiltonian.get([0,0,0])
            self.overlap_gamma = olpk_gamma

        projector = np.array(projector)
        if projector.ndim == 1:
            projector = np.expand_dims(projector, axis=0)

        #dd = np.matmul(np.conjugate(projector), olpk_gamma)
        projector_norm = np.sqrt(np.einsum('ij,ij->i', np.matmul(np.conjugate(projector), self.overlap_gamma), projector))
        projector = projector / projector_norm[:, np.newaxis]
        projector = np.transpose(projector)

        # vk.shape = [Nk, Nbasis, Nbasis]
        if occupation == 'dual':
            ls = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), olpk)
            occ_full = np.matmul(ls, projector)
            occ_onsite = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), projector)
            occ_dual = np.multiply(occ_full.conj(), occ_onsite)
            occ = occ_dual.real
        elif occupation == 'full':
            ls = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), olpk)
            occ_full = np.matmul(ls, projector)
            occ = np.square(np.absolute(occ_full, out=occ_full))
        elif occupation == 'onsite':
            occ_onsite = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), projector)
            occ = np.square(np.absolute(occ_onsite, out=occ_onsite))

        res = np.moveaxis(occ, -1, 0)
        # Since thisis 2d tetrahedron method, se need to pass self.kmesh.get()[:,:2]
        if self.method == '2d':
            dos_result = algo.integrate_delta_2d_tetra(self.kmesh.get()[:,:2], self.energies, res, erange)
        elif self.method == '3d':
            dos_result = algo.integrate_delta_3d_tetra(self.kmesh.get(), self.energies, res, erange)

        return dos_result

    def is_mesh_changed(self, mesh, unitcell):
        if np.array_equal(mesh, self.previous_mesh) and \
           np.array_equal(unitcell, self.previous_unitcell):

            return False
        else:
            logger.info(f'K mesh changed from {self.previous_mesh} to {mesh}')
            return True
    
    def load(self, prefix):
        p = Path(sys.argv[0]).parent / f'{prefix}_tb_restart'
#        with open(p.absolute().as_posix() + f'/{prefix}_hamiltonian.npy', 'rb') as f:
#            self.hamiltonian = np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_overlap.npy', 'rb') as f:
            self.overlaps= np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_overlap_gamma.npy', 'rb') as f:
            self.overlap_gamma= np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_eigenvalue.npy', 'rb') as f:
            self.energies = np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_eigenvector.npy', 'rb') as f:
            self.wavefunctions = np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_kpts.pickle', 'rb') as f:
            self.kmesh = pickle.load(f)

        self.previous_mesh = self.kmesh.mesh
        self.previous_unitcell = self.kmesh.unitcell



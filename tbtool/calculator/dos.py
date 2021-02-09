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

class Occupation:
    def __init__(self, hamiltonian=None, kmesh=None):
        self.hamiltonian = hamiltonian

        # kpts should implement kpts.get() method
        # and kpts.get() should return array of k points to be calculated.
        self.kmesh = kmesh

        self.wavefunctions = None
        self.energies = None
        self.overlaps = None

        self.previous_mesh = np.zeros([1,3])
        self.overlap_gamma = None
    
    def is_mesh_changed(self, mesh):
        if np.array_equal(mesh, self.previous_mesh):
            return False
        else:
            return True

    def _calculate_mesh(self):
        logger.info(f'Diagonalizing wavefunction on given meshes.')
        chk = self.is_mesh_changed(self.kmesh.mesh)
        if chk :
            kpts = self.kmesh.get()
            mesh_calculator = base.Eigen(self.hamiltonian, kpts)
            ens, evs = mesh_calculator.calculate()
            olps = []
            for kpt in kpts:
                olps.append(self.hamiltonian.get(kpt)[1])

            self.wavefunctions = np.array(evs)
            self.energies = np.array(ens)
            self.overlaps = np.array(olps) 
        else:
            pass
    
    def calculate(self, projector=None, occupation='dual'):
        logger.info('Calculate occupation numbers')
        self._calculate_mesh()

        ek, vk, olpk = self.energies, self.wavefunctions, self.overlaps 

        if self.overlap_gamma is None:
            h, olpk_gamma = self.hamiltonian.get([0,0,0])
            self.overlap_gamma = olpk_gamma

        # Default projector = unit matrix
        if projector is None:
            # ToDo: What is the best way to estimate the dimension of Hamiltonian?
            projector = np.eye(self.overlap_gamma.shape[0])
        else:
            projector = np.array(projector)

        if projector.ndim == 1:
            projector = np.expand_dims(projector, axis=0)

        # Normalize projector.
        projector_norm = np.sqrt(np.einsum('ij,ij->i', np.matmul(np.conjugate(projector), self.overlap_gamma), projector))
        projector = projector / projector_norm[:, np.newaxis]
        projector = np.transpose(projector)

        # vk.shape = [Nk, Nbasis, Nbasis]
        # vk[N, :, J] = Jth eigenvector at Nth k point.
        # occ_dual   = (<n,k|S|proj><proj|n,k> + H.C.) * 0.5
        # occ_full   = <n,k|S|proj><proj|S|n,k>
        # occ_onsite = <n,k|proj><proj|n,k>
        if occupation == 'dual':
            ls = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), olpk)
            occ_full = np.matmul(ls, projector)
            occ_onsite = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), projector)
            occ_dual = np.multiply(occ_full.conj(), occ_onsite)
            occ = occ_dual.real
        elif occupation == 'full':
            ls = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), olpk)
            occ_full = np.matmul(ls, projector)
            occ = np.square(np.absolute(occ_full))
        elif occupation == 'onsite':
            occ_onsite = np.matmul(np.transpose(vk, axes=(0,2,1)).conj(), projector)
            occ = np.square(np.absolute(occ_onsite))

        res = np.moveaxis(occ, -1, 0)
        return res
class Pdos:
    def __init__(self, hamiltonian=None, kmesh=None, method='3d'):
        self.hamiltonian = hamiltonian
#        assert isinstance(kmesh, kp.Kmesh), 'Argument {} must be kpoints.Kmesh'.format(kmesh)
        self.kmesh = kmesh
        self.method = method

        self.occupation = Occupation(hamiltonian = self.hamiltonian, kmesh=self.kmesh)

    def calculate(self, erange, projector=None, occupation='dual'):
        res = self.occupation.calculate(projector=projector, occupation=occupation)

        # 2d tetrahedron method -> pass 2d kpts. self.kmesh.get()[:,:2]
        if self.method == '2d':
            dos_result = algo.integrate_delta_2d_tetra(self.kmesh.get()[:,:2], self.occupation.energies, res, erange)
        elif self.method == '3d':
            dos_result = algo.integrate_delta_3d_tetra(self.kmesh.get(), self.occupation.energies, res, erange)
        return dos_result
    
    def load(self, prefix):
        p = Path(sys.argv[0]).parent / f'{prefix}_tb_restart'
#        with open(p.absolute().as_posix() + f'/{prefix}_hamiltonian.npy', 'rb') as f:
#            self.hamiltonian = np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_overlap.npy', 'rb') as f:
            self.occupation.overlaps= np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_overlap_gamma.npy', 'rb') as f:
            self.occupation.overlap_gamma= np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_eigenvalue.npy', 'rb') as f:
            self.occupation.energies = np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_eigenvector.npy', 'rb') as f:
            self.occupation.wavefunctions = np.load(f)
        with open(p.absolute().as_posix() + f'/{prefix}_kpts.pickle', 'rb') as f:
            self.kmesh = pickle.load(f)
            self.occupation.kmesh = self.kmesh

        self.occupation.previous_mesh = self.kmesh.mesh
        self.occupation.previous_unitcell = self.kmesh.unitcell

import sys
import functools
import logging
import numpy as np
import tbtool.kpoints as kp
import tbtool.calculator.algo as algo
import tbtool.calculator.dos as dos

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s - %(name)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Overlap:
    def __init__(self, hamiltonian=None, kmesh=None):
        self.hamiltonian = hamiltonian
        assert isinstance(kmesh, kp.Kmesh), 'Argument {} must be kpoints.Kmesh'.format(kmesh)
        self.kmesh = kmesh

        self.u12 = None
        self.u23 = None
        self.u34 = None
        self.u41 = None

        self.energy = None

        self.previous_mesh = np.zeros([1,3])
        self.previous_unitcell = np.zeros([3,3])

    def calculate(self):
        logger.info(f'Calculating overlap matrix.')

        chk = self.is_mesh_changed(self.kmesh.mesh, self.kmesh.unitcell)
        if chk :
            kpts = self.kmesh.get()
            n1, n2 = self.kmesh.mesh[:2]

            ens = []
            evs = []
            for kpt in kpts:
                en, ev = self.hamiltonian.diagonalize(kpt, eigvals_only=False)
                ens.append(en)
                evs.append(ev)

            self.energy = np.array(ens, dtype=float)
            self.energy = np.reshape(self.energy, (n1, n2, self.energy.shape[-1]))
            evs = np.array(evs)

            evs_n0m0 = np.reshape(evs, (n1, n2, evs.shape[1], evs.shape[2]))
            evs_n1m0 = np.roll(evs_n0m0, -1, axis=0)
            evs_n1m1 = np.roll(evs_n1m0, -1, axis=1)
            evs_n0m1 = np.roll(evs_n1m1, 1, axis=0)

            evs_n0m0_dagger = np.transpose(np.conj(evs_n0m0), axes=(0, 1, 3, 2))
            evs_n1m0_dagger = np.transpose(np.conj(evs_n1m0), axes=(0, 1, 3, 2))
            evs_n1m1_dagger = np.transpose(np.conj(evs_n1m1), axes=(0, 1, 3, 2))
            evs_n0m1_dagger = np.transpose(np.conj(evs_n0m1), axes=(0, 1, 3, 2))

            self.u12 = np.matmul(evs_n0m0_dagger, evs_n1m0)
            self.u23 = np.matmul(evs_n1m0_dagger, evs_n1m1)
            self.u34 = np.matmul(evs_n1m1_dagger, evs_n0m1)
            self.u41 = np.matmul(evs_n0m1_dagger, evs_n0m0)

            self.previous_mesh = self.kmesh.mesh
            self.previous_unitcell = self.kmesh.unitcell
        else:
            pass

    def is_mesh_changed(self, mesh, unitcell):
        if np.array_equal(mesh, self.previous_mesh) and \
           np.array_equal(unitcell, self.previous_unitcell):

            return False
        else:
            logger.info(f'K mesh changed from {self.previous_mesh} to {mesh}')
            return True


class BerryCurvature:
    def __init__(self, hamiltonian, kmesh=None):
        self.hamiltonian = hamiltonian
        assert isinstance(kmesh, kp.Kmesh), 'Argument {} must be kpoints.Kmesh'.format(kmesh)
        self.kmesh = kmesh
        self.overlap = Overlap(hamiltonian=self.hamiltonian, kmesh=self.kmesh)
    
    def calculate(self):
        self.overlap.calculate()
        u12 = self.overlap.u12
        u23 = self.overlap.u23
        u34 = self.overlap.u34
        u41 = self.overlap.u41

        u12_abelian = np.diagonal(u12, axis1=2, axis2=3)
        u23_abelian = np.diagonal(u23, axis1=2, axis2=3)
        u34_abelian = np.diagonal(u34, axis1=2, axis2=3)
        u41_abelian = np.diagonal(u41, axis1=2, axis2=3)
    
        u_abelian = functools.reduce(
            np.multiply, [u12_abelian, u23_abelian, u34_abelian, u41_abelian]
        )
        Fk_abelian = np.angle(u_abelian)

        return Fk_abelian

    def calculate_only_filled_bands(self, n:float=None, chempo:float=None, chempo_kmesh=None):

        fk = self.calculate() / 2.0 / np.pi
        en = self.overlap.energy

        fk = np.reshape(fk, (-1, fk.shape[-1]))
        en = np.reshape(en, (-1, en.shape[-1]))

        if not chempo:
            if chempo_kmesh != None:
                chempo_calculator = dos.Fermi(hamiltonian=self.hamiltonian, kmesh=chempo_kmesh)
            else:
                chempo_calculator = dos.Fermi(hamiltonian=self.hamiltonian, kmesh=self.kmesh)
            chempo = chempo_calculator.calculate(n)
        return np.sum(fk, axis=0, where=(en <= chempo))
#    def calculate_only_filled_bands(self, n:float, chempo_kmesh=None):

class ChernNumber:
    def __init__(self, hamiltonian, kmesh=None):
        self.hamiltonian = hamiltonian
        assert isinstance(kmesh, kp.Kmesh), 'Argument {} must be kpoints.Kmesh'.format(kmesh)
        self.kmesh = kmesh
        self.berrycurvature = BerryCurvature(hamiltonian=self.hamiltonian, kmesh=self.kmesh)
    
    def calculate(self):
        fk = self.berrycurvature.calculate()
        chern_abelian = np.sum(fk, axis=(0, 1)) / 2.0 / np.pi
        return chern_abelian

class AnomalousHallConductivity:
    def __init__(self, hamiltonian, kmesh=None):
        self.hamiltonian = hamiltonian
        assert isinstance(kmesh, kp.Kmesh), f'Argument {kmesh} must be kpoints.Kmesh'
        self.kmesh = kmesh
        self.berrycurvature = BerryCurvature(hamiltonian=self.hamiltonian, kmesh=self.kmesh)

    def calculate(self, emin=-1, emax=1, ediff=0.01):
        """
        calculate AHC band by band in the given energy range.

        Args:
           emin (float, optional): lower energy bound of AHC calculation.
           emax (float, optional): upper energy bound of AHC calculation.
           ediff (float, optional): energy spacing in the region.

        Returns:
            numpy.ndarray: [(emax-emin)/emin, band_count] shaped AHC values. 

        """

        assert emin < emax, f'emin = {emin} has to be smaller than emax = {emax}'

        fk = self.berrycurvature.calculate() / 2.0 / np.pi
        en = self.berrycurvature.overlap.energy

        fk = np.reshape(fk, (-1, fk.shape[-1]))
        en = np.reshape(en, (-1, en.shape[-1]))

        ahc = []
        energies = np.arange(emin, emax, ediff)

        for energy in energies:
            ahc.append(np.sum(fk, axis=0, where=(en <= energy)))

        return energies, np.array(ahc, dtype=float)
    
    def calculate_only_filled_bands(self, n:float=None, chempo:float=None, chempo_kmesh=None):
        """
        calculate the AHC value for the given electron number n or chemical potential.

        Args:
           n (float, optional): Number of electron in the system.
           chempo (float, optional): Chemical potential in the system.
           chempo_kmesh (kpoint.kmesh, optional):
               Kpoint mesh used for chemical potential calculation.
               If not given, original Kpoint mesh is used.
        
        If electron number `n` is given, chemical potential is automatically calculated.
        If chemical potential `chempo` is given, this value is directly used.
        So, only one of either `n` or `chempo` should be given.

        In general case, we need less Kpoints for the determination of
        chemical potential than the case of Berry curvature.
        So, if you want to use different Kpoints for these two calculations,
        set `chempo_kmesh`.

        Returns:
            float: Sum of AHC 
            ndarray: AHC at each k point
        """

        if n is not None and chempo is not None:
            logger.error('----- ERROR -----')
            logger.error('Please set only one value of electron number and chemical potential.')
            logger.error(f'electron number    : {n}')
            logger.error(f'chemical potential : {chempo}')
            sys.exit(1)
        elif n is None and chempo is None:
            logger.error('----- ERROR -----')
            logger.error('Please set one value of electron number or chemical potential.')
            logger.error(f'electron number    : {n}')
            logger.error(f'chemical potential : {chempo}')
            sys.exit(1)

        fk = self.berrycurvature.calculate() / 2.0 / np.pi
        en = self.berrycurvature.overlap.energy

        #fk = np.reshape(fk, (-1, fk.shape[-1]))
        #en = np.reshape(en, (-1, en.shape[-1]))

        if chempo is not None:
            if not chempo.isdigit():
                logger.error('----- ERROR -----')
                logger.error('The value of chemical potential has to be number type')
                logger.error(f'chemical potential : {chempo}')
                sys.exit(1)

            if chempo_kmesh != None:
                chempo_calculator = dos.Fermi(hamiltonian=self.hamiltonian, kmesh=chempo_kmesh)
            else:
                chempo_calculator = dos.Fermi(hamiltonian=self.hamiltonian, kmesh=self.kmesh)
            chempo = chempo_calculator.calculate(n)
        ahc = np.sum(fk, where=(en <= chempo), axis=2)
        return ahc

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh, eigvalsh
import tbtool.unit as unit

class Hamiltonian(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get(self):
        # return Hamiltonian matrix.
        pass

    @abstractmethod
    def diagonalize(self):
        # return eigenvalue(+eigenvectors)
        pass


class Wannier(Hamiltonian):
    TYPE = "Wannier Hamiltonian"

    def __init__(self, hopping, cell, filename=None):
        self.hopping = np.array(hopping)
        self.cell = np.array(cell)
        self.filename = filename
        self.unit = {"energy: eV"}

    def get(self, kpt):
        exp_ikr = np.exp(1j * 2.0 * np.pi * np.dot(self.cell, kpt))
        ham = np.sum(
            np.multiply(self.hopping, exp_ikr[:, np.newaxis, np.newaxis]),
            axis=0
        )
        return ham

    def diagonalize(self, kpt, eigvals_only=True):
        ham = self.get(kpt)
        if eigvals_only:
            return eigvalsh(ham)
        else:
            return eigh(ham)

class Openmx(Hamiltonian):
    TYPE = "OpenMX Hamiltonian"

    def __init__(self, mxscfout, unit='ev'):
        self.scfout = mxscfout
        self.scfout.readfile()
        self.hopping, self.overlap, self.cell, self.dimension, self.chemp \
            = self.scfout.get_hamiltonian()
        self.unit = {'energy': 'ev'}
    
    def get(self, kpt):
        exp_ikr = np.exp(1j * 2.0 * np.pi * np.dot(self.cell, kpt))
        ham = np.sum(
            np.multiply(self.hopping, exp_ikr[:, np.newaxis, np.newaxis]),
            axis=0
        )
        olp = np.sum(
            np.multiply(self.overlap, exp_ikr[:, np.newaxis, np.newaxis]),
            axis=0
        )
        return ham, olp

    def diagonalize(self, kpt, eigvals_only=True):
        ham, olp = self.get(kpt)
        if eigvals_only:
            en = eigvalsh(ham, olp, lower=False)
            return (en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy'])
        else:
            en, ev = eigh(ham, olp, lower=False)
            return ((en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy']), ev)
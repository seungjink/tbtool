from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh, eigvalsh
import tbtool.unit as unit

class Hamiltonian(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get(self):
        # return Hamiltonian matrix that can be diagonalized.
        pass

    @abstractmethod
    def diagonalize(self):
        # return eigenvalue(+eigenvectors) after diagonalization.
        pass


class Wannier(Hamiltonian):
    TYPE = "Wannier Hamiltonian"

    def __init__(self, hopping, cell, filename=None, chemp=0):
        self.hopping = np.array(hopping)
        self.cell = np.array(cell)
        self.filename = filename
        self.unit = {'energy': 'ev'}
        self.chemp = chemp

    def get(self, kpt):
        exp_ikr = np.exp(1j * 2.0 * np.pi * np.dot(self.cell, kpt))
        ham = np.sum(
            np.multiply(self.hopping, exp_ikr[:, np.newaxis, np.newaxis]),
            axis=0
        )
        olp = np.eye(ham.shape[0]) # * ham.shape[1]).reshape((ham.shape))
        return ham, olp

    def diagonalize(self, kpt, eigvals_only=True, fermilevel=True):
        ham = self.get(kpt)[0]
        if eigvals_only:
            en = eigvalsh(ham, lower=False)
            return (en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy'])
        else:
            en, ev = eigh(ham, lower=False)
            return ((en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy']), ev)

class Openmx(Hamiltonian):
    TYPE = "OpenMX Hamiltonian"

    def __init__(self, mxscfout, unit='ev', spin=None):
        self.scfout = mxscfout
        self.scfout.readfile()
        if spin == 'up':
            self.hopping, self.overlap, self.cell, self.dimension, self.chemp \
                = self.scfout.get_hamiltonian()
            self.hopping = self.hopping[0]
        elif spin == 'down':
            self.hopping, self.overlap, self.cell, self.dimension, self.chemp \
                = self.scfout.get_hamiltonian()
            self.hopping = self.hopping[1]
        else:
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
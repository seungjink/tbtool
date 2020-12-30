import numpy as np
from scipy.linalg import eigh, eigvalsh

class Hamiltonian:

    def __init__(self, hopping, cell, basis=None):
        self.hopping = np.array(hopping)
        self.cell = np.array(cell)
        pass

    def solve(self, rtn=True):
        # return eigenvalue (+ eigenvector)
        pass

    def get(self):
        # return hamiltonian at given k point.
        pass


class Wannier:
    TYPE = "Wannier Hamiltonian"

    def __init__(self, hopping, cell, filename=None):
        self.hopping = np.array(hopping)
        self.cell = np.array(cell)
        self.filename = filename

    def get(self, kpt):
        exp_ikr = np.exp(1j * 2.0 * np.pi * np.dot(self.cell, kpt))
        ham = np.sum(
            np.multiply(self.hopping, exp_ikr[:, np.newaxis, np.newaxis]),
            axis=0
        )
        return ham

    def solve(self, kpt, eigvals_only=True):
        ham = self.get(kpt)
        if eigvals_only:
            return eigvalsh(ham)
        else:
            return eigh(ham)

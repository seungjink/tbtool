import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('D:\\Project\\tbtool'))
import tbtool.hamiltonian as ham
import tbtool.unit as unit
import numpy as np
from scipy.linalg import eigh, eigvalsh

class Cst(ham.Hamiltonian):
    def __init__(self):
        self.t1 = 0.0

        self.chemp = 0.0
        self.unit = {"energy": "ev"}

    def get(self, kpt):
        kx, ky, kz = kpt

        I = 1j
        exp = np.exp
        pi = np.pi
        sqrt = np.sqrt

        mat = self.t1 * np.array([
            [0,1 + exp(-2*I*pi*kx), exp(2*I*pi*ky) + exp(-2*I*pi*kx)],
            [0,0,1 + exp(2*I*pi*ky)],
            [0,0,0]])

        return mat, np.eye(3)

    def diagonalize(self, kpt, eigvals_only=True):
        ham, olp = self.get(kpt)
        if eigvals_only:
            en = eigvalsh(ham, olp, lower=False)
            return (en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy'])
        else:
            en, ev = eigh(ham, olp, lower=False)
            return ((en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy']), ev)
import numpy as np
class Eigenvalues:


    def __init__(self, hamiltonian, kpts):
        self.hamiltonian = hamiltonian
        self.kpts = kpts
    
    def calculate(self):
        result = []
        for kpt in self.kpts:
            result.append(self.hamiltonian.diagonalize(kpt, eigvals_only=True))

        return np.array(result)

class Eigenvectors:


    def __init__(self, hamiltonian, kpts):
        self.hamiltonian = hamiltonian
        self.kpts = kpts

    def calculate(self):
        result = []
        for kpt in self.kpts:
            result.append(self.hamiltonian.diagonalize(kpt, eigvals_only=False)[1])

        return np.array(result)
import numpy as np
class Eigenvalues:


    def __init__(self, hamiltonian, kpts):
        self.hamiltonian = hamiltonian
        self.kpts = kpts
        self.eigenvalues = None
    
    def calculate(self):
        result = []
        for kpt in self.kpts:
            result.append(self.hamiltonian.diagonalize(kpt, eigvals_only=True))
        self.eigenvalues = np.array(result)
        return self.eigenvalues

class Eigenvectors:

    def __init__(self, hamiltonian, kpts):
        self.hamiltonian = hamiltonian
        self.kpts = kpts
        self.eigenvectors = None

    def calculate(self):
        result = []
        for kpt in self.kpts:
            result.append(self.hamiltonian.diagonalize(kpt, eigvals_only=False)[1])
        self.eigenvectors = np.array(result)
        return result

class Eigen:

    def __init__(self, hamiltonian, kpts):
        self.hamiltonian = hamiltonian
        self.kpts = kpts
        self.eigenvalues = None
        self.eigenvectors = None

    def calculate(self):
        result_en = []
        result_ev = []
        for kpt in self.kpts:
            en, ev = self.hamiltonian.diagonalize(kpt, eigvals_only=False)
            result_en.append(en)
            result_ev.append(ev)
        self.eigenvalues = np.array(result_en)
        self.eigenvectors = np.array(result_ev)
        return self.eigenvalues, self.eigenvectors
import sys
import numpy as np
import pickle
from scipy import sparse
from pathlib import Path
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
    """Calculator for eigenvalues and eigenvectors

    Args:
        hamiltonian (tbtool.hamiltonian): tbtool.hamiltonian object
        kpts (array_like): list of k points where eigenvalues are calculated.
    """

    def __init__(self, hamiltonian, kpts):
        self.hamiltonian = hamiltonian
        self.kpts = kpts
        self.eigenvalues = None
        self.eigenvectors = None

    def calculate(self):
        """Diagonalize Hamiltonian on given mesh

        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        result_en = []
        result_ev = []
        for kpt in self.kpts:
            en, ev = self.hamiltonian.diagonalize(kpt, eigvals_only=False)
            result_en.append(en)
            result_ev.append(ev)
        self.eigenvalues = np.array(result_en)
        self.eigenvectors = np.array(result_ev)
        return self.eigenvalues, self.eigenvectors

class Mesh:

    def __init__(self, hamiltonian, kmesh):
        self.hamiltonian = hamiltonian
        self.kmesh = kmesh
    
    def save(self, prefix, hamiltonian=True, overlap=True, eigenvalues=True, eigenvectors=True):
        p = Path(sys.argv[0]).parent / f'{prefix}_tb_restart'
        p.mkdir(exist_ok=True, parents=True)

        kmesh = self.kmesh.get()
        ham_k = []
        olp_k = []
        en_k = []
        ev_k = []
        for kpt in kmesh:
            ham, olp = self.hamiltonian.get(kpt)
            en, ev = self.hamiltonian.diagonalize(kpt, eigvals_only=False)
            ham_k.append(ham)
            olp_k.append(olp)
            en_k.append(en)
            ev_k.append(ev)
        ham_k = np.array(ham_k)
        olp_k = np.array(olp_k)
        en_k = np.array(en_k)
        ev_k = np.array(ev_k)
        
        if overlap:
            with open(p.absolute().as_posix() + f'/{prefix}_overlap.npy', 'wb') as f:
                np.save(f, olp_k)
        if hamiltonian:
            with open(p.absolute().as_posix() + f'/{prefix}_hamiltonian.npy', 'wb') as f:
                np.save(f, ham_k)
        if eigenvalues:
            with open(p.absolute().as_posix() + f'/{prefix}_eigenvalue.npy', 'wb') as f:
                np.save(f, en_k)
        if eigenvectors:
            with open(p.absolute().as_posix() + f'/{prefix}_eigenvector.npy', 'wb') as f:
                np.save(f, ev_k)
        with open(p.absolute().as_posix() + f'/{prefix}_kpts.pickle', 'wb') as f:
            pickle.dump(self.kmesh, f)

        ham_gamma, olp_gamma = self.hamiltonian.get([0,0,0])
        with open(p.absolute().as_posix() + f'/{prefix}_hamiltonian_gamma.npy', 'wb') as f:
            np.save(f, ham_gamma)
        with open(p.absolute().as_posix() + f'/{prefix}_overlap_gamma.npy', 'wb') as f:
            np.save(f, olp_gamma)
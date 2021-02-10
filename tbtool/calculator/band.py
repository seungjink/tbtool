import sys
import functools
import logging
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

class Band:
    def __init__(self, hamiltonian=None, kpath=None):
        self.hamiltonian = hamiltonian
#        assert isinstance(kmesh, kp.Kmesh), 'Argument {} must be kpoints.Kmesh'.format(kmesh)
        self.kpath = kpath
    
    def calculate(self, n=None):
        kpts = self.kpath.get()
        mesh_calculator = base.Eigenvalues(self.hamiltonian, kpts)
        evs = mesh_calculator.calculate()
        return evs

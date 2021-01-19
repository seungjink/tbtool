import traceback
from collections import namedtuple
import numpy as np
import tbtool.io

class Basis:
    """
    Set basis information during the calculation.

    Args:
        quantumstate (list[str] or str): Set of index names for each basis state.

    Example:
       >>> basis = Basis(["spin", "orbital", "site"])
       >>> basis = Basis("orbital")
    """

    def __init__(self, quantumnumber):
        try:
            iter(quantumnumber)
        except TypeError:
            quantumnumber = [quantumnumber]
        self._quantumnumber = quantumnumber
        self.basis = []
        self.index = {}
        self.state = namedtuple("state", self._quantumnumber)

    def add(self, *args):
        """
        Add basis to current list.

        Args:
            *args : names of quantumnumbers


        The size should match to that of quantumnumber.

        Example:
            >>> bs = Basis(["spin", "orbital", "site"])
            >>> bs.add(0, "px", (0,0,1))
            >>> bs.add(1, "s", (1,1,1))
            >>> print(bs.basis)
            [state(spin=0, orbital='px', site=(0, 0, 1)), state(spin=1, orbital='s', site=(1, 1, 1))]
        """
        try:
            newbasis = self.state(*args)
        except TypeError:
            print(traceback.format_exc())
            exit("[Error] length of basis and args does not match.\n"
                 "Please check the number of arguments.")
        self.basis.append(newbasis)
        self.index[args] = len(self.basis)-1
    
    def getindex(self, *args):
        return self.index[args]

    def getfieldname(self):
        """
        Returns:
            list[str]: Name of each field
        """
        return self.state._fields

    def getdimension(self):
        """
        Returns:
            int: Number of basis funcitons
        """
        return len(self.basis)

    def swap(self, i, j):
        """
        Exchange the order between two basis

        Args:
            i (int): First index
            j (int): Second index
        """
        self.basis[i], self.basis[j] = self.basis[j], self.basis[i]

    def permute(self, permutation):
        """
        Exchange the order using permutation index.

        Args:
            permutation (list[int]): Permutation index

        Example:
            >>> b = Basis("orbitals")
            >>> b.add("px")
            >>> b.add("s")
            >>> b.add("dxy")
            >>> print(b.basis)
            [state(orbitals='px'), state(orbitals='s'), state(orbitals='dxy')]
            >>> b.permute([0,2,1])
            >>> print(b.basis)
            [state(orbitals='px'), state(orbitals='dxy'), state(orbitals='s')]
        """
        if len(permutation) != self.getdimension():
            exit("[Error] The length of permutation array does not match\n"
                 "to your basis dimension")
        else:
            self.basis = [self.basis[i] for i in permutation]

class Openmx:
    def __init__(self, mxscfout):
        self.basis = None
        self._set_basis(mxscfout)
    
    def _set_basis(self, mxscfout):
        orbitalorder = {
            's': ['s'],
            'p': ['px', 'py', 'pz'],
            'd': ['dz2', 'dx2', 'dxy', 'dxz', 'dyz'],
            'f': ['z2', 'xz', 'yz', 'zx', 'xyz', 'x3', 'y3']
        }

        mxscfout.readfile()
        inputfile = mxscfout.inputfile
        data = tbtool.io.read_openmx_input(inputfile)
        atom_count = int(data['atoms.number'])
        spin_polarization = data['scf.spinpolarization']
        if spin_polarization == 'on' or spin_polarization == 'off':
            self.basis = Basis(['atom', 'orbital'])
            spinpol = 1
        elif spin_polarization == 'nc':
            self.basis = Basis(['spin', 'atom', 'orbital'])
            spinpol = 2


        basisorbitals = {}
        atoms = []

        orbital_species = data['definition.of.atomic.species']
        for orbital in orbital_species:
            species, orb = orbital.split()[:2] #[1].split('-')[:2]
            basisorbitals[species] = orb.split('-')[1]
        
        atoms_coord = data['atoms.speciesandcoordinates']
        for atoms_coord in atoms_coord:
            atoms.append(atoms_coord.split()[1])
        
        if spin_polarization == 'on' or spin_polarization == 'off':
            for i, atom in enumerate(atoms):
                basisorbital = basisorbitals[atom]
                for k, orbital in enumerate(basisorbital[::2]):
                    for n in range(int(basisorbital[2*k+1])):
                        for orbital_angular in orbitalorder[orbital]:
                            orbname = f'{n+1}{orbital_angular}'
                            self.basis.add(i+1, orbname)
        if spin_polarization == 'nc':
            for spin in ['u', 'd']:
                for i, atom in enumerate(atoms):
                    basisorbital = basisorbitals[atom]
                    for k, orbital in enumerate(basisorbital[::2]):
                        for n in range(int(basisorbital[2*k+1])):
                            for orbital_angular in orbitalorder[orbital]:
                                orbname = f'{n+1}{orbital_angular}'
                                self.basis.add(spin, i+1, orbname)
    def get_projector(self, wavefunctions):
        result = np.zeros((len(wavefunctions), self.basis.getdimension()))
        for i, wavefunction in enumerate(wavefunctions):
            orbital = wavefunction[:-1]
            coeff = wavefunction[-1]
            result[i, self.basis.getindex(*orbital)] = coeff
        return result

    
#def read_openmx_input(inp, inputtype="file"):
#    if inputtype == 'file':
#        file_input = Path(inp)
#        logger.info(f'Checking {file_input.absolute()} exists...')
#        if file_input.is_file():
#            logger.info(f'Found : {file_input.absolute()}')
#        else:
#            logger.info(f'Not found : {file_input.absolute()}')
#            logger.info('Check your path for .scfout file.')
#            exit(1)
#        data = file_input.read_text()
#    elif inputtype == 'string':
#        data = inp
#
#    inputdata = {}
#
#    # skip <AAAA.BBB.CC ~~~~ AAAA.BBB.CC> block.
#    # Read only X.X.X ooo lines.
#    isblock = False
#    for line in data.rsplit('\n'):
#        line = line.strip()
#
#        if line and line[0] != "#":
#            if line[0] == "<":
#                key = line.split("<", 1)[1].split("#",1)[0].rstrip().lower()
#                blocktmp = []
#                isblock = True
#                continue
#            if isblock and line[-1] == ">":
#                isblock = False
#                inputdata[key] = blocktmp
#                continue
#            elif isblock:
#                val = line.split("#", 1)[0]
#                blocktmp.append(val)
#                continue
#
#            if not isblock:
#                keyval = line.split("#", 1)[0].split()
#                inputdata[keyval[0].lower()] = keyval[1:][0]
#    return inputdata
#    
#

#        self.inputfile = tbtool.io.read_openmx_input(inputfile, inputtype='string')
import sys
import logging
import struct
from pathlib import Path
import numpy as np
import tbtool.hamiltonian as hamiltonian
import tbtool.basis as basis
import tbtool.unit as unit

OPENMX_LATEST_VERSION = 3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s - %(name)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)

def read_hwr(filepath):
    file_hwr = Path(sys.argv[0]).parent / filepath

    if not file_hwr.is_file():
        print('Wrong path for HWR file')
        print('Input path : {}'.format(file_hwr))
        sys.exit(0)

    with open(file_hwr, 'r') as fp:
        lines = fp.readlines()
        basis_count = int(lines[1].rsplit()[-1])
        basisinfo_count = basis_count * basis_count + 1
        neighbor_count = int((len(lines)-9)/basisinfo_count)

        hopping = np.zeros(
            [neighbor_count, basis_count, basis_count], dtype=np.complex
        )
        cell = np.zeros(
            [neighbor_count, 3], dtype=np.int
        )
        degeneracy = np.zeros(
            [neighbor_count], dtype=np.int
        )

        a1 = np.asarray(lines[4].rsplit(), dtype=float)
        a2 = np.asarray(lines[5].rsplit(), dtype=float)
        a3 = np.asarray(lines[6].rsplit(), dtype=float)

        chemp = np.asarray(lines[8].rsplit()[2], dtype=float)

        unitcell = np.array([a1, a2, a3]) * \
            unit.get_conversion_factor('length', 'bohr', 'angstrom')

        for n, line in enumerate(lines[9:]):
            line = line.rsplit()
            n = n // basisinfo_count
            if line[0] == 'R':
                cell[n, :] = np.asarray(line[2:5], dtype=int)
                degeneracy[n] = np.asarray(line[-1], dtype=int)
            else:
                hopping[n, int(line[0])-1, int(line[1])-1] \
                    = float(line[2]) + 1j * float(line[3])
        return hamiltonian.Wannier(hopping, cell, filename=file_hwr.resolve(), chemp=chemp)


class ByteReader(object):
    BYTESIZE = {"int" : 4,
                "double" : 8,
                "str" : 1}
   
    def __init__(self, bytedata, endian="little"):
        self.data = bytedata
        self.currentbyte = 0
        self.endian = endian
    
    @property
    def endian(self):
        return self._endian
    
    @endian.setter
    def endian(self, value):
        if value == "little" :
            self._endian = value
            self._dict_endian = {
                "int" : "<i4",
                "double" : "<f8",
                "str" :  "<256c"
            }
        elif value == "big" :
            self._dict_endian = {
                "int" : ">i4",
                "double" : ">f8",
                "str" :  ">256c"
            }
    
    def read(self, dtype, multiplier):
        dtype_converted = self._dict_endian[dtype]
        readbyte = ByteReader.BYTESIZE[dtype] * multiplier
        if dtype == "str":
            inpstr = struct.unpack(dtype_converted,self.data[self.currentbyte: self.currentbyte + readbyte])
            inpstr_decoded=[x.decode('utf-8','ignore') for x in inpstr]
            tmpread = ((''.join(inpstr_decoded)).partition('\n')[0])
        else:
            tmpread = np.fromstring(self.data[self.currentbyte: self.currentbyte + readbyte], dtype=dtype_converted)
        self.currentbyte = self.currentbyte + readbyte
        return tmpread

# The behaviour os SpinP_switch has been changed since OpenMX 3.9
# SpinP_switch -> SpinP_switch + 4*version, version= 0 ~ 3

def read_openmx_hamiltonian(scfout, version, endian="little", pathtype="relative", spin=None):
    file_scfout = Path(sys.argv[0]).parent / scfout
    if version == 3.9:
        return hamiltonian.Openmx(MXscfoutV3(file_scfout, endian=endian, pathtype=pathtype), spin=spin)
    elif version == 3.8:
        return hamiltonian.Openmx(MXscfoutV0(file_scfout, endian=endian, pathtype=pathtype))

def read_openmx_basis(scfout, version, endian="little", pathtype="relative"):
    file_scfout = Path(sys.argv[0]).parent / scfout
    if version == 3.9:
        return basis.Openmx(MXscfoutV3(file_scfout, endian=endian, pathtype=pathtype))
    elif version == 3.8:
        return basis.Openmx(MXscfoutV0(file_scfout, endian=endian, pathtype=pathtype))

class MXscfoutBase(object):
    
    def __init__(self, scfout, endian="little", pathtype="relative"):
        self._scfout = scfout
        self.endian = endian
        self.scfoutversion = None 
        self.pathtype = pathtype
        self.atomnum = None
        self.SpinP_switch = None
        self.Catomnum = None
        self.Latomnum = None
        self.Ratomnum = None
        self.TCpyCell = None
        self.atv = None
        self.atv_ijk = None
        self.Total_NumOrbs = None
        self.FNAN = None 
        self.natn = None
        self.ncn = None
        self.tv = None
        self.rtv = None
        self.Gxyz = None
        self.Hks = None
        self.iHks = None
        self.OLP = None
        self.DM = None
        self.Solver = None
        self.ChemP = None
        self.E_Temp = None
        self.dipole_moment_core = None
        self.dipole_moment_background = None
        self.Valence_Electrons = None
        self.Total_SpinS = None
        self.inputfile = []
   
    @property
    def scfout(self):
        return self._scfout
    
    @scfout.setter
    def scfout(self, scfout):
        logger.info(f'Changing input file path to {scfout}.')
        self._scfout = scfout
        self.scfoutversion = None 
        self.atomnum = None
        self.SpinP_switch = None
        self.Catomnum = None
        self.Latomnum = None
        self.Ratomnum = None
        self.TCpyCell = None
        self.atv = None
        self.atv_ijk = None
        self.Total_NumOrbs = None
        self.FNAN = None 
        self.natn = None
        self.ncn = None
        self.tv = None
        self.rtv = None
        self.Gxyz = None
        self.Hks = None
        self.iHks = None
        self.OLP = None
        self.DM = None
        self.Solver = None
        self.ChemP = None
        self.E_Temp = None
        self.dipole_moment_core = None
        self.dipole_moment_background = None
        self.Valence_Electrons = None
        self.Total_SpinS = None
        self.inputfile = []
    
    def chkfile(self):
        """Check if file exists.
        
        """
 
        file_scfout = Path(self.scfout)
        logger.info(f'Checking {file_scfout.absolute()} exists...')
        if file_scfout.is_file():
            logger.info("Found %s .", file_scfout.absolute())
        else:
            logger.info("File %s not found.", file_scfout.absolute())
            logger.info("Check your path for .scfout file.")
            exit()
    
    def readfile(self):
        pass

    def get_hamiltonian(self):
        logger.info("Generating Hamiltonian from scfout...")
        logger.info("\u2500\u2500\u2500\u2500\u2500\u2500 Calculation Details \u2500\u2500\u2500\u2500\u2500\u2500")
        #logger.info("Spin Switch : %d, %s", self.SpinP_switch, constant.SPINP_SWITCH[self.SpinP_switch])

        # idx_basis : Impose unique number (basis index) to each orbital.
        # ex) (atom 1, s orbital)  -> 1,
        #     (atom 1, px orbital) -> 2,
        #     (atom 1, py orbital) -> 3,
        #     (atom 2, s orbital)  -> 4, 
        #             ....
        idx_basis= {}

        # Determine dimension of Hamiltonian
        # If SpinP_switch == 0 or 1 (nonmagnetic or spin-polarized calc),
        # the dimension equals to the total number of orbitals.
        # If SpinP_switch == 3 (non-collinear),
        # the dimension equals double of the total number of orbitals.
        dimension = 0
        if self.SpinP_switch in (0, 1):
            for ct_AN in range(self.atomnum): # ct_AN = 0 ~ atomnum-1
                TNO1 = self.Total_NumOrbs[ct_AN]
                for i in range(TNO1):
                    idx_basis[(ct_AN, i)] = dimension
                    dimension += 1
        elif self.SpinP_switch == 3:
            for spin in range(2):
                for ct_AN in range(self.atomnum): # ct_AN = 0 ~ atomnum-1
                    TNO1 = self.Total_NumOrbs[ct_AN]
                    for i in range(TNO1):
                        idx_basis[(spin, ct_AN, i)] = dimension
                        dimension += 1

        if self.SpinP_switch == 0:
            # self.ncn = Global cell index where overlap != 0.
            # effective_cell = tabulate all global index where overlap is non-zeor
            effective_cell = np.array(np.unique(self.ncn), dtype=int)

            ham = np.zeros([effective_cell.shape[0], dimension, dimension], dtype=np.complex)
            olp = np.zeros([effective_cell.shape[0], dimension, dimension], dtype=np.complex)
            cell = np.zeros([effective_cell.shape[0], 3])

            cell[1:,:] = self.atv_ijk[effective_cell[1:] -1 ,:]

            for ct_AN in range(self.atomnum): # ct_AN = 0 ~ atomnum-1
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1): # h_AN = Total number of neighbor
                    Gh_AN = self.natn[ct_AN][h_AN] # Gh_AN = neighboring atom index -> 0, 1, 2, ...
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    basis_idx = np.zeros([TNO1*TNO2, 2], dtype=int)
                    count = 0
                    cell_idx, = np.where(effective_cell==self.ncn[ct_AN][h_AN])[0]
                    for i in range(TNO1):
                        for j in range(TNO2):
                            basis1 = idx_basis[(ct_AN, i)]
                            basis2 = idx_basis[(Gh_AN, j)]
                            basis_idx[count,:] =  int(basis1), int(basis2)
                            count+=1

                    ham[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.Hks[0][ct_AN][h_AN]).flatten()
                    olp[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.OLP[ct_AN][h_AN]).flatten()

        elif self.SpinP_switch == 1:
            # self.ncn = Global cell index where overlap != 0.
            # effective_cell = tabulate all global index where overlap is non-zeor
            effective_cell = np.array(np.unique(np.concatenate(self.ncn)), dtype=int)

            ham = np.zeros([2, effective_cell.shape[0], dimension, dimension], dtype=np.complex)
            olp = np.zeros([effective_cell.shape[0], dimension, dimension], dtype=np.complex)
            cell = np.zeros([effective_cell.shape[0], 3])

            cell[1:,:] = self.atv_ijk[effective_cell[1:] -1 ,:]

            for spin in range(2):
                for ct_AN in range(self.atomnum): # ct_AN = 0 ~ atomnum-1
                    TNO1 = self.Total_NumOrbs[ct_AN]
                    for h_AN in range(self.FNAN[ct_AN]+1): # h_AN = Total number of neighbor
                        Gh_AN = self.natn[ct_AN][h_AN] # Gh_AN = neighboring atom index -> 0, 1, 2, ...
                        TNO2 = self.Total_NumOrbs[Gh_AN]
                        basis_idx = np.zeros([TNO1*TNO2, 2], dtype=int)
                        count = 0
                        cell_idx, = np.where(effective_cell==self.ncn[ct_AN][h_AN])[0]
                        for i in range(TNO1):
                            for j in range(TNO2):
                                basis1 = idx_basis[(ct_AN, i)]
                                basis2 = idx_basis[(Gh_AN, j)]
                                basis_idx[count,:] =  int(basis1), int(basis2)
                                count+=1

                        ham[spin, cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.Hks[spin][ct_AN][h_AN]).flatten()
                        if spin == 0:
                            olp[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.OLP[ct_AN][h_AN]).flatten()

        elif self.SpinP_switch == 3:
            # self.ncn = Global cell index where overlap != 0.
            # effective_cell = tabulate all global index where overlap is non-zeor
            effective_cell = np.array(np.unique(np.concatenate(self.ncn)), dtype=int)

            ham = np.zeros([effective_cell.shape[0], dimension, dimension], dtype=np.complex)
            olp = np.zeros([effective_cell.shape[0], dimension, dimension], dtype=np.complex)
            cell = np.zeros([effective_cell.shape[0], 3])

            cell[1:,:] = self.atv_ijk[effective_cell[1:] -1 ,:]

            for spin in range(2):
                for ct_AN in range(self.atomnum): # ct_AN = 0 ~ atomnum-1
                    TNO1 = self.Total_NumOrbs[ct_AN]
                    for h_AN in range(self.FNAN[ct_AN]+1): # h_AN = Total number of neighbor
                        Gh_AN = self.natn[ct_AN][h_AN] # Gh_AN = neighboring atom index -> 0, 1, 2, ...
                        TNO2 = self.Total_NumOrbs[Gh_AN]
                        basis_idx = np.zeros([TNO1*TNO2, 2], dtype=int)
                        count = 0
                        cell_idx, = np.where(effective_cell==self.ncn[ct_AN][h_AN])[0]
                        for i in range(TNO1):
                            for j in range(TNO2):
                                basis1 = idx_basis[(spin, ct_AN, i)]
                                basis2 = idx_basis[(spin, Gh_AN, j)]
                                basis_idx[count,:] =  int(basis1), int(basis2)
                                count+=1

                        ham[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.Hks[spin][ct_AN][h_AN]).flatten()
                        ham[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.iHks[spin][ct_AN][h_AN]).flatten() * 1j
                        olp[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.OLP[ct_AN][h_AN]).flatten()
                        
            for spin in range(2,3):
                for ct_AN in range(self.atomnum): # ct_AN = 0 ~ atomnum-1
                    TNO1 = self.Total_NumOrbs[ct_AN]
                    for h_AN in range(self.FNAN[ct_AN]+1): # h_AN = Total number of neighbor
                        Gh_AN = self.natn[ct_AN][h_AN] # Gh_AN = neighboring atom index -> 0, 1, 2, ...
                        TNO2 = self.Total_NumOrbs[Gh_AN]
                        basis_idx = np.zeros([TNO1*TNO2, 2], dtype=int)
                        count = 0
                        cell_idx, = np.where(effective_cell==self.ncn[ct_AN][h_AN])[0]
                        for i in range(TNO1):
                            for j in range(TNO2):
                                basis1 = idx_basis[(0, ct_AN, i)]
                                basis2 = idx_basis[(1, Gh_AN, j)]
                                basis_idx[count,:] =  int(basis1), int(basis2)
                                count+=1

                        ham[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.Hks[spin][ct_AN][h_AN]).flatten()
                        ham[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.Hks[3][ct_AN][h_AN]).flatten() * 1j
                        ham[cell_idx, basis_idx[:,0], basis_idx[:,1]] += np.array(self.iHks[spin][ct_AN][h_AN]).flatten() * 1j

            dim2 = int(dimension/2)
            ham[:, dim2:, 0:dim2] = np.conjugate(np.transpose(ham[:, 0:dim2, dim2:], axes=[0,2,1]))
            olp[:, dim2:, 0:dim2] = np.conjugate(np.transpose(olp[:, 0:dim2, dim2:], axes=[0,2,1]))

        return ham, olp, cell, dimension, self.ChemP
    
    def chkversion(self, mxversion):
        if mxversion <= 3.8:
            scfouttype = 0
        elif mxversion >= 3.9:
            scfouttype = 3

        logger.info("Checking OpenMX and .scfout version ...")
        logger.info("Input parameters")
        logger.info("{:>12s} : {:>8s} {:>12s} : {:>3s}"\
            .format("OpenMX", str(mxversion), "scfout", str(scfouttype)))
        logger.info("Estimated OpenMX and scfout version from .scfout file")
        if self.scfoutversion == 0:
            logger.info("{:>12s} : {:>8s} {:>12s} : {:>3s}"\
                .format("OpenMX", "<=3.8", "scfout", str(0)))
        elif self.scfoutversion == 1:
            logger.info("{:>12s} : {:>8s} {:>12s} : {:>3s}"\
                .format("OpenMX", "3.7.x", "scfout", str(1)))
        elif self.scfoutversion == 2:
            logger.info("{:>12s} : {:>8s} {:>12s} : {:>3s}"\
                .format("OpenMX", "3.7.x", "scfout", str(2)))
        elif self.scfoutversion == 3:
            logger.info("{:>12s} : {:>8s} {:>12s} : {:>3s}"\
                .format("OpenMX", ">=3.9", "scfout", str(3)))

        if self.scfoutversion == scfouttype:
            logger.info("Version matched.")
        else:
            logger.error("OpenMX and scfout versions not matched.")
            logger.error("Check your .scfout file and OpenMX version")
            exit(1)

class MXscfoutV3(MXscfoutBase):
    """scfout reader for OpenMX 3.9
    
    """
    
    def __init__(self, scfout, endian="little", pathtype="relative"):
        """Test
        
        Parameters
        ----------
        file : [type]
            [description]
        pathtype : str, optional
            [description], by default "relative"
        """
        super(MXscfoutV3, self).__init__(scfout=scfout, endian=endian, pathtype=pathtype)
        self.OLPpo = None
        self.OLPmo = None
        self.iDM = None
        self.MXVERSION=3.9

    def readfile(self, write_input=False):
        self.chkfile()
        file_scfout = Path(self.scfout)
        data = file_scfout.read_bytes()
        bytereader = ByteReader(data, self.endian)

        res = bytereader.read("int", 6)

        self.atomnum, tmpval, self.Catomnum, self.Latomnum, self.Ratomnum, self.TCpyCell = res
        self.SpinP_switch = tmpval % 4
        self.scfoutversion = int(tmpval / 4)
        
        self.chkversion(self.MXVERSION)

        # Check Endian 
        # Todo : Checking endian using raw value is not a strict way of doing this.
        if (tmpval < 0 or tmpval > OPENMX_LATEST_VERSION * 4 + 3):
            logger.error("\u2500\u2500\u2500\u2500\u2500 ERROR \u2500\u2500\u2500\u2500\u2500")
            logger.error("Mismatch of endians.")
            logger.error("Input endian : {}".format(self.endian))
            exit(1)

        self.order_max = bytereader.read("int", 1)[0]

        self.atv = np.zeros([self.TCpyCell, 3], dtype="double")
        bytereader.read("double", 4) # skip 4 byte (dummy byte)
        for i in range(self.TCpyCell):
            res = bytereader.read("double", 4) 
            self.atv[i,:] = res[1:]
        
        self.atv_ijk = np.zeros([self.TCpyCell, 3], dtype="int")
        res = bytereader.read("int", 4) # skip 4 bytes (dummy bytes)
        for i in range(self.TCpyCell):
            res = bytereader.read("int", 4)
            self.atv_ijk[i,:] = res[1:]

        self.Total_NumOrbs = np.zeros(self.atomnum, dtype="int")
        self.Total_NumOrbs[:] = bytereader.read("int", self.atomnum)

        self.FNAN = np.zeros(self.atomnum, dtype="int")
        self.FNAN[:] = bytereader.read("int", self.atomnum)

        self.natn = []
        for i in range(self.atomnum):
            res = bytereader.read("int", self.FNAN[i]+1) - 1
            self.natn.append(res)

        self.ncn = []
        for i in range(self.atomnum):
            res = bytereader.read("int", self.FNAN[i]+1)
            self.ncn.append(res)
        
        self.tv = np.zeros([3,3])
        res = bytereader.read("double", 12)
        self.tv[:,:] = res.reshape((3,4))[:,1:]

        self.rtv = np.zeros([3,3])
        res = bytereader.read("double", 12)
        self.rtv[:,:] = res.reshape((3,4))[:,1:]

        ####################################
        #    double Gxyz[][1-3]:           #
        #    atomic coordinates in Bohr    #
        ####################################
        self.Gxyz = np.zeros([self.atomnum+1, 60])
        res = bytereader.read("double", self.atomnum * 4)
        self.Gxyz[1:,0:4] = res.reshape([self.atomnum, 4])

        self.Hks = []
        for spin in range(self.SpinP_switch+1):
            self.Hks.append([])
            for ct_AN in range(self.atomnum):
                self.Hks[spin].append([])
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1):
                    self.Hks[spin][ct_AN].append([])
                    Gh_AN = self.natn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        res = bytereader.read("double", TNO2)
                        self.Hks[spin][ct_AN][h_AN].append(res)

        self.iHks = []
        for spin in range(3):
            self.iHks.append([])
            for ct_AN in range(self.atomnum):
                self.iHks[spin].append([])
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1):
                    self.iHks[spin][ct_AN].append([])
                    Gh_AN = self.natn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        if self.SpinP_switch == 3: 
                            res = bytereader.read("double", TNO2)
                            self.iHks[spin][ct_AN][h_AN].append(res)
                        else:
                            res = np.zeros(TNO2, dtype="double")
                            self.iHks[spin][ct_AN][h_AN].append(res)

        self.OLP = []
        for ct_AN in range(self.atomnum):
            self.OLP.append([])
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1):
                self.OLP[ct_AN].append([])
                Gh_AN = self.natn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    res = bytereader.read("double", TNO2)
                    self.OLP[ct_AN][h_AN].append(res)

        self.OLPpo = []
        for direction in range(3):
            self.OLPpo.append([])
            for order in range(self.order_max):
                self.OLPpo[direction].append([])
                for ct_AN in range(self.atomnum):
                    self.OLPpo[direction][order].append([])
                    TNO1 = self.Total_NumOrbs[ct_AN]
                    for h_AN in range(self.FNAN[ct_AN]+1):
                        self.OLPpo[direction][order][ct_AN].append([])
                        Gh_AN = self.natn[ct_AN][h_AN]
                        TNO2 = self.Total_NumOrbs[Gh_AN]
                        for i in range(TNO1):
                            res = bytereader.read("double", TNO2)
                            self.OLPpo[direction][order][ct_AN][h_AN].append(res)

        self.OLPmo = []
        for direction in range(3):
            self.OLPmo.append([])
            for order in range(self.order_max):
                self.OLPmo[direction].append([])
                for ct_AN in range(self.atomnum):
                    self.OLPmo[direction][order].append([])
                    TNO1 = self.Total_NumOrbs[ct_AN]
                    for h_AN in range(self.FNAN[ct_AN]+1):
                        self.OLPmo[direction][order][ct_AN].append([])
                        Gh_AN = self.natn[ct_AN][h_AN]
                        TNO2 = self.Total_NumOrbs[Gh_AN]
                        for i in range(TNO1):
                            res = bytereader.read("double", TNO2)
                            self.OLPmo[direction][order][ct_AN][h_AN].append(res)

        self.DM = []
        for spin in range(self.SpinP_switch+1):
            self.DM.append([])
            for ct_AN in range(self.atomnum):
                self.DM[spin].append([])
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1):
                    self.DM[spin][ct_AN].append([])
                    Gh_AN = self.natn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        res = bytereader.read("double", TNO2)
                        self.DM[spin][ct_AN][h_AN].append(res)

        self.iDM = []
        for spin in range(2):
            self.iDM.append([])
            for ct_AN in range(self.atomnum):
                self.iDM[spin].append([])
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1):
                    self.iDM[spin][ct_AN].append([])
                    Gh_AN = self.natn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        res = bytereader.read("double", TNO2)
                        self.iDM[spin][ct_AN][h_AN].append([])

        self.Solver = bytereader.read("int", 1)[0]

        res = bytereader.read("double", 10)
        self.ChemP = res[0]
        self.E_Temp = res[1]
        self.dipole_moment_core = res[2:5]
        self.dipole_moment_background = res[5:8]
        self.Valence_Electrons = res[8]
        self.Total_SpinS = res[9]

        ####################################
        #    Original input file of calc   #
        ####################################
        file_scfout = Path(self.scfout)
        inputfile="{}".format(file_scfout.absolute()).split(".")[0]+".input"

        res = bytereader.read("int", 1)[0]  # res = Total number of lines of the input file.
        with open(inputfile, "w") as fp:
            for i in range(res):
                res = bytereader.read("str", 256)
                if write_input:
                    fp.write(res + '\n')
                self.inputfile.append(res)

class MXscfoutV0(MXscfoutBase):
    """scfout reader for OpenMX <= 3.8
    
    """
    
    def __init__(self, scfout, endian="little", pathtype="relative"):
        """Test
        
        Parameters
        ----------
        file : [type]
            [description]
        pathtype : str, optional
            [description], by default "relative"
        """
        super(MXscfoutV0, self).__init__(scfout=scfout, endian=endian, pathtype=pathtype)
        self.OLPpox = None
        self.OLPpoy = None
        self.OLPpoz = None
        self.MXVERSION = 3.8

    def readfile(self, write_input=False):
        self.chkfile()
        file_scfout = Path(self.scfout)
        data = file_scfout.read_bytes()
        bytereader = ByteReader(data)

        res = bytereader.read("int", 6)
        self.atomnum, self.SpinP_switch, self.Catomnum, self.Latomnum, self.Ratomnum, self.TCpyCell = res
        self.scfoutversion = int(self.SpinP_switch / 4)

        self.chkversion(self.MXVERSION)

        self.atv = np.zeros([self.TCpyCell, 3], dtype="double")
        bytereader.read("double", 4) # Skip dummy line
        for i in range(self.TCpyCell):
            res = bytereader.read("double", 4)
            self.atv[i,:] = res[1:]

        self.atv_ijk = np.zeros([self.TCpyCell, 3], dtype="int")
        bytereader.read("int", 4)
        for i in range(self.TCpyCell):
            res = bytereader.read("int", 4)
            self.atv_ijk[i,:] = res[1:]

        self.Total_NumOrbs = np.zeros(self.atomnum, dtype="int")
        self.Total_NumOrbs[:] = bytereader.read("int", self.atomnum)

        self.FNAN = np.zeros(self.atomnum, dtype="int")
        self.FNAN[:] = bytereader.read("int", self.atomnum)

        self.natn = []
        for i in range(self.atomnum):
            res = bytereader.read("int", self.FNAN[i]+1) - 1
            self.natn.append(res)

        self.ncn = []
        for i in range(self.atomnum):
            res = bytereader.read("int", self.FNAN[i]+1)
            self.ncn.append(res)
        
        self.tv = np.zeros([3,3])
        res = bytereader.read("double", 12)
        self.tv[:,:] = res.reshape((3,4))[:,1:]

        self.rtv = np.zeros([3,3])
        res = bytereader.read("double", 12)
        self.rtv[:, :] = res.reshape((3, 4))[:,1:]

        self.Gxyz = np.zeros([self.atomnum+1, 60])
        res = bytereader.read("double", self.atomnum * 4)
        self.Gxyz[1:,0:4] = res.reshape([self.atomnum, 4])

        self.Hks = []
        for spin in range(self.SpinP_switch+1):
            self.Hks.append([])
            for ct_AN in range(self.atomnum):
                self.Hks[spin].append([])
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1):
                    self.Hks[spin][ct_AN].append([])
                    Gh_AN = self.natn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        res = bytereader.read("double", TNO2)
                        self.Hks[spin][ct_AN][h_AN].append(res)

        self.iHks = []
        for spin in range(3):
            self.iHks.append([])
            for ct_AN in range(self.atomnum):
                self.iHks[spin].append([])
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1):
                    self.iHks[spin][ct_AN].append([])
                    Gh_AN = self.natn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        if self.SpinP_switch == 3: 
                            res = bytereader.read("double", TNO2)
                            self.iHks[spin][ct_AN][h_AN].append(res)
                        else:
                            res = np.zeros(TNO2, dtype="double")
                            self.iHks[spin][ct_AN][h_AN].append(res)

        self.OLP = []
        for ct_AN in range(self.atomnum):
            self.OLP.append([])
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1):
                self.OLP[ct_AN].append([])
                Gh_AN = self.natn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    res = bytereader.read("double", TNO2)
                    self.OLP[ct_AN][h_AN].append(res)

        self.OLPpox = []
        for ct_AN in range(self.atomnum):
            self.OLPpox.append([])
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1):
                self.OLPpox[ct_AN].append([])
                Gh_AN = self.natn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    res = bytereader.read("double", TNO2)
                    self.OLPpox[ct_AN][h_AN].append(res)

        self.OLPpoy = []
        for ct_AN in range(self.atomnum):
            self.OLPpoy.append([])
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1):
                self.OLPpoy[ct_AN].append([])
                Gh_AN = self.natn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    res = bytereader.read("double", TNO2)
                    self.OLPpoy[ct_AN][h_AN].append(res)

        self.OLPpoz = []
        for ct_AN in range(self.atomnum):
            self.OLPpoz.append([])
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1):
                self.OLPpoz[ct_AN].append([])
                Gh_AN = self.natn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    res = bytereader.read("double", TNO2)
                    self.OLPpoz[ct_AN][h_AN].append(res)

        self.DM = []
        for spin in range(self.SpinP_switch+1):
            self.DM.append([])
            for ct_AN in range(self.atomnum):
                self.DM[spin].append([])
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1):
                    self.DM[spin][ct_AN].append([])
                    Gh_AN = self.natn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        res = bytereader.read("double", TNO2)
                        self.DM[spin][ct_AN][h_AN].append(res)

        self.Solver = bytereader.read("int", 1)

        res = bytereader.read("double", 10)
        self.ChemP = res[0]
        self.E_Temp = res[1]
        self.dipole_moment_core = res[2:5]
        self.dipole_moment_background = res[5:8]
        self.Valence_Electrons = res[8]
        self.Total_SpinS = res[9]

        ####################################
        #    Original input file of calc   #
        ####################################
        file_scfout = Path(self.scfout)
        inputfile="{}".format(file_scfout.absolute()).split(".")[0]+".input"

        res = bytereader.read("int", 1)[0]
        with open(inputfile, "w") as fp:
            for i in range(res):
                res = bytereader.read("str", 256)
                if write_input:
                    fp.write(res + '\n')
                self.inputfile.append(res)

def read_openmx_input(data):

    inputdata = {}

    # skip <AAAA.BBB.CC ~~~~ AAAA.BBB.CC> block.
    # Read only X.X.X ooo lines.
    isblock = False
    for line in data:
        line = line.strip()

        if line and line[0] != "#":
            if line[0] == "<":
                key = line.split("<", 1)[1].split("#",1)[0].rstrip().lower()
                blocktmp = []
                isblock = True
                continue
            if isblock and line[-1] == ">":
                isblock = False
                inputdata[key] = blocktmp
                continue
            elif isblock:
                val = line.split("#", 1)[0]
                blocktmp.append(val)
                continue

            if not isblock:
                keyval = line.split("#", 1)[0].split()
                inputdata[keyval[0].lower()] = keyval[1:][0]
    return inputdata

import sys
from pathlib import Path
import numpy as np
import tbtool.hamiltonian as hamiltonian
import tbtool.unit as unit


def read_hwr(filepath):
    file_hwr = Path(sys.argv[0]).parent / filepath

    if not file_hwr.is_file():
        print("Wrong path for HWR file")
        print("Input path : {}".format(file_hwr))
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

        unitcell = np.array([a1, a2, a3]) * \
            unit.get_conversion_factor("length", "bohr", "angstrom")

        for n, line in enumerate(lines[9:]):
            line = line.rsplit()
            n = n // basisinfo_count
            if line[0] == 'R':
                cell[n, :] = np.asarray(line[2:5], dtype=int)
                degeneracy[n] = np.asarray(line[-1], dtype=int)
            else:
                hopping[n, int(line[0])-1, int(line[1])-1] \
                    = float(line[2]) + 1j * float(line[3])
        return hamiltonian.Wannier(hopping, cell, filename=file_hwr.resolve())
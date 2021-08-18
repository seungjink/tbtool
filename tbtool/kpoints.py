import sys
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s - %(name)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)


def monkhorst_pack(arr, with_boundary=False, gamma_center=True):
    """Generate Monkhorst-Pack meshes.

    Args:
        arr (array_like): [k1[,k2, k3]] shaped array, list or tuple.
        with_boundary (bool, optional): 
            If true, meshes are generated including the zone boundaries.
            Default value is False.
        gamma_center (bool, optional):
            The original mesh points are geneted between [0 ~ 1].
            If gamma_center == True, the meshes are shift to [-0.5 ~ 0.5].
            Default value is True.

    Returns:
        numpy.array: [k1[, k2, k3]] shaped array.

    Example:
       >>> kpts = monkhorst_pack([2,2,1])
       >>> print(kpts)
       [[-0.25 -0.25  0.  ]
        [-0.25  0.25  0.  ]
        [ 0.25 -0.25  0.  ]
        [ 0.25  0.25  0.  ]]


    Two dimensionary mesh

       >>> kpts = monkhorst_pack([2,2])
       >>> print(kpts)
       [[-0.25 -0.25]
        [-0.25  0.25]
        [ 0.25 -0.25]
        [ 0.25  0.25]]


    Including boundary values

       >>> kpts = monkhorst_pack([2,2,1], with_boundary=True)
       >>> print(kpts)
       [[-0.5 -0.5  0.  ]
        [-0.5  0.5  0.  ]
        [ 0.5 -0.5  0.  ]
        [ 0.5  0.5  0.  ]]


    Not :math:`\Gamma=(0,0,0)` centered (mesh between 0~1)

       >>> kpts = monkhorst_pack([2,2,1], gamma_center=False)
       >>> print(kpts)
       [[0.  0.  0. ]
        [0.  0.5 0. ]
        [0.5 0.  0. ]
        [0.5 0.5 0. ]]
    """
    try:
        input_arr = np.array(arr)
        arr_shape = input_arr.shape[0]
        kpts = np.indices(input_arr).transpose(
            np.roll(np.arange(arr_shape+1), -1)).reshape((-1, arr_shape))
        arr_offset = np.zeros(arr_shape)
        arr_shiftorigin = np.zeros(arr_shape)
    except ValueError:
        logger.error('----- ERROR -----')
        logger.error(
            'Check your input grid %s matches the shape of [n,[l m]], array', arr)
        sys.exit(1)
    arr_divider = np.array(input_arr)

    if gamma_center:
        if not with_boundary:
            arr_shiftorigin = 1/2 - 1/2/arr_divider
        else:
            arr_shiftorigin[input_arr != 1] = 1/2

    if with_boundary:
        arr_divider[input_arr > 1] -= 1

    return (kpts + arr_offset) / arr_divider - arr_shiftorigin


class Kmesh:
    """
    Generate k point mesh 

    Args:
        mesh (list[int]): Mesh size
        unitcell (list, optional): Specifies reciprocal vector.
            This parameter is needed when you want to geneate mesh
            wirh respect to new cell vectors. For example, if you want
            to generate mesh with respect to new cell vectors

            .. math::
                \\begin{aligned}
                    \\bm{\\widetilde{b_1}} &= \\vec{b_1} + \\vec{b_2} \\\\
                    \\bm{\\widetilde{b_2}} &= \\vec{b_1} - \\vec{b_2} \\\\
                    \\bm{\\widetilde{b_3}} &= \\vec{b_3}
                \\end{aligned}

            , then set 

            >>> mesh = Kmesh([2,2,1], unitcell=[[1,1,0],[1,-1,0],[0,0,1]])

        with_boundary (bool, optional): Include boundary values. Defaults to False.
        gamma_center (bool, optional): Shift mesh center to origin. Defaults to True.

    Example:
       >>> mesh = Kmesh([2,2,1])
       >>> mesh = Kmesh([5])
       >>> mesh = Kmesh([5], with_boundary=True)

    """

    def __init__(self, mesh, unitcell=np.eye(3), with_boundary=False, gamma_center=True):
        self._mesh = np.array(mesh, dtype=int)
        self._unitcell = unitcell
        self.with_boundary = with_boundary
        self.gamma_center = gamma_center

        if unitcell is not None:
            self.unitcell = np.array(unitcell)
            if self.unitcell.shape != (self.mesh.shape[0], self.mesh.shape[0]):
                logger.error('----- ERROR -----')
                logger.error(
                    'Dimension of unitcell {} does not match to mesh size {}'
                    .format(self.unitcell, self.mesh)
                )
                sys.exit(1)

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = np.array(mesh, dtype=int)

    @property
    def unitcell(self):
        return self._unitcell

    @unitcell.setter
    def unitcell(self, unitcell):
        self._unitcell = np.array(unitcell, dtype=float)

    def get(self, conversion=False):
        """Return mesh array

        Args:
            conversion (bool, optional): If True, meshes are represented in unitcell basis. Defaults to False.

        Returns:
            (numpy.array): mesh array

        Example:
           >>> mesh = Kmesh([2,2])
           >>> mesh.get()
           [[-0.25,-0.25],
            [-0.25, 0.25],
            [ 0.25,-0.25],
            [ 0.25, 0.25]]

        Changing unitcell vector

           >>> mesh = Kmesh([2,2,1], unitcell=[[0,0,1],[1,0,0],[0,1,0]])
           >>> mesh.get()
           [[-0.25, 0, -0.25],
            [0.25,  0, -0.25],
            [-0.25,  0, 0.25],
            [0.25,  0, 0.25]]
        """
        kmesh = monkhorst_pack(
            self.mesh, with_boundary=self.with_boundary, gamma_center=self.gamma_center)
        if conversion:
            return np.matmul(kmesh, self.unitcell)
        else:
            return kmesh

    def get_cartesian(self):
        """Generate meshgrid in Cartesian coordinates.

        Returns
        -------
        numpy.ndarray
            (self.mesh)-shaped mesh points in Cartesian untis.
        """
        kmesh = monkhorst_pack(
            self.mesh, with_boundary=self.with_boundary, gamma_center=self.gamma_center)
        kmesh = np.matmul(kmesh, self.unitcell)
        return kmesh


class Kpath(object):
    """Mesh points for line path in reciprocal space 

    Args:
        kpts (arr_like): Set of discrete K points specifiying the lines.
        label (list[str], optional): Label name for each kpt. The shape should be the same as kpts. Defaults to None.
        unitcell (array_like, optional): Three unit cell vectors defining the Brillouin zone. Defaults to None.
    """

    def __init__(self, kpts, label=None, unitcell=None, n=30):
        self.kpts = np.array(kpts)
        self.label = label
        self.unitcell = unitcell
        self._n = None
        self.n = n

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        try:
            iter(n)
            self._n = np.array(n)
        except TypeError:
            self._n = np.array([n]*(self.kpts.shape[0]-1))

    def get(self):
        """Generate line mesh between each k points.

        Returns:
            numpy.array: An array of line mesh.
        """

        kmesh = np.zeros([np.sum(self.n), 3])
        idx = 0
        for i, steps in enumerate(self.n):
            kmesh[idx: idx + steps, 0] = \
                np.linspace(self.kpts[i, 0], self.kpts[i+1, 0], steps)
            kmesh[idx: idx + steps, 1] = \
                np.linspace(self.kpts[i, 1], self.kpts[i+1, 1], steps)
            kmesh[idx: idx + steps, 2] = \
                np.linspace(self.kpts[i, 2], self.kpts[i+1, 2], steps)
            idx += steps
        
        return kmesh
    
    def get_linear(self, label=None):
        if self.unitcell is not None:
            if np.array(self.unitcell).shape != (3, 3):
                logger.error("----- ERROR -----")
                logger.error("Shape of unitcell = %s does not match to (3,3)", np.array(unitcell))
                exit(1)
            reciprocal_cell = (np.linalg.inv(self.unitcell) * np.pi * 2).T

            kmesh = self.get()

            count_k = kmesh.shape[0]
            klinear = np.zeros(count_k, dtype=float)

            dk = kmesh[1:, :] - kmesh[:-1, :]
            dk_length = np.linalg.norm(
                np.matmul(dk, reciprocal_cell[np.newaxis, :, :])[0], axis=1)
            for i in range(1, count_k):
                klinear[i] = klinear[i-1] + dk_length[i-1]
        else:
            klinear = np.linspace(0, 1, count_k)

        labelidx = [0]
        for i in range(self.n.shape[0]):
            labelidx.append(self.n[i] + labelidx[i])
        labelidx[-1] -= 1
        return klinear, klinear[labelidx]
#        if label is not None:
#            return klinear, label
#
#        if return_label:
#            klabel = self.label
#            idx_klabel = np.zeros(self.kpts.shape[0], dtype=int)
#            for i, dk in enumerate(n):
#                idx_klabel[i+1] = idx_klabel[i] + dk
#            idx_klabel[1:] -= 1
#            return kmesh, klinear, klabel, idx_klabel
#        else:
#            return kmesh, klinear
#
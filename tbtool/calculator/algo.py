import functools
import logging
import numpy as np
import tbtool.kpoints as kp
from scipy.spatial import Delaunay

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s - %(name)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)

def integrate_delta_2d_tetra(kmesh, bandenergy, fn, w):
    """Integrate delta function by 2d tetrahecron method.

    This function calculates the following delta funciton integral on the Brillouin zone using 2d tetrahedron method.

    .. math::
        \\begin{aligned}
            g(\\omega) &= \\sum_{n=1}^{band} \\frac{1}{V_G} \\int_{BZ} d^2 k f_{n\\bm{k}}(\\omega_{n\\bm{k}}) \delta(\\omega - \\omega_{n\\bm{k}}) \\\\
                       &= \\sum_{\\tau}^{N_{\\tau}} f_{\\tau}(\omega) D_{\\tau}(\\omega)
        \\end{aligned}

    Args:
        kmesh (numpy.ndarray): [Nmesh, 2] shaped mesh array.
        bandenergy (numpy.ndarray): [Nmesh, Nband] shaped band energy.
        fn (numpy.ndarray): [Nmesh, Nband] shaped function value.
        w (float or list): function values to be calculated.

    """

    # If function argument w is single number (not array-like)
    # convert it to numpy array.
    # Then we can treat w as we do for array-like case.
    try:
        iter(w)
    except TypeError:
        w = np.array([w])

    # Convert input fn to [N_fn, ]
    if fn.shape == bandenergy.shape:
        fn = np.expand_dims(fn, axis=0)

    band_count = bandenergy.shape[1]
    fn_vals_count = fn.shape[2]
    fn_count = fn.shape[0]
    kmesh_count = kmesh.shape
    w_count = w.shape[0]

    # Check whether the kmesh is 2d grid
    if kmesh.shape[1] != 2:
        logger.error('----- ERROR -----')
        logger.error('Input kpoint mesh does look as 2d grid.')
        logger.error(f'Shape of Input mesh: {kmesh_count}')
        logger.error('Check the shape of grid.')
        sys.exit(1)

    if band_count != fn_vals_count:
        logger.error('----- ERROR -----')
        logger.error('The shape of of band energy and function value does not match.')
        logger.error(f'Shape of band energy   : {band_count}')
        logger.error(f'Shape of function value: {fn_vals_count}')
        logger.error('Check the shape of each one.')
        sys.exit(1)
 
    # Generate triangular mesh
    tri = Delaunay(kmesh)
    tetrahedron_count = tri.simplices.shape[0]

    # Generate fn * dos values 
    results = np.zeros(w_count)

    # min and max energy of each band
    band_emin = np.amin(bandenergy, axis=0)
    band_emax = np.amax(bandenergy, axis=0)
    
    for i in range(band_count):

        # idx_valid_energy:
        #     For the function argument w (single number of array),
        #     include contributions in
        #         band[i].min_energy <= w < band[i].max_energy 
        #     and saves the index.

        idx_valid_energy = np.where(
            (band_emin[i] <= w) & (w < band_emax[i])
        )[0]

        if idx_valid_energy.shape[0] == 0:
            pass
        else:
            energy_tetrahedron = bandenergy[tri.simplices, i]
            fn_tetrahedron = fn[:, tri.simplices, i]

            sort_arg = np.argsort(energy_tetrahedron)
            energy_tetrahedron_sorted = np.take_along_axis(energy_tetrahedron, sort_arg, axis=-1)
            fn_tetrahedron_sorted = np.take_along_axis(fn_tetrahedron, sort_arg[np.newaxis, :], axis=-1)

            # Iterate w satisfying emin_kpt <= w1, w2, ... < emax_kpt
            for idx in idx_valid_energy:
                energy = w[idx]

                # Tetra 1
                # No Contribution

                # Tetra 2
                idx_tetra = np.logical_and(
                    energy_tetrahedron_sorted[:,0] <= energy,
                    energy < energy_tetrahedron_sorted[:,1]
                )
                tetra = energy_tetrahedron_sorted[idx_tetra]
                tetra_fn = fn_tetrahedron_sorted[:, idx_tetra]

                w1, w2, w3 = tetra[:,0], tetra[:,1], tetra[:,2]
                w21 = w2 - w1
                w31 = w3 - w1
                w32 = w3 - w2

                f1, f2, f3 = tetra_fn[:, :, 0], tetra_fn[:, :, 1], tetra_fn[:, :, 2]

                tmpdos = 2 * (energy - w1) / w31 / w21 / tetrahedron_count
                tmpfn =(
                    f1 * ( (w2 - energy) / w21 + (w3 - energy) / w31) + \
                    f2 * (energy - w1) / w21 + \
                    f3 * (energy - w1) / w31
                ) * 0.5

                results[idx] +=  np.sum(tmpdos * tmpfn, axis=1)

                # Tetra 3
                idx_tetra = np.logical_and(
                    energy_tetrahedron_sorted[:,1] <= energy,
                    energy < energy_tetrahedron_sorted[:,2]
                )
                tetra = energy_tetrahedron_sorted[idx_tetra]
                tetra_fn = fn_tetrahedron_sorted[:, idx_tetra]

                w1, w2, w3 = tetra[:,0], tetra[:,1], tetra[:,2]
                w21 = w2 - w1
                w31 = w3 - w1
                w32 = w3 - w2

                f1, f2, f3 = tetra_fn[:, :, 0], tetra_fn[:, :, 1], tetra_fn[:, :, 2]

                tmpdos = 2 * (w3 - energy) / w31 / w32 / tetrahedron_count
                tmpfn = (
                    f1 *  (w3 - energy) / w31 + \
                    f2 * (w3 - energy) / w32 + \
                    f3 * ( (energy - w1) / w31 + (energy - w2) / w32)
                ) * 0.5

                results[idx] +=  np.sum(tmpdos * tmpfn, axis=1)

                # Tetra 4
                # No Contribution

    return results


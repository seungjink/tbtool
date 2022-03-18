import functools
import sys
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
    results = np.zeros([fn_count, w_count])

    # min and max energy of each band
    band_emin = np.amin(bandenergy, axis=0)
    band_emax = np.amax(bandenergy, axis=0)
    
    for i in range(band_count):

        # idx_valid_energy:
        #     if w is not between the minimum and maximum value of given band,
        #     it gives no contribution on Tetrahedron summation.
        #     So, we only consider [i]th band which satisfies following condition.
        #         band[i].min_energy <= w < band[i].max_energy 

        idx_valid_energy = np.where(
            (band_emin[i] <= w) & (w < band_emax[i])
        )[0]

        if idx_valid_energy.shape[0] == 0:
            pass
        else:
            energy_tetrahedron = bandenergy[tri.simplices, i]
            fn_tetrahedron = fn[:, tri.simplices, i]

            # Sort energies and function values in each tetrahedron in increasing order.
            # E   -> (e1 <= e2 <= e3)
            # fn -> [f(e1) <= f(e2) <= f(e3)]
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

                results[:, idx] +=  np.sum(tmpdos * tmpfn, axis=1, dtype=float)

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
                    f1 * (w3 - energy) / w31 + \
                    f2 * (w3 - energy) / w32 + \
                    f3 * ( (energy - w1) / w31 + (energy - w2) / w32)
                ) * 0.5

                results[:, idx] +=  np.sum(tmpdos * tmpfn, axis=1, dtype=float)

                # Tetra 4
                # No Contribution

    return results

def integrate_delta_3d_tetra(kmesh, bandenergy, fn, w):
    """Integrate delta function by 3d tetrahecron method.

    This function calculates the following delta funciton integral on the Brillouin zone using 2d tetrahedron method.

    .. math::
        \\begin{aligned}
            g(\\omega) &= \\sum_{n=1}^{band} \\frac{1}{V_G} \\int_{BZ} d^2 k f_{n\\bm{k}}(\\omega_{n\\bm{k}}) \delta(\\omega - \\omega_{n\\bm{k}}) \\\\
                       &= \\sum_{\\tau}^{N_{\\tau}} f_{\\tau}(\omega) D_{\\tau}(\\omega)
        \\end{aligned}

    Args:
        kmesh (numpy.ndarray): [Nmesh, 3] shaped mesh array.
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

    # Check whether the kmesh is 3d grid
    if kmesh.shape[1] != 3:
        logger.error('----- ERROR -----')
        logger.error('Input kpoint mesh does look as 3d grid.')
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
    results = np.zeros([fn_count, w_count])

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
                w1, w2, w3, w4 = tetra[:,0], tetra[:,1], tetra[:,2], tetra[:,3]
                w21 = w2 - w1
                w31 = w3 - w1
                w41 = w4 - w1
                w32 = w3 - w2
                w42 = w4 - w2
                w43 = w4 - w3

                f1, f2, f3, f4 = tetra_fn[:, :, 0], tetra_fn[:, :, 1], tetra_fn[:, :, 2], tetra_fn[:, :, 3]

                tmpdos = 3 * (energy - w1)**2 / w21 / w31 / w41 / tetrahedron_count
                tmpfn = 1/3 * (f1 * ( (w2 - energy) / w21 + (w3 - energy) / w31+ (w4 - energy) / w41) + \
                               f2 * (energy - w1) / w21 + \
                               f3 * (energy - w2) / w31 + \
                               f4 * (energy - w3) / w41)

                results[:, idx] +=  np.sum(tmpdos * tmpfn, axis=1, dtype=float)

                # Tetra 3
                idx_tetra = np.logical_and(
                    energy_tetrahedron_sorted[:,1] <= energy,
                    energy < energy_tetrahedron_sorted[:,2]
                )
                tetra = energy_tetrahedron_sorted[idx_tetra]
                tetra_fn = fn_tetrahedron_sorted[:, idx_tetra]

                w1, w2, w3, w4 = tetra[:,0], tetra[:,1], tetra[:,2], tetra[:,3]
                w21 = w2 - w1
                w31 = w3 - w1
                w41 = w4 - w1
                w32 = w3 - w2
                w42 = w4 - w2
                w43 = w4 - w3

                f1, f2, f3, f4= tetra_fn[:, :, 0], tetra_fn[:, :, 1], tetra_fn[:, :, 2], tetra_fn[:, :, 3]
                tmpdos = ( (3 * w21 + 6 * (energy - w2) - 3 * (w31 + w42) * (energy - w2)**2 / w32 / w42) / w31 / w41 ) / tetrahedron_count

                gi = 3 / w41 * ( (w3 - energy)/w32 * (energy - w2)/w31 + (energy - w2)/w32 * (w4 - energy)/w42)

                weight_fn1 = ((w4 - energy) / w41 / 3 + (w3 - energy) / w31 * (energy - w1) / w31 * (w3 - energy) / w32 / gi / w41) * f1 
                weight_fn2 = ((w3 - energy) / w32 / 3 + (w4 - energy) / w42 * (w4 - energy) / w42 * (energy - w2) / w32 / gi / w41) * f2 
                weight_fn3 = ((energy - w2) / w32 / 3 + (energy - w1) / w31 * (energy - w1) / w31 * (w3 - energy) / w32 / gi / w41) * f3 
                weight_fn4 = ((energy - w1) / w41 / 3 + (energy - w2) / w42 * (w4 - energy) / w42 * (energy - w2) / w32 / gi / w41) * f4 

                tmpfn = weight_fn1 + weight_fn2 + weight_fn3 + weight_fn4
                results[:, idx] +=  np.sum(tmpdos * tmpfn, axis=1, dtype=float)

                # Tetra 4
                idx_tetra = np.logical_and(
                    energy_tetrahedron_sorted[:,2] <= energy,
                    energy < energy_tetrahedron_sorted[:,3]
                )
                tetra = energy_tetrahedron_sorted[idx_tetra]
                tetra_fn = fn_tetrahedron_sorted[:, idx_tetra]

                w1, w2, w3, w4 = tetra[:,0], tetra[:,1], tetra[:,2], tetra[:,3]
                w21 = w2 - w1
                w31 = w3 - w1
                w41 = w4 - w1
                w32 = w3 - w2
                w42 = w4 - w2
                w43 = w4 - w3

                f1, f2, f3, f4= tetra_fn[:, :, 0], tetra_fn[:, :, 1], tetra_fn[:, :, 2], tetra_fn[:, :, 3]
                tmpdos = 3 * (w4 - energy)**2 / w41 / w42 / w43 / tetrahedron_count
                tmpfn = 1/3 * (f1 * ( (energy - w1) / w41 + (energy - w2) / w42 + (energy - w3) / w43) + \
                               f2 * (w4 - energy) / w41 + \
                               f3 * (w4 - energy) / w42 + \
                               f4 * (w4 - energy) / w43)

                results[:, idx] +=  np.sum(tmpdos * tmpfn, axis=1, dtype=float)

                # Tetra 5
                # No Contribution

    return results

def integrate_step_2d_tetra(kmesh, bandenergy, fn, w, fntype=float):
    """Integrate step function by 2d tetrahecron method.

    This function calculates the following step funciton integral on the Brillouin zone using 2d tetrahedron method.

    .. math::
        \\begin{aligned}
            g(\\omega) &= \\sum_{n=1}^{band} \\frac{1}{V_G} \\int_{BZ} d^2 k f_{n\\bm{k}}(\\omega_{n\\bm{k}}) \\theta(\\omega - \\omega_{n\\bm{k}}) \\\\
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
    results = np.zeros([fn_count, w_count], dtype=fntype)

    # min and max energy of each band
    band_emin = np.amin(bandenergy, axis=0)
    band_emax = np.amax(bandenergy, axis=0)
    
    for i in range(band_count):

        # idx_valid_energy:
        #     if w is not between the minimum and maximum value of given band,
        #     it gives no contribution on Tetrahedron summation.
        #     So, we only consider [i]th band which satisfies following condition.
        #         band[i].min_energy <= w < band[i].max_energy 

        idx_valid_energy = np.where(
            (band_emin[i] <= w)
        )[0]

        if idx_valid_energy.shape[0] == 0:
            pass
        else:
            energy_tetrahedron = bandenergy[tri.simplices, i]
            fn_tetrahedron = fn[:, tri.simplices, i]

            # Sort energies and function values in each tetrahedron in increasing order.
            # E   -> (e1 <= e2 <= e3)
            # fn -> [f(e1) <= f(e2) <= f(e3)]
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

                tmpocc = (energy - w1) ** 2 / w31 / w21 / tetrahedron_count
                tmpfn =(
                    f1 * (1 + (w2 - energy) / w21 + (w3 - energy) / w31) + \
                    f2 * (energy - w1) / w21 + \
                    f3 * (energy - w1) / w31
                ) / 3.0

                # Curature correction
                tmpfn += (
                    f1 * (w2 + w3 - 2 * w1) + \
                    f2 * (w3 + w1 - 2 * w2) + \
                    f3 * (w1 + w2 - 2 * w3)
                ) / 12.0 / (energy-w1)

                results[:, idx] +=  np.sum(tmpocc * tmpfn, axis=1, dtype=fntype)

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

                tmpocc = (w3 - energy) ** 2 / w31 / w32 / tetrahedron_count
                tmpfn = (
                    f1 * ( 1 / tetrahedron_count - tmpocc * (w3 - energy) / w31) + \
                    f2 * ( 1 / tetrahedron_count - tmpocc * (w3 - energy) / w32) + \
                    f3 * ( 1 / tetrahedron_count - tmpocc * (1 + (energy - w1) / w31 + (energy - w2) / w32))
                ) / 3.0

                # Curature correction
                tmpfn += (
                    f1 * ( w2 + w3 - 2*w1) + \
                    f2 * ( w3 + w1 - 2*w2) + \
                    f3 * ( w1 + w2 - 2*w3)
                ) / 12.0 / (w3 - energy) * tmpocc

                results[:, idx] += np.sum(tmpfn, axis=1, dtype=fntype)

                # Tetra 4

                idx_tetra = energy_tetrahedron_sorted[:,2] <= energy

                tetra = energy_tetrahedron_sorted[idx_tetra]
                tetra_fn = fn_tetrahedron_sorted[:, idx_tetra]

                f1, f2, f3 = tetra_fn[:, :, 0], tetra_fn[:, :, 1], tetra_fn[:, :, 2]

                tmpfn = (
                    f1 + f2 + f3
                ) / 3.0 / tetrahedron_count

                results[:, idx] += np.sum(tmpfn, axis=1, dtype=fntype)

                # No Contribution
    return results







#
#def integrate_step_2d_tetra(kmesh, bandenergy, fn, w):
#
#    """Integrate step function by 2d tetrahecron method.
#
#    This function calculates the following step funciton integral on the Brillouin zone using 2d tetrahedron method.
#
#    .. math::
#        \\begin{aligned}
#            g(\\omega) &= \\sum_{n=1}^{band} \\frac{1}{V_G} \\int_{BZ} d^2 k f_{n\\bm{k}}(\\omega_{n\\bm{k}}) \\theta(\\omega - \\omega_{n\\bm{k}}) \\\\
#                       &= \\sum_{\\tau}^{N_{\\tau}} f_{\\tau}(\omega) D_{\\tau}(\\omega)
#        \\end{aligned}
#
#    Args:
#        kmesh (numpy.ndarray): [Nmesh, 2] shaped mesh array.
#        bandenergy (numpy.ndarray): [Nmesh, Nband] shaped band energy.
#        fn (numpy.ndarray): [Nmesh, Nband] shaped function value.
#        w (float or list): function values to be calculated.
#
#    """

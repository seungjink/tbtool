import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('D:\\Project\\tbtool'))
import tbtool.hamiltonian as ham
import tbtool.unit as unit
import numpy as np
from scipy.linalg import eigh, eigvalsh

class Cst(ham.Hamiltonian):
    def __init__(self):
        self.t1_x2 = 0.0
        self.t1_z2 = 0.0
        self.t2 = 0.0
        self.t3 = 0.0
        self.t5_a = 0.0
        self.t5_b = 0.0

        self.t1_soc = 0.0
        self.t3_soc = 0.0
        self.aniso = 0.0
        self.t2_soc_z = 0.0
        self.t2_soc_xy = 0.0

        self.magmom = [0,0,1]

        self.chemp = 0.0
        self.unit = {"energy": "ev"}

    def get(self, kpt):
        kx, ky, kz = kpt

        X = np.array([-1/np.sqrt(2),-1/np.sqrt(6) , 1/np.sqrt(3)])
        Y = np.array([1/np.sqrt(2), -1/np.sqrt(6), 1/np.sqrt(3)])
        Z = np.array([0, 2/np.sqrt(6), 1/np.sqrt(3)])

        I = 1j
        exp = np.exp
        pi = np.pi
        sqrt = np.sqrt

        vec_normalized = np.array(self.magmom) / np.linalg.norm(np.array(self.magmom))

        ####################################
        ### Set spin-independent hopping ###
        ####################################
        h1_a = np.array([[0, 0, self.t1_x2, 0], [0, 0, 0, self.t1_z2], [self.t1_x2, 0, 0, 0], [0, self.t1_z2, 0, 0]])
        h1_b = np.array([[0, 0, (self.t1_x2/4 + 3*self.t1_z2/4)*exp(2*I*pi*ky), sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(2*I*pi*ky)/4], [0, 0, sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(2*I*pi*ky)/4, (3*self.t1_x2/4 + self.t1_z2/4)*exp(2*I*pi*ky)], [(self.t1_x2/4 + 3*self.t1_z2/4)*exp(-2*I*pi*ky), sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(-2*I*pi*ky)/4, 0, 0], [sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(-2*I*pi*ky)/4, (3*self.t1_x2/4 + self.t1_z2/4)*exp(-2*I*pi*ky), 0, 0]])
        h1_c = np.array([[0, 0, (self.t1_x2/4 + 3*self.t1_z2/4)*exp(-2*I*pi*kx), -sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(-2*I*pi*kx)/4], [0, 0, -sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(-2*I*pi*kx)/4, (3*self.t1_x2/4 + self.t1_z2/4)*exp(-2*I*pi*kx)], [(self.t1_x2/4 + 3*self.t1_z2/4)*exp(2*I*pi*kx), -sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(2*I*pi*kx)/4, 0, 0], [-sqrt(3)*(self.t1_x2 - self.t1_z2)*exp(2*I*pi*kx)/4, (3*self.t1_x2/4 + self.t1_z2/4)*exp(2*I*pi*kx), 0, 0]])
        h1 = h1_a + h1_b + h1_c    

        h3_a = np.array([[0, 0, 3*self.t3*exp(2*I*pi*(-kx + ky)), 0], [0, 0, 0, -self.t3*exp(2*I*pi*(-kx + ky))], [3*self.t3*exp(2*I*pi*(kx - ky)), 0, 0, 0], [0, -self.t3*exp(2*I*pi*(kx - ky)), 0, 0]])
        h3_b = np.array([[0, 0, 0, sqrt(3)*self.t3*exp(2*I*pi*(-kx - ky))], [0, 0, sqrt(3)*self.t3*exp(2*I*pi*(-kx - ky)), 2*self.t3*exp(2*I*pi*(-kx - ky))], [0, sqrt(3)*self.t3*exp(2*I*pi*(kx + ky)), 0, 0], [sqrt(3)*self.t3*exp(2*I*pi*(kx + ky)), 2*self.t3*exp(2*I*pi*(kx + ky)), 0, 0]] )
        h3_c = np.array([[0, 0, 0, -sqrt(3)*self.t3*exp(2*I*pi*(kx + ky))], [0, 0, -sqrt(3)*self.t3*exp(2*I*pi*(kx + ky)), 2*self.t3*exp(2*I*pi*(kx + ky))], [0, -sqrt(3)*self.t3*exp(2*I*pi*(-kx - ky)), 0, 0], [-sqrt(3)*self.t3*exp(2*I*pi*(-kx - ky)), 2*self.t3*exp(2*I*pi*(-kx - ky)), 0, 0]])
        h3 = h3_a + h3_b + h3_c

        h2_a = np.array([[0, 0, 0, 0], [0, -2*self.t2*np.cos(np.pi*(2*kx + 2*ky)), 0, 0], [0, 0, 0, 0], [0, 0, 0, -2*self.t2*np.cos(np.pi*(2*kx + 2*ky))]])
        h2_b = np.array([[-3*self.t2*np.cos(2*np.pi*kx)/2, np.sqrt(3)*self.t2*np.cos(2*np.pi*kx)/2, 0, 0], [np.sqrt(3)*self.t2*np.cos(2*np.pi*kx)/2, -self.t2*np.cos(2*np.pi*kx)/2, 0, 0], [0, 0, -3*self.t2*np.cos(2*np.pi*kx)/2, np.sqrt(3)*self.t2*np.cos(2*np.pi*kx)/2], [0, 0, np.sqrt(3)*self.t2*np.cos(2*np.pi*kx)/2, -self.t2*np.cos(2*np.pi*kx)/2]])
        h2_c = np.array([[-3*self.t2*np.cos(2*np.pi*ky)/2, -np.sqrt(3)*self.t2*np.cos(2*np.pi*ky)/2, 0, 0], [-np.sqrt(3)*self.t2*np.cos(2*np.pi*ky)/2, -self.t2*np.cos(2*np.pi*ky)/2, 0, 0], [0, 0, -3*self.t2*np.cos(2*np.pi*ky)/2, -np.sqrt(3)*self.t2*np.cos(2*np.pi*ky)/2], [0, 0, -np.sqrt(3)*self.t2*np.cos(2*np.pi*ky)/2, -self.t2*np.cos(2*np.pi*ky)/2]])
        h2 = h2_a + h2_b + h2_c

        h5_a = np.array([[2*self.t5_a*np.cos(np.pi*(-2*kx + 2*ky)), 0, 0, 0], [0, 2*self.t5_b*np.cos(np.pi*(-2*kx + 2*ky)), 0, 0], [0, 0, 2*self.t5_a*np.cos(np.pi*(-2*kx + 2*ky)), 0], [0, 0, 0, 2*self.t5_b*np.cos(np.pi*(-2*kx + 2*ky))]])
        h5_b = np.array([[(self.t5_a + 3*self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2, np.sqrt(3)*(self.t5_a - self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2, 0, 0], [np.sqrt(3)*(self.t5_a - self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2, (3*self.t5_a + self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2, 0, 0], [0, 0, (self.t5_a + 3*self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2, np.sqrt(3)*(self.t5_a - self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2], [0, 0, np.sqrt(3)*(self.t5_a - self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2, (3*self.t5_a + self.t5_b)*np.cos(np.pi*(2*kx + 4*ky))/2]])
        h5_c = np.array([[(self.t5_a + 3*self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2, np.sqrt(3)*(-self.t5_a + self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2, 0, 0], [np.sqrt(3)*(-self.t5_a + self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2, (3*self.t5_a + self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2, 0, 0], [0, 0, (self.t5_a + 3*self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2, np.sqrt(3)*(-self.t5_a + self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2], [0, 0, np.sqrt(3)*(-self.t5_a + self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2, (3*self.t5_a + self.t5_b)*np.cos(np.pi*(4*kx + 2*ky))/2]])

        h5 = h5_a + h5_b + h5_c

        #######################################################################
        # FM CASE
        ma_ani = np.dot(X,vec_normalized)
        mb_ani = np.dot(Y,vec_normalized)
        mc_ani = np.dot(Z,vec_normalized)

        ma = np.dot(X,vec_normalized) * np.exp(1j * 2 * np.pi * ky)
        mb = np.dot(Y,vec_normalized) * np.exp(-1j * 2 * np.pi * kx)
        mc = np.dot(Z,vec_normalized) 

        ma3 = np.dot(X,vec_normalized) * np.exp(1j * 2 * np.pi * (-kx-ky))
        mb3 = np.dot(Y,vec_normalized) * np.exp(1j * 2 * np.pi * (kx+ky))
        mc3 = np.dot(Z,vec_normalized) * np.exp(1j * 2 * np.pi * (-kx+ky))

        ma_soc_2a = np.dot(Z, vec_normalized) * np.cos(2 * np.pi * (kx + ky)) \
            + np.dot(X, vec_normalized) * np.cos(2 * np.pi * (kx)) \
            + np.dot(Y, vec_normalized) * np.cos(2 * np.pi * (ky))


        #soc 2nn xy part
        mzsin = np.sqrt(1-np.dot(Z, vec_normalized)**2)
        mxsin = np.sqrt(1-np.dot(X, vec_normalized)**2)
        mysin = np.sqrt(1-np.dot(Y, vec_normalized)**2)

        ma_soc_2xy = np.dot(vec_normalized, X+Y) * mzsin * np.cos(2 * np.pi * (kx+ky)) \
            + np.dot(vec_normalized, Y+Z)  * mxsin * np.cos(2 * np.pi * (kx)) \
            + np.dot(vec_normalized, Z+Y) * mysin * np.cos(2 * np.pi * (ky)) 
        mb_soc_2xy = np.dot(vec_normalized, X+Y)  * mzsin * np.cos(2 * np.pi * (kx+ky)) \
            +np.dot(vec_normalized, X+Z)  * mysin * np.cos(2 * np.pi * (kx)) \
            +np.dot(vec_normalized, Z+X)  * mxsin * np.cos(2 * np.pi * (ky)) 

        soc_aniso = np.array([[0,-1,0,0],
                              [1,0,0,0],
                              [0,0,0,-1],
                              [0,0,1,0]]) * I * self.aniso * (ma_ani+mb_ani+mc_ani)
        soc_nn = np.array([[0,0,0,1],
                           [0,0,-1,0],
                           [0,1,0,0],
                           [-1,0,0,0]]) * I * (-1)* self.t1_soc * np.conjugate((ma+mb+mc))
        soc_3nn = np.array([[0,0,0,1],
                           [0,0,-1,0],
                           [0,1,0,0],
                           [-1,0,0,0]]) * I * (-1)* self.t3_soc * np.conjugate((ma3+mb3+mc3))

        soc_2nn_a = np.array([[0,-1,0,0],
                              [1,0,0,0],
                              [0,0,0,-1],
                              [0,0,1,0]]) * 2 * I * self.t2_soc_z * ma_soc_2a
        soc_2nn_tmp_a = np.array([[0,-1],[1,0]]) * 2 * I * self.t2_soc_xy * ma_soc_2xy 
        soc_2nn_tmp_b = np.array([[0,-1],[1,0]]) * 2 * I * self.t2_soc_xy * mb_soc_2xy 
        soc_2nn_xy = np.zeros([4,4], dtype=np.complex)
        soc_2nn_xy[0:2, 0:2] = soc_2nn_tmp_a
        soc_2nn_xy[2:4, 2:4] = soc_2nn_tmp_b

        soc_2nn_full = soc_2nn_a + soc_2nn_xy

        #mat = h2_afm + h1_afm + h3_afm  + soc_B_afm +h5_afm  #+soc_aniso_afm + soc_nn_afm + soc_3nn_afm # + soc_B_afm #+ h1 + h2 + h3 #+ soc_3nn  # + soc_nn #soc_3nn #h1 + h3 + soc_3nn # soc_aniso #soc_nn #oc_aniso + soc_nn 
        #print(soc_nn_afm)
        mat = h1 + h2 + h3 + h5 + soc_nn + soc_3nn + soc_aniso + soc_2nn_full
        #mat2 =    soc_nn + soc_3nn + soc_aniso

        return mat, np.eye(4)

    def diagonalize(self, kpt, eigvals_only=True):
        ham, olp = self.get(kpt)
        if eigvals_only:
            en = eigvalsh(ham, olp, lower=False)
            return (en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy'])
        else:
            en, ev = eigh(ham, olp, lower=False)
            return ((en - self.chemp) * unit.get_conversion_factor('energy', 'hartree', self.unit['energy']), ev)
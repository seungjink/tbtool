import unittest
import tbtool.kpoints as kpt
import numpy as np

class KpointsTests(unittest.TestCase): 

    def test_2d_mesh(self):
        kmesh = kpt.Kmesh([2,2])
        self.assertTrue(
            np.array_equal(kmesh.get(),np.array([[-0.25,-0.25],[-0.25,0.25],[0.25,-0.25],[0.25,0.25]]))
        )

    def test_3d_mesh(self):
        kmesh = kpt.Kmesh([2,2,1])
        self.assertTrue(
            np.array_equal(kmesh.get(),
            np.array([[-0.25,-0.25,0],[-0.25,0.25,0],[0.25,-0.25,0],[0.25,0.25,0]]))
        )

    def test_cartesian(self):
        kmesh = kpt.Kmesh([2,2,1], unitcell=[[4,0,0],[0,2,0],[0,0,1]])
        cart = kmesh.get_cartesian()
        self.assertTrue(
            np.array_equal(kmesh.get_cartesian(),
            np.array([[-1,-0.5,0],[-1,0.5,0],[1,-0.5,0],[1,0.5,0]]))
        )

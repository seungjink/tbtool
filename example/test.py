import os
import time
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('D:\\Project\\tbtool'))

import tbtool.io as io
from tbtool.calculator import base
from tbtool.kpoints import monkhorst_pack
import tbtool.kpoints as kp
from tbtool.calculator import berry, algo, dos

import tbtool.basis as basis

bs = basis.Basis('orbital')
bs.add('a'); bs.add('b'); bs.add('c')
print(bs.basis[1])

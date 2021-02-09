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

a = kp.Kpath([[0,0,0],[0.5,0,0],[1/3,1/3,0]])
print(a.get(3))
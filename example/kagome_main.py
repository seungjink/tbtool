# -*- coding: utf-8 -*-
# Copyright (C) 2017 cnmp.SNU

import logging
from ase.visualize import view
from ase.io import write
from tbmodel.calculator import TBmodel
from tbmodel.io import logger
from tbmodel import initialize

def main():
    """ Initial Setting """
    logger.setup_logging(default_level=logging.DEBUG)
    DBG = logging.getLogger(__name__)
    filename = 'example//kagome_tb.dat' # [Modify] input option path 
    tb_data = logger.TBinput(filename)
    atoms = initialize.setAtoms(tb_data)
    calc = TBmodel(kpts={'path': 'GKMG', 'npoints': 5}) # [Modify] Band path
    """"""""""""""""""""""""

    """ Options for band structure"""
    calc.calculate(atoms, tb_data)
    bs = calc.band_structure()
    bs.plot(filename=filename + 'bands.png', show=True, emin=-0.5, emax=2.0)
    """"""""""""""""""""""""""""""""""""

if __name__ == "__main__":
    main()

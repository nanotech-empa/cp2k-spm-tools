#!/bin/bash -l

cd out

wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz

tar -xvf bader_lnx_64.tar.gz

./bader -b weight -p all_atom  charge_density.cube -ref charge_density_artif.cube



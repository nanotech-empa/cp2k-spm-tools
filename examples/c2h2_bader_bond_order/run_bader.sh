#!/bin/bash -l

cd out

wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz

tar -xvf bader_lnx_64.tar.gz

./bader -p sel_atom 1 2 3 4 charge_density_artif.cube



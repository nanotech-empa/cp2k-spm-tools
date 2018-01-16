
import numpy as np

import xml.etree.ElementTree as et

def read_scf_fermi(scf_out_file):
    fermi = None
    with open(scf_out_file) as f:
        for l in f:
            if "Fermi energy" in l:
                fermi = float(l.split()[-2])
    if fermi is None:
        print("Couldn't find Fermi energy!")
    return fermi


def read_band_data(data_dir):

    data_file_xml = et.parse(data_dir+"data-file.xml")
    data_file_root = data_file_xml.getroot()

    kpts = []
    eig_vals = []

    # Loop through all K-POINTS xml nodes
    for kpt in data_file_root.find('EIGENVALUES'):
        #print(kpt.tag, kpt.attrib)

        # Save the kpoint coordinate to kpts[]
        kpt_coords = kpt.find('K-POINT_COORDS')
        kpts.append(np.double(kpt_coords.text.split()))
        #print("    ",kpts[-1])

        # Find the file containing eigenvalues corresponding to this k-point
        eig_datafile_xml = kpt.find('DATAFILE')
        eig_datafile = eig_datafile_xml.attrib['iotk_link']
        eig_datafile = eig_datafile[2:] # remove "./" from start
        #print("    ", eig_datafile)

        # Go retrieve the eigenvalues
        eig_file_xml = et.parse(data_dir+eig_datafile)
        eig_file_root = eig_file_xml.getroot()

        eigval_node = eig_file_root.find('EIGENVALUES')

        # Convert from hartree to eV and subtract E Fermi
        eig_vals.append(np.double(eigval_node.text.split())*27.21138602)
        #print(eig_vals[-1])

    return np.array(kpts), np.array(eig_vals)

def read_and_shift_bands(scf_out_file, data_dir):
    efermi = read_scf_fermi(scf_out_file)
    kpts, eig_vals = read_band_data(data_dir)
    eig_vals -= efermi
    return kpts, eig_vals

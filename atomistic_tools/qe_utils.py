
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

def read_band_xml_datafile(eig_datafile_xml_node, data_dir):
    eig_datafile = eig_datafile_xml_node.attrib['iotk_link']
    eig_datafile = eig_datafile[2:] # remove "./" from start
     # Go retrieve the eigenvalues
    eig_file_xml = et.parse(data_dir+eig_datafile)
    eig_file_root = eig_file_xml.getroot()

    eigval_node = eig_file_root.find('EIGENVALUES')

    # Convert from hartree to eV
    eig_vals = [float(i)*27.21138602 for i in eigval_node.text.split()]
    return eig_vals

def read_band_data(data_dir):

    data_file_xml = et.parse(data_dir+"data-file.xml")
    data_file_root = data_file_xml.getroot()

    # Find fermi
    band_info_node = data_file_root.find('BAND_STRUCTURE_INFO')
    fermi_en = float(band_info_node.find('FERMI_ENERGY').text)*27.21138602
    nspin = int(band_info_node.find('NUMBER_OF_SPIN_COMPONENTS').text)

    kpts = []
    eig_vals = [[] for _ in range(nspin)]

    # Loop through all K-POINTS xml nodes
    for kpt in data_file_root.find('EIGENVALUES'):
        #print(kpt.tag, kpt.attrib)

        # Save the kpoint coordinate to kpts[]
        kpt_coords = kpt.find('K-POINT_COORDS')
        kpts.append([float(i) for i in kpt_coords.text.split()])
        #print("    ",kpts[-1])

        if nspin == 1:
            # Find the file containing eigenvalues corresponding to this k-point
            eig_datafile_xml = kpt.find('DATAFILE')
            eig_vals[0].append(read_band_xml_datafile(eig_datafile_xml, data_dir))
        else:
            eig_datafile_xml1 = kpt.find('DATAFILE.1')
            eig_vals[0].append(read_band_xml_datafile(eig_datafile_xml1, data_dir))
            eig_datafile_xml2 = kpt.find('DATAFILE.2')
            eig_vals[1].append(read_band_xml_datafile(eig_datafile_xml2, data_dir))

    kpts = np.array(kpts)
    eig_vals = [np.array(ev).T for ev in eig_vals]

    return kpts, eig_vals, fermi_en

def vb_onset(bands, fermi_en):
    """
    In case of metallic system, returns fermi energy,
    otherwise the HOMO or the VB onset
    """
    # Start from the highest energy band and move down
    for i in range(len(bands)-1, -1, -1):
        if np.min(bands[i]) < fermi_en:
            # we found a band that is at least partially occupied
            return np.min([np.max(bands[i]), fermi_en])
    print("Error: VB onset not found!")
    return None

def gap_middle(bands, fermi_en):
    """
    In case of metallic system, returns fermi energy,
    otherwise the middle of the band-gap
    """
    # Start from the highest energy band and move down
    for i in range(len(bands)-1, -1, -1):
        if np.min(bands[i]) < fermi_en:
            if np.max(bands[i]) > fermi_en:
                return fermi_en
            return 0.5*(np.max(bands[i]) + np.min(bands[i+1]))
    return None


def read_atomic_proj(atomic_proj_xml):
    """
    Reads the atomic_proj.xml output file produced by projwfc.x
    """

    data_file_xml = et.parse(atomic_proj_xml)
    data_file_root = data_file_xml.getroot()
    
    header_node = data_file_root.find("HEADER")

    n_bands = int(header_node.find("NUMBER_OF_BANDS").text)
    n_kpts = int(header_node.find("NUMBER_OF_K-POINTS").text)
    n_spin = int(header_node.find("NUMBER_OF_SPIN_COMPONENTS").text)
    n_at_wfc = int(header_node.find("NUMBER_OF_ATOMIC_WFC").text)
    n_el = int(float(header_node.find("NUMBER_OF_ELECTRONS").text))
    fermi_en = float(header_node.find("FERMI_ENERGY").text) * 13.605698065894
    
    kpts_node = data_file_root.find("K-POINTS")

    kpts = []
    for kpt_str in kpts_node.text.split('\n'):
        kpt_str_split = kpt_str.split()
        if len(kpt_str_split) < 3:
            continue
        kpts.append(np.array(kpt_str_split).astype(float))
    kpts = np.array(kpts)
    
    eigvals_node = data_file_root.find("EIGENVALUES")

    eigvals = []
    for i_spin in range(n_spin):
        eigvals.append([])
        for i_kpt in range(n_kpts):
            eigval_arr = np.array(eigvals_node[i_kpt][i_spin].text.split()).astype(float)
            eigvals[i_spin].append(eigval_arr)
    eigvals = np.array(eigvals)*13.605698065894
    
    projections_node = data_file_root.find("PROJECTIONS")

    wfc_projs = np.zeros((n_spin, n_kpts, n_at_wfc, n_bands), dtype=np.complex)
    for i_kpt in range(n_kpts):
        for i_spin in range(n_spin):
            for i_wfc in range(n_at_wfc):
                split_lines = projections_node[i_kpt][i_spin][i_wfc].text.split('\n')
                wfc_proj_arr = []
                for x in split_lines:
                    if len(x.strip()) < 1:
                        continue
                    y = np.array(x.split(',')).astype(float)
                    wfc_proj_arr.append(y[0] + 1j*y[1])

                wfc_proj_arr = np.array(wfc_proj_arr)

                wfc_projs[i_spin, i_kpt, i_wfc, :] =  wfc_proj_arr
    
    return kpts, eigvals, wfc_projs, fermi_en

def correct_band_crossings(kpts, eigvals_in, wfc_projs_in):
    """
    Using parabolic fitting, orders the band segments correctly 
    """

    eigvals = np.copy(eigvals_in)
    wfc_projs = np.copy(wfc_projs_in)
    
    for i_spin in range(2):
        for i_k in range(1, eigvals.shape[1]-1):
            k_vals = kpts[i_k-1:i_k+2, 0]

            for i_band in range(eigvals.shape[2]):
                for i_band2 in range(i_band+1, eigvals.shape[2]):

                    eigval = eigvals[i_spin, i_k, i_band]
                    dif = np.abs(eigval - eigvals[i_spin, i_k-1, i_band])
                    if np.abs(eigvals[i_spin, i_k+1, i_band2] - eigval) < 2*dif:

                        e_vals_cur = eigvals[i_spin, i_k-1:i_k+2, i_band]
                        e_vals_pre = eigvals[i_spin, i_k-1:i_k+2, i_band2]

                        # switched last points
                        e_vals_cur_sw = [e_vals_cur[0], e_vals_cur[1], e_vals_pre[2]]
                        e_vals_pre_sw = [e_vals_pre[0], e_vals_pre[1], e_vals_cur[2]]

                        # fit parabolas
                        fit_cur = np.polyfit(k_vals, e_vals_cur, 2)
                        fit_pre = np.polyfit(k_vals, e_vals_pre, 2)
                        fit_cur_sw = np.polyfit(k_vals, e_vals_cur_sw, 2)
                        fit_pre_sw = np.polyfit(k_vals, e_vals_pre_sw, 2)

                        #if 0.12 < k_vals[1] < 0.16:
                        #    print(k_vals)
                        #    print(e_vals_cur, fit_cur[0])
                        #    print(e_vals_pre, fit_pre[0])
                        #    print(e_vals_cur_sw, fit_cur_sw[0])
                        #    print(e_vals_pre_sw, fit_pre_sw[0])
                        #    print("--")

                        # Switch if the sum of the curvature of parabolas is smaller
                        if (np.abs(fit_cur_sw[0]) + np.abs(fit_pre_sw[0]))+20.0 < (np.abs(fit_cur[0]) + np.abs(fit_pre[0])):
                        #if np.abs(fit_cur_sw[0])*np.abs(fit_pre_sw[0]) < np.abs(fit_cur[0])*np.abs(fit_pre[0]):
                            #print("Switch pos: ", k_vals[1], e_vals_cur[1])

                            orig_band = np.copy(eigvals[i_spin, i_k+1:, i_band])
                            eigvals[i_spin, i_k+1:, i_band] = eigvals[i_spin, i_k+1:, i_band2]
                            eigvals[i_spin, i_k+1:, i_band2] = orig_band
                            
                            orig_wfc = np.copy(wfc_projs[i_spin, i_k+1:, :, i_band])
                            wfc_projs[i_spin, i_k+1:, :, i_band] = wfc_projs[i_spin, i_k+1:, :, i_band2]
                            wfc_projs[i_spin, i_k+1:, :, i_band2] = orig_wfc
    
    return eigvals, wfc_projs


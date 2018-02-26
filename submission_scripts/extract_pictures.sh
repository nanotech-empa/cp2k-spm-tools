#!/bin/bash -l


FOLDER="/scratch/snx3000/keimre/"

List=(
    "cp2k_cnt_12_0/L50-15-h-AA-de/diag/"
    "cp2k_cnt_12_0/L50-15-h-AA-se/diag/"
    "cp2k_cnt_12_0/L50-15-h-AB1-de/diag/"
    "cp2k_cnt_12_0/L50-15-h-AB1-se/diag/"
    "cp2k_cnt_12_0/L50-ideal-de/diag/"
    "cp2k_cnt_12_0/L50-ideal-se/diag/"
    "cp2k_cnt_12_0/L60-12-h-AA/diag/"
    "cp2k_cnt_12_0/L60-12-h-AAx/diag/"
    "cp2k_cnt_12_0/L60-12-h-AAy/diag/"
    "cp2k_cnt_12_0/L60-12-h-ABA02A11/diag/"
    "cp2k_cnt_12_0/L60-12-oxy-AA/diag/"
    "cp2k_cnt_12_0/L60-12-vac-circular/diag/"
    "cp2k_cnt_12_0/L60-12-vac-opt-AA/diag/"
    "cp2k_cnt_12_0/L60-12-vac-opt-AA/diag/"
    "cp2k_cnt_12_0/L100-15-h-AA-se/diag/"
    "cp2k_cnt_12_0/L12-ideal"
    "cp2k_cnt_12_0/L12-ideal-no-ends"
    "cp2k_cnt_12_0/L15-ideal"
    "cp2k_cnt_12_12/L50-ideal/diag/"
    "cp2k_cnt_12_0/L50-15-h-A-AB/diag/"
)

H_LIST=(
    "1.0"
    "5.0"
)

FWHM_LIST=(
    "0.01"
    "0.03"
    "0.05"
)

DEST_PATH="/scratch/snx3000/keimre/cnt_postproc_output/"

rm -rf $DEST_PATH
mkdir $DEST_PATH

for DATAPATH in "${List[@]}" 
do
    echo $FOLDER$DATAPATH
    cd $FOLDER$DATAPATH
    cd ./sts_output
    
    for H in "${H_LIST[@]}"
    do
        ORB_PIC=$(ls -tr | grep orb_a.*h$H.*\.png | tail -n 1)
	mkdir ${DEST_PATH}h${H}
	cp $ORB_PIC ${DEST_PATH}h${H}
	
	for FWHM in "${FWHM_LIST[@]}"
	do
	    mkdir ${DEST_PATH}h${H}/fwhm$FWHM
	    FTSTS_PIC=$(ls -tr | grep ldos.*h$H.*fwhm$FWHM.*\.png | tail -n 1)
	    cp $FTSTS_PIC ${DEST_PATH}h${H}/fwhm$FWHM 
	done

    done 
done

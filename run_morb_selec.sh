python3 select_morbs_restart-wfn.py \
  --cp2k_output ./out.log \
  --restart_file ./PROJ-RESTART.wfn \
  --output_file ./restart_sel_morbs.wfn \
  --emin -2.0 \
  --emax 2.0 \
  | tee morb_selec.out

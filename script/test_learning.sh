mkdir -p output/debug_learn_shadow
ln -s /mnt/disk1/jiayichen/code/DexLearn/output/dexonomy_shadow_nflow_type1/tests output/debug_learn_shadow/grasp_data
dexrun op=eval exp_name=debug_learn tmpl_upd_mode=disabled
dexrun op=stat exp_name=debug_learn
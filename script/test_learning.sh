mkdir -p output/debug_learn_shadow
ln -s /mnt/disk1/jiayichen/code/DexLearn/output/dexonomy_shadow_nflow_type1/tests output/debug_learn_shadow/graspdata
python -m dexonomy.main task=syn_test exp_name=debug_learn update_template=False
python -m dexonomy.main task=stat exp_name=debug_learn
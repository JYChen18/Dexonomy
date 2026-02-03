# test admm strategy
# dexrun op=grasp_admm 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'init_dir=${save_dir}/graspdata' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_3' 'log_id=3'

# dexrun op=eval 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'init_dir=${save_dir}/graspdata' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_3' 'succ_grasp_dir=${save_dir}/succ_grasp_3' 'log_id=3'

# dexrun op=stat 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'init_dir=${save_dir}/graspdata' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_3' 'succ_grasp_dir=${save_dir}/succ_grasp_3' 'log_id=3'

# test evaluation correctness
dexrun op=eval 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/succgrasp' 'succ_grasp_dir=${save_dir}/succgrasp_confirm' 'log_id=4'

dexrun op=stat 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/succgrasp' 'succ_grasp_dir=${save_dir}/succgrasp_confirm' 'log_id=4'

# test harder benchmark
dexrun op=eval 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/graspdata' 'succ_grasp_dir=${save_dir}/succgrasp_hard' 'log_id=5' 'op.hard=True'

dexrun op=stat 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/graspdata' 'succ_grasp_dir=${save_dir}/succgrasp_hard' 'log_id=5'

# dexrun op=eval 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_3' 'succ_grasp_dir=${save_dir}/succ_grasp_3_hard' 'log_id=6' 'op.hard=True'

# dexrun op=stat 'exp_name=Dexonomy_GRASP_small' 'hand=shadow' 'init_dir=${save_dir}/graspdata' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_3' 'succ_grasp_dir=${save_dir}/succ_grasp_3_hard' 'log_id=6'

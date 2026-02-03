# dexrun op=grasp_admm 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 
# dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True'
# dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True'

# dexrun op=grasp 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'log_id=1'
# dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_1_base' 'log_id=1'
# dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_1_base' 'log_id=1'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_4_base_mu_01' 'log_id=4'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data' 'succ_grasp_dir=${save_dir}/succ_grasp_5_admm_mu_01' 'log_id=5'

dexrun op=grasp_admm 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_6_old_config' 'log_id=6'
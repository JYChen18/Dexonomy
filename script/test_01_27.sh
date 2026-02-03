dexrun op=grasp_admm 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_18_config_01_27' 'log_id=18' 'op.grasp.square_res=True'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_18_config_01_27' 'succ_grasp_dir=${save_dir}/succ_grasp_18_config_01_27' 'log_id=18'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_18_config_01_27' 'succ_grasp_dir=${save_dir}/succ_grasp_18_config_01_27' 'log_id=18'
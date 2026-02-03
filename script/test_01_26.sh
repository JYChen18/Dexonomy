## harder bench
# 0.1
dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_7_base_mu_01' 'log_id=7' 'op.miu_coef=[0.1, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_7_base_mu_01' 'log_id=7'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_8_admm_mu_01' 'log_id=8' 'op.miu_coef=[0.1, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_8_admm_mu_01' 'log_id=8'

# 0.2
dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_9_base_mu_02' 'log_id=9' 'op.miu_coef=[0.2, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_9_base_mu_02' 'log_id=9'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_10_admm_mu_02' 'log_id=10' 'op.miu_coef=[0.2, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_10_admm_mu_02' 'log_id=10'

# 0.3
dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_11_base_mu_03' 'log_id=11' 'op.miu_coef=[0.3, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_11_base_mu_03' 'log_id=11'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_12_admm_mu_03' 'log_id=12' 'op.miu_coef=[0.3, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_12_admm_mu_03' 'log_id=12'

# 0.4
dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_13_base_mu_04' 'log_id=13' 'op.miu_coef=[0.4, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_13_base_mu_04' 'log_id=13'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_14_admm_mu_04' 'log_id=14' 'op.miu_coef=[0.4, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_14_admm_mu_04' 'log_id=14'

# 0.5
dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_15_base_mu_05' 'log_id=15' 'op.miu_coef=[0.5, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_1_base' 'succ_grasp_dir=${save_dir}/succ_grasp_15_base_mu_05' 'log_id=15'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_16_admm_mu_05' 'log_id=16' 'op.miu_coef=[0.5, 0.0]'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'succ_grasp_dir=${save_dir}/succ_grasp_16_admm_mu_05' 'log_id=16'

# try parameter
dexrun op=grasp_admm 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_17_old_config' 'log_id=17'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_17_old_config' 'log_id=17'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_17_old_config' 'log_id=17'
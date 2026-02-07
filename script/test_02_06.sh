# dexsyn_admm exp_name=speed_admm_50 tmpl_name=[1_Large_Diameter,2_Small_Diameter,3_Medium_Wrap,4_Adducted_Thumb,5_Light_Tool,6_Prismatic_4_Finger,7_Prismatic_3_Finger,8_Prismatic_2_Finger] hand=shadow +init.object.n_cfg=100 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1 skip_grasp_eval=True +grasp_admm.grasp.square_res=True

# dexsyn exp_name=speed_dexonomy tmpl_name=[1_Large_Diameter,2_Small_Diameter,3_Medium_Wrap,4_Adducted_Thumb,5_Light_Tool,6_Prismatic_4_Finger,7_Prismatic_3_Finger,8_Prismatic_2_Finger] hand=shadow +init.object.n_cfg=100 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1 skip_grasp_eval=True

# dexsyn_admm exp_name=speed_admm_100 tmpl_name=[1_Large_Diameter,2_Small_Diameter,3_Medium_Wrap,4_Adducted_Thumb,5_Light_Tool,6_Prismatic_4_Finger,7_Prismatic_3_Finger,8_Prismatic_2_Finger] hand=shadow +init.object.n_cfg=100 +init_gpu=[0,1,2,3,4,5,6,7] +init.epoch=1 skip_grasp_eval=True +grasp_admm.grasp.square_res=True +grasp_admm.grasp.step=100

dexrun op=grasp_admm 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_24_config_02_07' 'log_id=24' 'op.grasp.square_res=True'

dexrun op=eval 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_24_config_02_07' 'succ_grasp_dir=${save_dir}/succ_grasp_24_config_02_07' 'log_id=24'

dexrun op=stat 'exp_name=test_100' 'hand=shadow' 'legacy_api=True' 'grasp_dir=${save_dir}/grasp_data_24_config_02_07' 'succ_grasp_dir=${save_dir}/succ_grasp_24_config_02_07' 'log_id=24'
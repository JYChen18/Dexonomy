exp_name=horse_mu_01
# source=objaverse
source=DGN_5k
obj_name=sem_Horse_de2eaf86f8ce18d68b3a450ac65b5b2d
# obj_name=sem_VideoGameController_6f790be92e6e44659bd828aa88bf7e51
# obj_name=core_pistol_41bca1dbde9fe5159220647403cfb896
# obj_name=2cc905713cf3471cb685a751c3e95774
# obj_name=216509ca06644f659c133e9ec77015fe
# obj_name=core_camera_21f65f53f74f1b58de8982fc28ddacc3
# obj_name=mujoco_PureFlow_2_Color_RylPurHibiscusBlkSlvrWht_Size_50
# obj_name=sem_TissueBox_ddf889fc432a0f869bc18b58d4f2bc4
asset_root=/mnt/home/ruanliangwang/Dexonomy-private/assets
template="[11_Power_Sphere,31_Ring,23_Adduction_Grip,25_Lateral_Tripod,29_Stick,8_Prismatic_2_Finger,13_Precision_Sphere,14_Tripod]"

dexsyn exp_name=${exp_name}_old tmpl_name=${template} hand=shadow +init.epoch=10 +init.object.cfg_path=${asset_root}/object/${source}/scene_cfg/${obj_name}/floating/*.npy init_gpu=[0,1,2,3,4,5,6,7] '+eval.miu_coef=[0.1,0.0]' new_tmpl_dir=output/test_100_shadow/new_tmpl tmpl_upd_mode=disabled

dexsyn_admm exp_name=${exp_name} tmpl_name=${template} hand=shadow +init.epoch=10 +init.object.cfg_path=${asset_root}/object/${source}/scene_cfg/${obj_name}/floating/*.npy init_gpu=[0,1,2,3,4,5,6,7] +grasp_admm.grasp.square_res=True '+eval.miu_coef=[0.1,0.0]' new_tmpl_dir=output/test_100_shadow/new_tmpl tmpl_upd_mode=disabled
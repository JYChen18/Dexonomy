# fingertip_small,fingertip_mid,fingertip_large
exp_name=tissuebox
hand=allegro
# tmpl_name=1_Large_Diameter
tmpl_name=fingertip_mid
# obj_name=sem_LightBulb_8338a18d589c26d21c648623457982d0
obj_name=sem_TissueBox_ddf889fc432a0f869bc18b58d4f2bc4
asset_root=/mnt/home/ruanliangwang/Dexonomy-private/assets
echo "${exp_name}_${hand}, tmpl_name=${tmpl_name}, obj_name=${obj_name}"

# dexrun op=init hand=$hand exp_name=$exp_name tmpl_name=$tmpl_name 'init_gpu=[0,1,2,3]' "op.object.cfg_path=${asset_root}/object/DGN_5k/scene_cfg/${obj_name}/floating/*.npy"
dexrun op=grasp_admm hand=$hand exp_name=$exp_name debug_name="scale005/3_4_grasp"
# dexrun op=eval hand=$hand exp_name=$exp_name
# dexrun op=v3d hand=$hand exp_name=$exp_name